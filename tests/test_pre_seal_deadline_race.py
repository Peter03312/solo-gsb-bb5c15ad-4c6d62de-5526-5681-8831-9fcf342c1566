"""预封存“截止判定以取得写锁后的实际时刻为准”竞态测试。

三类同类竞态（请求在截止前发起，BEGIN IMMEDIATE 等锁期间越过截止）：
- 发起：取得锁时截止已过 -> 沿用现有 400，且不落任何提案；
- 确认（第三票）：取得锁时已过 -> 先使提案失效并清空正文（提交），
  再沿用现有 409，不记新票、不生成资料或封存审计；
- 撤回：取得锁时已过 -> 同上，提案 expired 而非 revoked。

另覆盖：
- 截止前取得锁的有效第三票仍在同一事务生成唯一资料，且资料/提案的
  完成时刻取“锁内实际时刻”，不会记下过早时间；
- 轮换先提交（在持锁事务内完成）时，旧密钥的第三票在锁内复核即 401
  并整体回滚：提案不被连带失效、正文保留、无资料无新票。

为确定性复现“等锁越过截止”：测试线程持有一把 BEGIN IMMEDIATE 写锁，
端点在另一线程的独立连接上阻塞，期间把可控时钟拨过截止时刻后再放锁。
"""
import datetime as dt
import os
import sqlite3
import sys
import threading

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import auth, digest_of, register  # noqa: E402


T0 = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
DEADLINE = T0 + dt.timedelta(hours=1)
AFTER_DEADLINE = DEADLINE + dt.timedelta(seconds=1)
BEFORE_DEADLINE_LATER = T0 + dt.timedelta(seconds=10)

BODY = "锁等待越过截止的正文 DEADLINE-LOCK-SECRET"
WITNESSES = ["dl-bob", "dl-carol", "dl-dave"]


@pytest.fixture
def five_users(client):
    return {n: register(client, n) for n in
            ["dl-alice", "dl-bob", "dl-carol", "dl-dave", "dl-erin"]}


@pytest.fixture
def frozen_clock(monkeypatch):
    """可控时钟：替换 app.main.now_utc；返回可变时刻对象。"""
    from app import main

    state = {"now": T0}

    def clock():
        return state["now"]

    monkeypatch.setattr(main, "now_utc", clock)
    return state


@pytest.fixture
def silent_sweeper(monkeypatch):
    """停掉后台扫描器的到期写入：持锁期间到期只能由被阻塞的请求自行结算。"""
    from app import main

    monkeypatch.setattr(
        main.ExpirySweeper, "sweep_once", staticmethod(lambda *a, **k: 0)
    )


def _db_rows(query, args=()):
    conn = sqlite3.connect(os.environ["SEAL_DB_PATH"])
    try:
        return conn.execute(query, args).fetchall()
    finally:
        conn.close()


def _auth_user(key):
    from app.main import AuthUser, sha256_hex
    from app.db import connect

    conn = connect()
    row = conn.execute(
        "SELECT * FROM users WHERE key_hash = ?", (sha256_hex(key),)
    ).fetchone()
    assert row is not None
    return conn, AuthUser(row, key)


def _start_worker(fn):
    box: dict = {}

    def run():
        try:
            box["result"] = fn()
        except HTTPException as exc:
            box["http"] = exc
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, box


def _create_proposal(client, keys, deadline):
    r = client.post(
        "/pre-seal-proposals",
        json={"body": BODY, "witnesses": WITNESSES,
              "confirm_deadline": deadline.isoformat()},
        headers=auth(keys["dl-alice"]),
    )
    assert r.status_code == 201, r.text
    return r.json()["proposal_id"]


def _confirm_direct(key, proposal_id, dg):
    from app.main import confirm_pre_seal_proposal
    from app.schemas import PreSealConfirmIn

    conn, user = _auth_user(key)
    try:
        return confirm_pre_seal_proposal(
            proposal_id, PreSealConfirmIn(digest=dg), db=conn, user=user
        )
    finally:
        conn.close()


def _revoke_direct(key, proposal_id):
    from app.main import revoke_pre_seal_proposal

    conn, user = _auth_user(key)
    try:
        return revoke_pre_seal_proposal(proposal_id, db=conn, user=user)
    finally:
        conn.close()


def _create_direct(key, deadline):
    from app.main import create_pre_seal_proposal
    from app.schemas import PreSealIn

    conn, user = _auth_user(key)
    try:
        return create_pre_seal_proposal(
            PreSealIn(body=BODY, witnesses=WITNESSES, confirm_deadline=deadline),
            db=conn,
            user=user,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 发起：等锁越过截止 -> 400，不落提案
# ---------------------------------------------------------------------------


def test_create_waiting_past_deadline_rejected_without_row(
    client, five_users, frozen_clock, silent_sweeper
):
    from app.db import connect

    blocker = connect()
    blocker.execute("BEGIN IMMEDIATE")  # 占位写锁，让发起请求在 BEGIN IMMEDIATE 等待
    try:
        thread, box = _start_worker(
            lambda: _create_direct(five_users["dl-alice"], DEADLINE)
        )
        thread.join(timeout=0.6)
        assert thread.is_alive(), "发起请求应仍在等待写锁"
        # 等锁期间越过截止时刻
        frozen_clock["now"] = AFTER_DEADLINE
    finally:
        blocker.commit()
        blocker.close()
    thread.join(timeout=5)

    exc = box.get("http")
    assert exc is not None and exc.status_code == 400, box
    # 沿用现有 400 文案，且提案完全不落库
    assert exc.detail == "确认截止时刻必须晚于当前时间"
    assert _db_rows("SELECT COUNT(*) FROM pre_seal_proposals")[0][0] == 0
    assert _db_rows("SELECT COUNT(*) FROM pre_seal_witnesses")[0][0] == 0


# ---------------------------------------------------------------------------
# 第三票确认：等锁越过截止 -> 先失效（提交）再 409，不建档/不记票
# ---------------------------------------------------------------------------


def test_third_vote_waiting_past_deadline_expires_then_409(
    client, five_users, frozen_clock, silent_sweeper
):
    dg = digest_of(BODY)
    pid = _create_proposal(client, five_users, DEADLINE)
    assert client.post(f"/pre-seal-proposals/{pid}/confirmations",
                       json={"digest": dg}, headers=auth(five_users["dl-bob"])).status_code == 200
    assert client.post(f"/pre-seal-proposals/{pid}/confirmations",
                       json={"digest": dg},
                       headers=auth(five_users["dl-carol"])).status_code == 200

    from app.db import connect

    blocker = connect()
    blocker.execute("BEGIN IMMEDIATE")
    try:
        thread, box = _start_worker(
            lambda: _confirm_direct(five_users["dl-dave"], pid, dg)
        )
        thread.join(timeout=0.6)
        assert thread.is_alive(), "第三票应仍在等待写锁"
        frozen_clock["now"] = AFTER_DEADLINE
    finally:
        blocker.commit()
        blocker.close()
    thread.join(timeout=5)

    exc = box.get("http")
    assert exc is not None and exc.status_code == 409, box
    # 提案已失效、正文清空、仍是两票；无资料、无 seal 审计
    status, body = _db_rows(
        "SELECT status, body FROM pre_seal_proposals WHERE id=?", (pid,)
    )[0]
    assert status == "expired" and body == ""
    assert _db_rows(
        "SELECT COUNT(*) FROM pre_seal_confirmations WHERE proposal_id=?", (pid,)
    )[0][0] == 2
    assert _db_rows("SELECT COUNT(*) FROM documents")[0][0] == 0
    assert _db_rows("SELECT COUNT(*) FROM audit_log WHERE action='seal'")[0][0] == 0


# ---------------------------------------------------------------------------
# 撤回：等锁越过截止 -> 提案 expired（不是 revoked）并 409
# ---------------------------------------------------------------------------


def test_revoke_waiting_past_deadline_expires_then_409(
    client, five_users, frozen_clock, silent_sweeper
):
    dg = digest_of(BODY)
    pid = _create_proposal(client, five_users, DEADLINE)
    assert client.post(f"/pre-seal-proposals/{pid}/confirmations",
                       json={"digest": dg}, headers=auth(five_users["dl-bob"])).status_code == 200

    from app.db import connect

    blocker = connect()
    blocker.execute("BEGIN IMMEDIATE")
    try:
        thread, box = _start_worker(
            lambda: _revoke_direct(five_users["dl-alice"], pid)
        )
        thread.join(timeout=0.6)
        assert thread.is_alive(), "撤回应仍在等待写锁"
        frozen_clock["now"] = AFTER_DEADLINE
    finally:
        blocker.commit()
        blocker.close()
    thread.join(timeout=5)

    exc = box.get("http")
    assert exc is not None and exc.status_code == 409, box
    status, body = _db_rows(
        "SELECT status, body FROM pre_seal_proposals WHERE id=?", (pid,)
    )[0]
    assert status == "expired" and body == ""
    assert _db_rows("SELECT COUNT(*) FROM documents")[0][0] == 0


# ---------------------------------------------------------------------------
# 截止前取得锁：有效第三票同事务生成唯一资料，时刻取锁内实际时刻
# ---------------------------------------------------------------------------


def test_third_vote_gets_lock_before_deadline_uses_lock_time(
    client, five_users, frozen_clock, silent_sweeper
):
    from app.db import utc_str

    dg = digest_of(BODY)
    pid = _create_proposal(client, five_users, DEADLINE)
    assert client.post(f"/pre-seal-proposals/{pid}/confirmations",
                       json={"digest": dg}, headers=auth(five_users["dl-bob"])).status_code == 200
    assert client.post(f"/pre-seal-proposals/{pid}/confirmations",
                       json={"digest": dg},
                       headers=auth(five_users["dl-carol"])).status_code == 200

    from app.db import connect

    blocker = connect()
    blocker.execute("BEGIN IMMEDIATE")
    try:
        thread, box = _start_worker(
            lambda: _confirm_direct(five_users["dl-dave"], pid, dg)
        )
        thread.join(timeout=0.6)
        assert thread.is_alive()
        # 只推进到截止之前的某时刻：锁内时刻才是有效判定与建档时刻
        frozen_clock["now"] = BEFORE_DEADLINE_LATER
    finally:
        blocker.commit()
        blocker.close()
    thread.join(timeout=5)

    assert "error" not in box, box["error"]
    data = box["result"]
    assert data["status"] == "confirmed" and data["confirmation_count"] == 3
    doc_id = data["document_id"]
    assert isinstance(doc_id, int)
    # 唯一资料；资料/提案完成时刻必须等于锁内实际时刻（而非发起时的 T0）
    assert _db_rows("SELECT COUNT(*) FROM documents")[0][0] == 1
    created_at, = _db_rows("SELECT created_at FROM documents WHERE id=?", (doc_id,))[0]
    assert created_at == utc_str(BEFORE_DEADLINE_LATER)
    assert created_at != utc_str(T0)
    resolved_at, status, body = _db_rows(
        "SELECT resolved_at, status, body FROM pre_seal_proposals WHERE id=?", (pid,)
    )[0]
    assert resolved_at == utc_str(BEFORE_DEADLINE_LATER)
    assert status == "confirmed" and body == ""
    assert _db_rows(
        "SELECT COUNT(*) FROM audit_log WHERE document_id=? AND action='seal'", (doc_id,)
    )[0][0] == 1


# ---------------------------------------------------------------------------
# 轮换先提交：旧密钥第三票锁内复核 401，整体回滚（不连带失效提案）
# ---------------------------------------------------------------------------


def test_rotation_committed_first_during_lock_wait_401_and_rolls_back(
    client, five_users, frozen_clock, silent_sweeper
):
    from app.main import sha256_hex
    from app.db import connect

    dg = digest_of(BODY)
    pid = _create_proposal(client, five_users, DEADLINE)
    assert client.post(f"/pre-seal-proposals/{pid}/confirmations",
                       json={"digest": dg}, headers=auth(five_users["dl-bob"])).status_code == 200
    assert client.post(f"/pre-seal-proposals/{pid}/confirmations",
                       json={"digest": dg},
                       headers=auth(five_users["dl-carol"])).status_code == 200

    old_dave = five_users["dl-dave"]
    import secrets
    new_key = secrets.token_hex(32)

    blocker = connect()
    blocker.execute("BEGIN IMMEDIATE")
    try:
        thread, box = _start_worker(
            lambda: _confirm_direct(old_dave, pid, dg)
        )
        thread.join(timeout=0.6)
        assert thread.is_alive(), "旧密钥第三票应仍在等待写锁"
        # 等锁期间时刻越过截止；持锁事务先把 dave 的密钥轮换提交
        frozen_clock["now"] = AFTER_DEADLINE
        blocker.execute(
            "UPDATE users SET key_hash=? WHERE id="
            "(SELECT id FROM users WHERE username='dl-dave')",
            (sha256_hex(new_key),),
        )
    finally:
        blocker.commit()  # 轮换先提交，随后第三票才能取得锁
        blocker.close()
    thread.join(timeout=5)

    exc = box.get("http")
    assert exc is not None and exc.status_code == 401, box
    assert exc.detail == "凭据无效"
    # 整体回滚：第三票不记、不生成资料；也没有借该请求把提案失效
    status, body = _db_rows(
        "SELECT status, body FROM pre_seal_proposals WHERE id=?", (pid,)
    )[0]
    assert status == "pending" and body == BODY
    assert _db_rows(
        "SELECT COUNT(*) FROM pre_seal_confirmations WHERE proposal_id=?", (pid,)
    )[0][0] == 2
    assert _db_rows("SELECT COUNT(*) FROM documents")[0][0] == 0
    # 新密钥随后可正常投第三票（截止仍有效）：把时刻拨回截止前
    frozen_clock["now"] = BEFORE_DEADLINE_LATER
    r = client.post(f"/pre-seal-proposals/{pid}/confirmations",
                    json={"digest": dg}, headers=auth(new_key))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "confirmed"
