"""HTTP 接口端到端：正常流程 + 可区分错误码 + 取消 + 恢复。"""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

from market_risk.api import create_app
from market_risk.engine import RiskEngine

warnings.filterwarnings("ignore")


@pytest.fixture
def client(tmp_path):
    eng = RiskEngine(tmp_path / "data", max_concurrency=4)
    c = TestClient(create_app(eng))
    yield c, eng
    eng.close()


def _history_payload(n=30, inst="A"):
    return {
        "observations": [
            {"instrument": inst, "day": f"2025-01-{d:02d}",
             "price": 100.0 + d}
            for d in range(1, n + 1)
        ],
        "as_of": f"2025-01-{n:02d}",
    }


def test_full_flow(client):
    c, eng = client
    r = c.post("/api/v1/history", json=_history_payload(30))
    assert r.status_code == 200 and r.json()["data_version"] == 1
    r = c.post("/api/v1/portfolios", json={
        "portfolio_id": "p1",
        "positions": [{"instrument": "A", "quantity": 2.0}]})
    assert r.status_code == 200
    r = c.post("/api/v1/calculations", json={
        "portfolio_id": "p1",
        "config": {"confidence": 0.95, "window": 10}})
    tid = r.json()["task_id"]
    r = c.get(f"/api/v1/results/{tid}?timeout=10")
    assert r.status_code == 200
    body = r.json()
    assert body["sample_size"] == 9 and body["data_version"] == 1
    assert "var" in body and "es" in body and body["method"] == "ROLLING"
    # 状态查询
    st = c.get(f"/api/v1/tasks/{tid}").json()
    assert st["state"] == "COMMITTED"


def test_error_codes_distinguishable(client):
    c, _ = client
    # 空批次
    r = c.post("/api/v1/history", json={"observations": []})
    assert r.status_code == 400
    assert r.json()["error_code"] == "MISSING_PRICE"
    # 非正价格
    r = c.post("/api/v1/history", json={"observations": [
        {"instrument": "A", "day": "2025-01-01", "price": -3}]})
    assert r.json()["error_code"] == "NON_POSITIVE_PRICE"
    # 迟到
    r = c.post("/api/v1/history", json={
        "observations": [{"instrument": "A",
                          "day": "2025-02-01", "price": 10}],
        "as_of": "2025-01-01"})
    assert r.json()["error_code"] == "LATE_OBSERVATION"
    # 非法置信水平
    r = c.post("/api/v1/history", json=_history_payload(5))
    c.post("/api/v1/portfolios", json={"portfolio_id": "p1",
            "positions": [{"instrument": "A", "quantity": 1}]})
    r = c.post("/api/v1/calculations", json={
        "portfolio_id": "p1",
        "config": {"confidence": 2.0, "window": 3}})
    # pydantic 层 confidence=2.0 通过 float 校验（值域由引擎校验）
    if r.status_code == 200:
        tid = r.json()["task_id"]
        gr = c.get(f"/api/v1/results/{tid}?timeout=5")
        assert gr.status_code == 400
        assert gr.json()["error_code"] == "INVALID_PARAMETER"
    else:
        assert r.status_code == 400


def test_duplicate_batch_rejected_no_version_change(client):
    c, _ = client
    c.post("/api/v1/history", json=_history_payload(10))
    r = c.post("/api/v1/history", json=_history_payload(10))
    assert r.status_code == 400
    assert r.json()["error_code"] == "DUPLICATE_OBSERVATION"
    assert c.get("/api/v1/health").json()["data_version"] == 1


def test_unknown_task_404(client):
    c, _ = client
    assert c.get("/api/v1/tasks/does-not-exist").status_code == 404


def test_recover_endpoint(client, tmp_path):
    c, eng = client
    c.post("/api/v1/history", json=_history_payload(20))
    c.post("/api/v1/portfolios", json={"portfolio_id": "p1",
            "positions": [{"instrument": "A", "quantity": 1}]})
    eng.close()
    # 新进程新 app 指向同一路径
    eng2 = RiskEngine(tmp_path / "data", max_concurrency=2)
    c2 = TestClient(create_app(eng2))
    assert c2.post("/api/v1/recover").json()["requeued"] == 0
    eng2.close()
