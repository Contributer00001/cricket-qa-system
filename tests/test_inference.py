"""
Integration smoke-tests for the FastAPI service.

These tests use the no-LLM path (use_llm=False, default) so no model
download is needed in CI.
"""
from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)

SAMPLE_COMMENTARY = {
    "commentaries": [
        {"event": "ball",   "over": 1, "run": 4, "four": True,  "six": False},
        {"event": "ball",   "over": 1, "run": 1, "four": False, "six": False},
        {"event": "ball",   "over": 1, "run": 0, "four": False, "six": False},
        {"event": "wicket", "over": 2, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 2, "run": 6, "four": False, "six": True},
    ]
}


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_before_model_load():
    r = client.get("/readyz")
    assert r.status_code == 200
    assert "ready" in r.json()


def test_infer_total_runs():
    r = client.post("/infer", json={
        "question": "How many total runs were scored?",
        "commentary": SAMPLE_COMMENTARY,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["answer"] == "11"
    assert body["intent"] == "total_runs"


def test_infer_total_wickets():
    r = client.post("/infer", json={
        "question": "How many wickets fell?",
        "commentary": SAMPLE_COMMENTARY,
    })
    assert r.status_code == 200
    assert r.json()["answer"] == "1"


def test_infer_last_n_overs():
    r = client.post("/infer", json={
        "question": "How many runs were scored in the last 1 overs?",
        "commentary": SAMPLE_COMMENTARY,
    })
    assert r.status_code == 200
    assert r.json()["answer"] == "6"


def test_infer_uses_default_sample_data_when_no_commentary():
    r = client.post("/infer", json={"question": "How many total runs were scored?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["answer"] == "47"


def test_infer_question_too_long():
    r = client.post("/infer", json={"question": "x" * 501})
    assert r.status_code == 422


def test_infer_missing_question_field():
    r = client.post("/infer", json={"commentary": SAMPLE_COMMENTARY})
    assert r.status_code == 422


def test_infer_empty_commentary_falls_back_to_sample():
    # Empty commentaries list → service falls back to default sample data
    r = client.post("/infer", json={
        "question": "How many total runs were scored?",
        "commentary": {"commentaries": []},
    })
    assert r.status_code == 200
    assert r.json()["answer"] == "47"  # sample_match.json total
