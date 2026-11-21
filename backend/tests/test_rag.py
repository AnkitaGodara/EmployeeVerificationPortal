from types import SimpleNamespace

import pytest


def emp(level="L3", status="Active"):
    return SimpleNamespace(id=1, name="Asha", employee_code="ONX1", department="R&D", designation="Engineer",
                           access_level=level, status=status, shift="General", office_location="", manager="",
                           joining_date="", email="", notes="")


@pytest.mark.parametrize("question,level,status,expected", [
    ("Can she enter the server room?", "L3", "Active", "NOT ALLOWED"),
    ("Can she enter the server room?", "L4", "Active", "ALLOWED"),
    ("Is the R&D lab okay?", "L3", "Active", "ALLOWED"),
    ("Can she go to the lab?", "L3", "On Leave", "NOT ALLOWED"),
    ("Can she use the library?", "L2", "On Leave", "NEEDS APPROVAL"),
    ("Can she enter the lobby?", "L4", "Suspended", "NOT ALLOWED"),
])
def test_access_check(question, level, status, expected):
    from app.rag.assistant import access_check
    assert f"Decision: {expected}" in access_check(question, emp(level, status))


def test_access_check_ignores_unrelated_words():
    from app.rag.assistant import access_check
    assert access_check("what is on her label", emp()) is None


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_search_modes_return_policy_chunks(client, mode):
    from app.rag.index import rag_index
    hits = rag_index.search("How many sick leave days per year?", k=4, doc_type="policy", mode=mode)
    assert hits and all(d.metadata["type"] == "policy" for d, _ in hits)
    assert any("Leave" in d.metadata["title"] for d, _ in hits)


def test_ask_without_llm_is_extractive_with_sources(client, admin_headers):
    r = client.post("/api/assistant/ask", headers=admin_headers, json={"question": "How many sick leave days do we get?"})
    a = r.json()
    assert r.status_code == 200 and a["mode"] == "extractive"
    assert a["sources"] and "[S" in a["answer"]


def test_ask_with_llm_uses_context(client, admin_headers, monkeypatch):
    """The LLM path, with a fake chat model so the test needs no API key or network."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    import app.rag.assistant as assistant
    monkeypatch.setattr(assistant, "get_llm", lambda: FakeListChatModel(responses=["Employees get 10 sick days [S1]."]))
    monkeypatch.setattr(assistant, "provider_name", lambda: "fake")
    a = client.post("/api/assistant/ask", headers=admin_headers, json={"question": "sick leave days?"}).json()
    assert a["mode"] == "llm" and a["answer"].startswith("Employees get 10")


def test_llm_failure_falls_back(client, admin_headers, monkeypatch):
    import app.rag.assistant as assistant

    class Broken:
        def invoke(self, *_):
            raise ConnectionError("no network")
    monkeypatch.setattr(assistant, "get_llm", lambda: Broken())
    a = client.post("/api/assistant/ask", headers=admin_headers, json={"question": "sick leave days?"}).json()
    assert a["mode"] == "extractive" and "failed" in a["note"]


def test_upload_and_delete_document(client, admin_headers):
    files = {"file": ("canteen_policy.md", b"# Canteen Policy\n\n## Hours\nThe canteen is open from 8:00 to 22:00.", "text/markdown")}
    r = client.post("/api/documents", headers=admin_headers, files=files)
    assert r.status_code == 201
    a = client.post("/api/assistant/ask", headers=admin_headers, json={"question": "When is the canteen open?"}).json()
    assert any(s["source"] == "canteen_policy.md" for s in a["sources"])
    assert client.delete("/api/documents/canteen_policy.md", headers=admin_headers).status_code == 204
    assert client.delete("/api/documents/01_access_control_policy.md", headers=admin_headers).status_code == 404
