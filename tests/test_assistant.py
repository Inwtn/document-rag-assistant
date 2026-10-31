from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.core import ABSTENTION, parse_response
from rag_assistant.service import DocumentAssistant


def response(cited_id="f1", results=True):
    return {"output": [
        {"type": "file_search_call", "results": [{"file_id": "f1"}] if results else []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The refund window is 30 days.", "annotations": [{"type": "file_citation", "file_id": cited_id, "filename": "policy.md"}]}]},
    ]}


def test_grounded_answer_requires_retrieved_citation():
    answer = parse_response(response())
    assert answer.grounded
    assert answer.citations[0].filename == "policy.md"
    assert "30 days" in answer.answer


def test_abstains_without_retrieval():
    answer = parse_response(response(results=False))
    assert not answer.grounded
    assert answer.answer == ABSTENTION


def test_abstains_for_unretrieved_citation():
    assert not parse_response(response(cited_id="invented-file")).grounded


def test_service_requests_evidence_and_disables_response_storage():
    create = Mock(return_value=response())
    assistant = DocumentAssistant(client=SimpleNamespace(responses=SimpleNamespace(create=create)), vector_store_id="vs_test", model="test-model")
    assert assistant.ask("What is the refund window?").grounded
    kwargs = create.call_args.kwargs
    assert kwargs["store"] is False
    assert kwargs["include"] == ["file_search_call.results"]
    assert kwargs["tools"][0]["vector_store_ids"] == ["vs_test"]


def test_api_validation_and_citations(monkeypatch):
    monkeypatch.setattr(api, "assistant", SimpleNamespace(ask=lambda _: parse_response(response())))
    client = TestClient(api.app)
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    result = client.post("/ask", json={"question": "What is the refund window?"})
    assert result.status_code == 200
    assert result.json()["citations"][0]["file_id"] == "f1"
    assert client.get("/").status_code == 200


def test_missing_configuration_is_actionable(monkeypatch):
    monkeypatch.setattr(api, "assistant", DocumentAssistant(vector_store_id=""))
    monkeypatch.delenv("OPENAI_VECTOR_STORE_ID", raising=False)
    result = TestClient(api.app).post("/ask", json={"question": "Where is the policy?"})
    assert result.status_code == 503
