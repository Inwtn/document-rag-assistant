"""Response parsing and evidence policy, independent of the web framework."""

from dataclasses import dataclass
from typing import Any


ABSTENTION = "I could not find enough evidence in the indexed documents to answer that question."


@dataclass(frozen=True)
class Citation:
    file_id: str
    filename: str


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    grounded: bool


def _mapping(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    return response.model_dump(mode="json")


def parse_response(response: Any) -> Answer:
    """Expose an answer only when retrieval and file citations are both present."""
    payload = _mapping(response)
    search_results: set[str] = set()
    citations: dict[str, Citation] = {}
    answer_parts: list[str] = []

    for item in payload.get("output") or []:
        if item.get("type") == "file_search_call":
            for result in item.get("results") or []:
                file_id = result.get("file_id")
                if file_id:
                    search_results.add(file_id)
        elif item.get("type") == "message" and item.get("role") == "assistant":
            for content in item.get("content") or []:
                if content.get("type") != "output_text":
                    continue
                answer_parts.append(content.get("text") or "")
                for annotation in content.get("annotations") or []:
                    if annotation.get("type") == "file_citation":
                        file_id = annotation.get("file_id")
                        filename = annotation.get("filename")
                        if file_id and filename:
                            citations[file_id] = Citation(file_id, filename)

    grounded_citations = [citation for file_id, citation in citations.items() if file_id in search_results]
    answer_text = "\n".join(part.strip() for part in answer_parts if part.strip()).strip()
    if not search_results or not grounded_citations or not answer_text:
        return Answer(ABSTENTION, [], False)
    return Answer(answer_text, grounded_citations, True)

