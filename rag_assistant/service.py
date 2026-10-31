"""OpenAI Responses API adapter."""

import os
from typing import Any

from .core import Answer, parse_response


SYSTEM_INSTRUCTIONS = (
    "Answer only from the indexed documents found by file_search. "
    "Cite the source file for every factual claim. If the documents do not support "
    "an answer, say you do not know. Do not use outside knowledge or invent citations."
)


class ConfigurationError(Exception):
    pass


class DocumentAssistant:
    def __init__(self, client: Any | None = None, vector_store_id: str | None = None, model: str | None = None):
        self.vector_store_id = vector_store_id or os.getenv("OPENAI_VECTOR_STORE_ID", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-astra")
        self.client = client

    def ask(self, question: str) -> Answer:
        if not self.vector_store_id:
            raise ConfigurationError("Set OPENAI_VECTOR_STORE_ID after indexing the documents.")
        if self.client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ConfigurationError("Set OPENAI_API_KEY to call the OpenAI API.")
            from openai import OpenAI

            self.client = OpenAI()
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=question,
            tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id], "max_num_results": 5}],
            include=["file_search_call.results"],
            store=False,
        )
        return parse_response(response)

