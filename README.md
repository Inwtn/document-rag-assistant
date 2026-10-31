# Document RAG Assistant

A document question-answering service built with FastAPI and the OpenAI Responses API. It searches an OpenAI vector store and returns an answer only when it includes a citation to a retrieved file.

## Features

- Hosted file search over Markdown, text, and PDF documents
- Source citations with grounded answers
- Abstention when retrieval or matching citations are missing
- Browser interface, HTTP API, and explicit document indexing CLI
- Automated tests and GitHub Actions CI without API credentials

## Run locally

Requires Python 3.11 or newer and an OpenAI API project for live retrieval.

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    pip install -e ".[test]"

Set OPENAI_API_KEY in your shell. The .env.example file lists settings; .env files are not loaded automatically.

    python -m rag_assistant.ingest sample_docs --name "RAG sample"

Set OPENAI_VECTOR_STORE_ID to the printed ID. OPENAI_MODEL is configurable and defaults to gpt-6-astra. Then start the app:

    uvicorn rag_assistant.api:app --reload

Open http://127.0.0.1:8000 or the API documentation at http://127.0.0.1:8000/docs. Ask: "What is the refund window?"

POST /ask accepts an object containing a question string. GET /health returns service status; it does not check upstream API access.

## Tests

    pytest -q

Tests cover evidence checks, fabricated citation rejection, API validation, and the provider request contract. They use mocked responses and do not prove live retrieval quality. Live indexing and generation require API access and may incur charges.

## Architecture and limitations

Browser -> FastAPI -> OpenAI Responses API with file_search -> vector store.

The parser verifies that a cited file was returned by search. This evidence check does not prove that every generated claim is supported. Review citations and evaluate against your own documents before production use. The included document is fictional demo data.

This is a local portfolio demo with no authentication or per-user document isolation. Indexing uploads documents to OpenAI; indexed files and stores should be removed through the API when no longer needed. Each indexing command creates a new store. The app does not upload files automatically.

Reference: [OpenAI File Search documentation](https://developers.openai.com/api/docs/guides/tools-file-search).

## License

MIT
