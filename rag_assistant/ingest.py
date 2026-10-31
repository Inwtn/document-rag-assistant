"""Explicitly index local documents into an OpenAI vector store."""

import argparse
import os
from pathlib import Path


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def index_directory(directory: Path, name: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before indexing.")
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise ValueError("No .md, .txt, or .pdf files found.")

    from openai import OpenAI

    client = OpenAI()
    store = client.vector_stores.create(name=name)
    for path in files:
        with path.open("rb") as stream:
            uploaded = client.vector_stores.files.upload_and_poll(vector_store_id=store.id, file=stream)
        if uploaded.status != "completed":
            raise RuntimeError(f"Indexing failed for {path.name}: {uploaded.status}")
        print(f"Indexed {path.name}")
    return store.id


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents for the RAG assistant")
    parser.add_argument("directory", type=Path, help="Directory containing .md, .txt, or .pdf files")
    parser.add_argument("--name", default="Document RAG demo", help="Vector store name")
    args = parser.parse_args()
    store_id = index_directory(args.directory, args.name)
    print(f"OPENAI_VECTOR_STORE_ID={store_id}")


if __name__ == "__main__":
    main()
