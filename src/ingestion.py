"""Build (or rebuild) the T20 rules vector index from a PDF.

Usage:
    python -m src.ingestion
    python -m src.ingestion --pdf knowledgebase/T20_rules.pdf --out vectorstore/t20_rules_index.json
"""
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PDF_PATH, VECTORSTORE_PATH, get_embeddings


def build_index(pdf_path: str = PDF_PATH, out_path: str = VECTORSTORE_PATH) -> int:
    """Load a PDF, split it into chunks, embed it, and persist the vector store."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'. Place your T20 rules PDF there, or pass --pdf."
        )

    documents = PyPDFLoader(str(pdf_file)).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    store = InMemoryVectorStore(get_embeddings())
    store.add_documents(chunks)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(out_file))

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a T20 rules PDF into the vector store.")
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to the source PDF file.")
    parser.add_argument("--out", default=VECTORSTORE_PATH, help="Output path for the vector index.")
    args = parser.parse_args()

    count = build_index(args.pdf, args.out)
    print(f"Indexed {count} chunk(s) from '{args.pdf}' into '{args.out}'.")


if __name__ == "__main__":
    main()
