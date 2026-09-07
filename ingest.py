"""
ingest.py — Indicizzazione documenti nel vector store Chroma.

Supporta:
  - PDF locali
  - URL / pagine web
  - File di testo (.txt, .md, .csv)

Utilizzo:
  python ingest.py --pdf path/to/doc.pdf
  python ingest.py --url https://example.com
  python ingest.py --text path/to/file.txt
  python ingest.py --dir path/to/folder/   (indicizza tutti i PDF nella cartella)
"""

import argparse
import sys
import codecs

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rich.console import Console
from rich.progress import track

import config

console = Console()


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL
    )


def get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# ── Loaders ─────────────────────────────────────────────────────────────────

def ingest_pdf(path: str):
    console.print(f"[cyan]📄 Carico PDF:[/] {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    _ingest(docs, source=path)


def ingest_url(url: str):
    console.print(f"[cyan]🌐 Carico URL:[/] {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    _ingest(docs, source=url)


def ingest_text(path: str):
    console.print(f"[cyan]📝 Carico file testo:[/] {path}")
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    _ingest(docs, source=path)


def ingest_directory(dir_path: str):
    console.print(f"[cyan]📁 Carico cartella PDF:[/] {dir_path}")
    loader = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    _ingest(docs, source=dir_path)


def _ingest(docs, source: str):
    if not docs:
        console.print("[red]⚠ Nessun documento trovato.[/]")
        return

    chunks = split_documents(docs)
    console.print(f"  → {len(docs)} documento/i → {len(chunks)} chunk")

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    for chunk in track(chunks, description="Indicizzando..."):
        pass  # track è solo per il progresso visivo

    vector_store.add_documents(chunks)
    console.print(f"[green]✅ Indicizzati {len(chunks)} chunk da:[/] {source}")
    console.print(f"   Vector store salvato in: [dim]{config.CHROMA_DIR}[/]")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Ingestor — indicizza documenti in Chroma")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",  metavar="PATH", help="Percorso a un file PDF")
    group.add_argument("--url",  metavar="URL",  help="URL di una pagina web")
    group.add_argument("--text", metavar="PATH", help="Percorso a un file .txt/.md")
    group.add_argument("--dir",  metavar="PATH", help="Cartella con PDF da indicizzare")

    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        console.print("[bold red]Errore:[/] GROQ_API_KEY non impostata. Crea il file .env partendo da .env.example")
        sys.exit(1)

    if args.pdf:
        ingest_pdf(args.pdf)
    elif args.url:
        ingest_url(args.url)
    elif args.text:
        ingest_text(args.text)
    elif args.dir:
        ingest_directory(args.dir)


if __name__ == "__main__":
    main()
