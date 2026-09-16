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
  python ingest.py --pdf doc.pdf --force    (salta controllo duplicati)
"""

import argparse
import sys
import codecs
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from datetime import datetime
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
from logging_config import get_logger

console = Console()
logger = get_logger(__name__)


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


def _check_duplicates(vector_store: Chroma, source: str) -> bool:
    """Controlla se una fonte è già presente nel vector store.
    Restituisce True se ci sono duplicati."""
    try:
        collection = vector_store._collection
        results = collection.get(where={"source": source})
        if results and results["ids"]:
            return True
    except Exception:
        pass
    return False


def _enrich_metadata(chunks, source_type: str):
    """Aggiunge metadati extra a ogni chunk."""
    timestamp = datetime.now().isoformat()
    for chunk in chunks:
        chunk.metadata["ingested_at"] = timestamp
        chunk.metadata["source_type"] = source_type
    return chunks


# ── Loaders ─────────────────────────────────────────────────────────────────

def ingest_pdf(path: str, force: bool = False):
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Errore:[/] il file non esiste: {path}")
        logger.error(f"File non trovato: {path}")
        return
    if not p.suffix.lower() == ".pdf":
        console.print(f"[bold red]Errore:[/] il file non è un PDF: {path}")
        return

    console.print(f"[cyan]📄 Carico PDF:[/] {path}")
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
    except Exception as e:
        console.print(f"[bold red]Errore nel caricamento del PDF:[/] {e}")
        logger.error(f"Errore caricamento PDF {path}: {e}")
        return
    _ingest(docs, source=str(p.resolve()), source_type="pdf", force=force)


def ingest_url(url: str, force: bool = False):
    console.print(f"[cyan]🌐 Carico URL:[/] {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
    except Exception as e:
        console.print(f"[bold red]Errore nel caricamento dell'URL:[/] {e}")
        logger.error(f"Errore caricamento URL {url}: {e}")
        return
    _ingest(docs, source=url, source_type="url", force=force)


def ingest_text(path: str, force: bool = False):
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Errore:[/] il file non esiste: {path}")
        logger.error(f"File non trovato: {path}")
        return

    console.print(f"[cyan]📝 Carico file testo:[/] {path}")
    try:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    except Exception as e:
        console.print(f"[bold red]Errore nel caricamento del file:[/] {e}")
        logger.error(f"Errore caricamento testo {path}: {e}")
        return
    _ingest(docs, source=str(p.resolve()), source_type="text", force=force)


def ingest_directory(dir_path: str, force: bool = False):
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        console.print(f"[bold red]Errore:[/] la cartella non esiste: {dir_path}")
        logger.error(f"Cartella non trovata: {dir_path}")
        return

    console.print(f"[cyan]📁 Carico cartella PDF:[/] {dir_path}")
    try:
        loader = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
    except Exception as e:
        console.print(f"[bold red]Errore nel caricamento della cartella:[/] {e}")
        logger.error(f"Errore caricamento cartella {dir_path}: {e}")
        return
    _ingest(docs, source=str(p.resolve()), source_type="directory", force=force)


def _ingest(docs, source: str, source_type: str, force: bool = False):
    if not docs:
        console.print("[red]⚠ Nessun documento trovato.[/]")
        return

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    # Controllo duplicati
    if not force and _check_duplicates(vector_store, source):
        console.print(
            f"[yellow]⚠ Questa fonte è già indicizzata:[/] {source}\n"
            f"  Usa [bold]--force[/] per re-indicizzare."
        )
        logger.info(f"Duplicato rilevato, skip: {source}")
        return

    chunks = split_documents(docs)
    chunks = _enrich_metadata(chunks, source_type)
    console.print(f"  → {len(docs)} documento/i → {len(chunks)} chunk")

    start = time.time()
    for chunk in track(chunks, description="Indicizzando..."):
        pass  # track è solo per il progresso visivo

    vector_store.add_documents(chunks)
    elapsed = time.time() - start

    # Conteggio totale nel DB
    try:
        total = vector_store._collection.count()
    except Exception:
        total = "?"

    console.print(f"[green]✅ Indicizzati {len(chunks)} chunk da:[/] {source}")
    console.print(f"   Tempo: [dim]{elapsed:.1f}s[/] — Totale nel DB: [bold]{total}[/] chunk")
    console.print(f"   Vector store: [dim]{config.CHROMA_DIR}[/]")
    logger.info(f"Indicizzati {len(chunks)} chunk da {source} in {elapsed:.1f}s (totale DB: {total})")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Ingestor — indicizza documenti in Chroma")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",  metavar="PATH", help="Percorso a un file PDF")
    group.add_argument("--url",  metavar="URL",  help="URL di una pagina web")
    group.add_argument("--text", metavar="PATH", help="Percorso a un file .txt/.md")
    group.add_argument("--dir",  metavar="PATH", help="Cartella con PDF da indicizzare")
    parser.add_argument("--force", action="store_true",
                        help="Forza re-indicizzazione anche se la fonte è già presente")

    args = parser.parse_args()

    if args.pdf:
        ingest_pdf(args.pdf, force=args.force)
    elif args.url:
        ingest_url(args.url, force=args.force)
    elif args.text:
        ingest_text(args.text, force=args.force)
    elif args.dir:
        ingest_directory(args.dir, force=args.force)


if __name__ == "__main__":
    main()
