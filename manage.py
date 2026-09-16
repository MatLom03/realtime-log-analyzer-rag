"""
manage.py — Gestione del vector store Chroma.

Utilizzo:
  python manage.py stats                    # Statistiche del DB
  python manage.py list                     # Lista fonti indicizzate
  python manage.py reset                    # Svuota il vector store
  python manage.py delete --source "file"   # Rimuovi una fonte specifica
"""

import argparse
import sys
import codecs

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import shutil
from collections import Counter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import config
from logging_config import get_logger

console = Console()
logger = get_logger(__name__)


def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def cmd_stats():
    """Mostra le statistiche del vector store."""
    vs = get_vector_store()
    try:
        collection = vs._collection
        count = collection.count()
    except Exception as e:
        console.print(f"[bold red]Errore nell'accesso al DB:[/] {e}")
        return

    # Dimensione su disco
    db_path = config.CHROMA_DIR
    if db_path.exists():
        total_size = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
        size_str = _format_size(total_size)
    else:
        size_str = "—"

    # Fonti distinte
    try:
        all_meta = collection.get(include=["metadatas"])
        sources = set()
        source_types = Counter()
        for meta in all_meta.get("metadatas", []):
            if meta:
                src = meta.get("source", "sconosciuta")
                sources.add(src)
                source_types[meta.get("source_type", "?")] += 1
    except Exception:
        sources = set()
        source_types = Counter()

    # Output
    table = Table(title="📊 Statistiche Vector Store", box=box.ROUNDED, show_header=False)
    table.add_column("Proprietà", style="bold cyan")
    table.add_column("Valore", style="bold")
    table.add_row("Chunk totali", str(count))
    table.add_row("Fonti distinte", str(len(sources)))
    table.add_row("Dimensione su disco", size_str)
    table.add_row("Directory", str(config.CHROMA_DIR))
    table.add_row("Collection", config.COLLECTION_NAME)
    table.add_row("Modello embeddings", config.HF_EMBEDDING_MODEL)

    if source_types:
        types_str = ", ".join(f"{t}: {c}" for t, c in source_types.most_common())
        table.add_row("Tipi sorgente", types_str)

    console.print(table)
    logger.info(f"Stats: {count} chunk, {len(sources)} fonti, {size_str}")


def cmd_list():
    """Lista le fonti indicizzate."""
    vs = get_vector_store()
    try:
        collection = vs._collection
        all_meta = collection.get(include=["metadatas"])
    except Exception as e:
        console.print(f"[bold red]Errore:[/] {e}")
        return

    # Raggruppa per fonte
    source_info: dict[str, dict] = {}
    for meta in all_meta.get("metadatas", []):
        if not meta:
            continue
        src = meta.get("source", "sconosciuta")
        if src not in source_info:
            source_info[src] = {
                "count": 0,
                "type": meta.get("source_type", "?"),
                "ingested_at": meta.get("ingested_at", "?"),
            }
        source_info[src]["count"] += 1

    if not source_info:
        console.print("[dim]Nessuna fonte indicizzata.[/]")
        return

    table = Table(title="📚 Fonti indicizzate", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Fonte", style="bold")
    table.add_column("Tipo", style="cyan")
    table.add_column("Chunk", justify="right")
    table.add_column("Indicizzato il", style="dim")

    for i, (src, info) in enumerate(sorted(source_info.items()), 1):
        # Troncata la data per leggibilità
        date_str = info["ingested_at"][:19].replace("T", " ") if info["ingested_at"] != "?" else "?"
        table.add_row(str(i), src, info["type"], str(info["count"]), date_str)

    console.print(table)
    console.print(f"\n  [dim]Totale: {sum(i['count'] for i in source_info.values())} chunk da {len(source_info)} fonti[/]")


def cmd_reset():
    """Svuota il vector store."""
    vs = get_vector_store()
    try:
        count = vs._collection.count()
    except Exception:
        count = 0

    if count == 0:
        console.print("[dim]Il vector store è già vuoto.[/]")
        return

    console.print(f"[bold yellow]⚠ Stai per cancellare {count} chunk dal vector store.[/]")
    confirm = console.input("[bold]Confermi? (s/N): [/]").strip().lower()
    if confirm not in ("s", "si", "sì", "y", "yes"):
        console.print("[dim]Operazione annullata.[/]")
        return

    try:
        # Cancella e ricrea la directory
        db_path = config.CHROMA_DIR
        if db_path.exists():
            shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✅ Vector store svuotato. {count} chunk rimossi.[/]")
        logger.info(f"Reset vector store: {count} chunk rimossi")
    except Exception as e:
        console.print(f"[bold red]Errore nel reset:[/] {e}")
        logger.error(f"Errore reset: {e}")


def cmd_delete(source: str):
    """Rimuove tutti i chunk di una fonte specifica."""
    vs = get_vector_store()
    try:
        collection = vs._collection

        # Trova gli ID dei chunk con quella fonte
        results = collection.get(where={"source": source})
        ids = results.get("ids", [])

        if not ids:
            # Prova ricerca parziale (il source potrebbe essere un sottostringa)
            all_meta = collection.get(include=["metadatas"])
            matching_ids = []
            for doc_id, meta in zip(all_meta["ids"], all_meta["metadatas"]):
                if meta and source.lower() in meta.get("source", "").lower():
                    matching_ids.append(doc_id)

            if not matching_ids:
                console.print(f"[yellow]Nessun chunk trovato per la fonte:[/] {source}")
                console.print("[dim]Usa 'python manage.py list' per vedere le fonti disponibili.[/]")
                return

            ids = matching_ids
            # Mostra cosa verrà cancellato
            console.print(f"[yellow]Trovati {len(ids)} chunk con fonte che contiene:[/] {source}")

        console.print(f"[bold yellow]⚠ Stai per cancellare {len(ids)} chunk.[/]")
        confirm = console.input("[bold]Confermi? (s/N): [/]").strip().lower()
        if confirm not in ("s", "si", "sì", "y", "yes"):
            console.print("[dim]Operazione annullata.[/]")
            return

        collection.delete(ids=ids)
        console.print(f"[green]✅ Rimossi {len(ids)} chunk per la fonte:[/] {source}")
        logger.info(f"Delete: rimossi {len(ids)} chunk per fonte '{source}'")

    except Exception as e:
        console.print(f"[bold red]Errore:[/] {e}")
        logger.error(f"Errore delete '{source}': {e}")


def _format_size(size_bytes: int) -> str:
    """Formatta una dimensione in bytes in formato leggibile."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Manager — gestione del vector store")
    subparsers = parser.add_subparsers(dest="command", help="Comando da eseguire")

    subparsers.add_parser("stats", help="Mostra statistiche del vector store")
    subparsers.add_parser("list", help="Lista le fonti indicizzate")
    subparsers.add_parser("reset", help="Svuota il vector store")

    del_parser = subparsers.add_parser("delete", help="Rimuovi una fonte specifica")
    del_parser.add_argument("--source", required=True, help="Nome o percorso della fonte da rimuovere")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "stats":
        cmd_stats()
    elif args.command == "list":
        cmd_list()
    elif args.command == "reset":
        cmd_reset()
    elif args.command == "delete":
        cmd_delete(args.source)


if __name__ == "__main__":
    main()
