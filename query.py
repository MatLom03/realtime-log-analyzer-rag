"""
query.py — Interroga il RAG in modalità interattiva o con una singola domanda.

Utilizzo:
  python query.py                          # modalità chat interattiva
  python query.py --question "Cos'è X?"   # singola domanda
  python query.py --question "..." --show-sources  # mostra i chunk usati
  python query.py --question "..." --top-k 10      # override chunk recuperati
"""

import argparse
import sys
import codecs
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

import config
from logging_config import get_logger

console = Console()
logger = get_logger(__name__)

# ── Costanti ───────────────────────────────────────────────────────────────

MAX_HISTORY = 5  # Numero massimo di coppie domanda-risposta in memoria

CHAT_COMMANDS = {
    "/help":        "Mostra i comandi disponibili",
    "/sources on":  "Attiva visualizzazione fonti",
    "/sources off": "Disattiva visualizzazione fonti",
    "/clear":       "Pulisci cronologia conversazione",
    "/history":     "Mostra la cronologia corrente",
}


# ── Prompt ──────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "history"],
    template="""Sei un assistente tecnico esperto. Rispondi alla domanda dell'utente basandoti sulle informazioni presenti nel contesto fornito.

Istruzioni:
- Basa la risposta esclusivamente sul contesto qui sotto.
- Il contesto può contenere frammenti di testo da documenti PDF: interpretali e sintetizzali in una risposta chiara e completa.
- Se il contesto contiene informazioni anche parziali sull'argomento, usale per rispondere al meglio.
- Solo se il contesto è completamente irrilevante rispetto alla domanda, rispondi: "Il contesto fornito non contiene informazioni su questo argomento."
- Rispondi nella stessa lingua della domanda.

CRONOLOGIA CONVERSAZIONE:
{history}

CONTESTO:
{context}

DOMANDA: {question}

RISPOSTA:""",
)


# ── Core ────────────────────────────────────────────────────────────────────

def build_chain(top_k: int | None = None):
    """Costruisce la catena RAG. Ritorna (chain, vector_store) oppure lancia un errore."""
    k = top_k or config.TOP_K

    embeddings = HuggingFaceEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL
    )
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    # Verifica che il DB non sia vuoto
    try:
        count = vector_store._collection.count()
    except Exception:
        count = 0

    if count == 0:
        console.print(
            "[bold red]Errore:[/] il vector store è vuoto.\n"
            "  Indicizza almeno un documento prima di fare query:\n"
            "  [dim]python ingest.py --pdf mio_documento.pdf[/]"
        )
        sys.exit(1)

    console.print(f"  📊 {count} chunk nel vector store")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    llm = ChatGroq(
        model_name=config.GROQ_MODEL,
        temperature=0,
        groq_api_key=config.GROQ_API_KEY,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableParallel({
        "source_documents": (lambda x: x["query"]) | retriever,
        "result": (
            {
                "context": (lambda x: x["query"]) | retriever | format_docs,
                "question": (lambda x: x["query"]),
                "history": (lambda x: x.get("history", "")),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
    })
    return chain


def format_history(history: list[tuple[str, str]]) -> str:
    """Formatta la cronologia per il prompt."""
    if not history:
        return "(nessuna conversazione precedente)"
    lines = []
    for q, a in history:
        lines.append(f"Utente: {q}")
        lines.append(f"Assistente: {a}")
    return "\n".join(lines)


def ask(chain, question: str, show_sources: bool = False,
        history: list[tuple[str, str]] | None = None) -> str:
    """Esegue una query RAG e stampa il risultato. Restituisce la risposta."""
    history_text = format_history(history or [])

    start = time.time()
    try:
        result = chain.invoke({"query": question, "history": history_text})
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            console.print("[bold red]Errore:[/] troppe richieste. Attendi qualche secondo e riprova.")
        elif "authentication" in error_msg.lower() or "401" in error_msg:
            console.print("[bold red]Errore:[/] GROQ_API_KEY non valida. Controlla il file .env")
        elif "timeout" in error_msg.lower():
            console.print("[bold red]Errore:[/] timeout nella risposta. Riprova.")
        else:
            console.print(f"[bold red]Errore nella query:[/] {e}")
        logger.error(f"Errore query '{question[:50]}...': {e}")
        return ""

    elapsed = time.time() - start
    answer = result["result"]

    console.print(Panel(
        Markdown(answer),
        title="[bold green]Risposta[/]",
        subtitle=f"[dim]{elapsed:.1f}s[/]",
        border_style="green",
    ))

    if show_sources and result.get("source_documents"):
        console.print("\n[dim]── Fonti usate ─────────────────────────────────[/]")
        for i, doc in enumerate(result["source_documents"], 1):
            src = doc.metadata.get("source", "sconosciuta")
            page = doc.metadata.get("page", "")
            page_info = f" (pag. {page})" if page else ""
            console.print(f"  [cyan]{i}.[/] {src}{page_info}")
            console.print(f"     [dim]{doc.page_content[:120].strip()}…[/]")

    logger.info(f"Query: '{question[:80]}' → {len(answer)} chars in {elapsed:.1f}s")
    return answer


def show_help():
    """Mostra i comandi disponibili nella chat."""
    table = Table(title="Comandi disponibili", show_header=True, header_style="bold cyan")
    table.add_column("Comando", style="bold")
    table.add_column("Descrizione")
    for cmd, desc in CHAT_COMMANDS.items():
        table.add_row(cmd, desc)
    table.add_row("exit / quit / esci", "Esci dalla chat")
    console.print(table)


def show_history(history: list[tuple[str, str]]):
    """Mostra la cronologia corrente."""
    if not history:
        console.print("[dim]Nessuna cronologia.[/]")
        return
    for i, (q, a) in enumerate(history, 1):
        console.print(f"  [bold yellow]{i}. Tu:[/] {q}")
        console.print(f"     [green]→[/] {a[:100]}{'…' if len(a) > 100 else ''}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Query — interroga il tuo knowledge base")
    parser.add_argument("--question", "-q", metavar="TESTO", help="Domanda singola")
    parser.add_argument("--show-sources", "-s", action="store_true", help="Mostra i documenti usati")
    parser.add_argument("--top-k", "-k", type=int, metavar="N",
                        help=f"Numero di chunk da recuperare (default: {config.TOP_K})")
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        console.print("[bold red]Errore:[/] GROQ_API_KEY non impostata. Crea il file .env partendo da .env.example")
        sys.exit(1)

    console.print("[bold]🔗 Connessione al vector store...[/]")
    chain = build_chain(top_k=args.top_k)
    console.print("[green]✅ Pronto.[/]\n")

    if args.question:
        ask(chain, args.question, show_sources=args.show_sources)
    else:
        # Modalità chat interattiva
        console.print(Panel(
            "[bold]RAG Chat interattiva[/]\n"
            "Digita la tua domanda e premi Invio.\n"
            "Scrivi [cyan]/help[/] per i comandi, [cyan]exit[/] per uscire.",
            border_style="blue"
        ))

        history: list[tuple[str, str]] = []
        show_sources = args.show_sources

        while True:
            try:
                question = console.input("\n[bold yellow]Tu:[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print()  # Nuova riga dopo ^C
                break

            if not question:
                continue

            # Comandi speciali
            q_lower = question.lower()
            if q_lower in ("exit", "quit", "esci"):
                break
            elif q_lower == "/help":
                show_help()
                continue
            elif q_lower == "/sources on":
                show_sources = True
                console.print("[green]✅ Fonti attivate.[/]")
                continue
            elif q_lower == "/sources off":
                show_sources = False
                console.print("[green]✅ Fonti disattivate.[/]")
                continue
            elif q_lower == "/clear":
                history.clear()
                console.print("[green]✅ Cronologia pulita.[/]")
                continue
            elif q_lower == "/history":
                show_history(history)
                continue

            answer = ask(chain, question, show_sources=show_sources, history=history)
            if answer:
                history.append((question, answer))
                # Mantieni solo le ultime N coppie
                if len(history) > MAX_HISTORY:
                    history = history[-MAX_HISTORY:]

    console.print("\n[dim]Arrivederci![/]")


if __name__ == "__main__":
    main()
