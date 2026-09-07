"""
query.py — Interroga il RAG in modalità interattiva o con una singola domanda.

Utilizzo:
  python query.py                          # modalità chat interattiva
  python query.py --question "Cos'è X?"   # singola domanda
  python query.py --question "..." --show-sources  # mostra i chunk usati
"""

import argparse
import sys
import codecs

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

import config

console = Console()


# ── Prompt ──────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Sei un assistente esperto. Usa SOLO le informazioni nel contesto qui sotto per rispondere.
Se la risposta non è nel contesto, dì "Non ho abbastanza informazioni su questo argomento."

CONTESTO:
{context}

DOMANDA: {question}

RISPOSTA:""",
)


# ── Core ────────────────────────────────────────────────────────────────────

def build_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL
    )
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K},
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
            {"context": (lambda x: x["query"]) | retriever | format_docs, "question": (lambda x: x["query"])}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
    })
    return chain


def ask(chain, question: str, show_sources: bool = False) -> str:
    result = chain.invoke({"query": question})
    answer = result["result"]

    console.print(Panel(Markdown(answer), title="[bold green]Risposta[/]", border_style="green"))

    if show_sources and result.get("source_documents"):
        console.print("\n[dim]── Fonti usate ─────────────────────────────────[/]")
        for i, doc in enumerate(result["source_documents"], 1):
            src = doc.metadata.get("source", "sconosciuta")
            page = doc.metadata.get("page", "")
            page_info = f" (pag. {page})" if page else ""
            console.print(f"  [cyan]{i}.[/] {src}{page_info}")
            console.print(f"     [dim]{doc.page_content[:120].strip()}…[/]")

    return answer


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Query — interroga il tuo knowledge base")
    parser.add_argument("--question", "-q", metavar="TESTO", help="Domanda singola")
    parser.add_argument("--show-sources", "-s", action="store_true", help="Mostra i documenti usati")
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        console.print("[bold red]Errore:[/] GROQ_API_KEY non impostata.")
        sys.exit(1)

    console.print("[bold]🔗 Connessione al vector store...[/]")
    chain = build_chain()
    console.print("[green]✅ Pronto.[/]\n")

    if args.question:
        ask(chain, args.question, show_sources=args.show_sources)
    else:
        # Modalità chat interattiva
        console.print(Panel(
            "[bold]RAG Chat interattiva[/]\nDigita la tua domanda e premi Invio. Scrivi [cyan]exit[/] per uscire.",
            border_style="blue"
        ))
        while True:
            try:
                question = console.input("\n[bold yellow]Tu:[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if question.lower() in ("exit", "quit", "esci"):
                break
            if not question:
                continue
            ask(chain, question, show_sources=args.show_sources)

    console.print("\n[dim]Arrivederci![/]")


if __name__ == "__main__":
    main()
