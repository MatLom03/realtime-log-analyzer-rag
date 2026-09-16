# RAG Project 🔍

Pipeline **Retrieval-Augmented Generation** con Python, LangChain, Chroma, Groq e HuggingFace Embeddings.

## Architettura

```
                    ┌──────────────┐
                    │   Documenti  │
                    │ PDF/URL/TXT  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   ingest.py  │  Caricamento + chunking
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  HuggingFace Embeddings │  all-MiniLM-L6-v2 (locale)
              │  sentence-transformers  │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │    Chroma    │  Vector store persistente
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   query.py   │  Retrieval + generazione
                    └──────┬───────┘
                           │
                  ┌────────▼────────┐
                  │    Groq LLM     │  qwen/qwen3.8-27b
                  │  (API remota)   │
                  └─────────────────┘
```

## Funzionalità

- 📄 **Ingestione multi-formato** — PDF, URL, file di testo, cartelle intere
- 🔍 **Ricerca semantica** — similarity search su embeddings locali
- 💬 **Chat interattiva** — conversazione continua con cronologia
- 📖 **Fonti visibili** — mostra i chunk usati per ogni risposta
- 🗂️ **Gestione vector store** — stats, reset, lista fonti, cancellazione
- ⚡ **Embeddings locali** — nessuna API key necessaria per indicizzare
- 🚀 **LLM veloce** — Groq per inferenza rapida

## Struttura

```
rag-project/
├── config.py          ← Configurazione centralizzata
├── ingest.py          ← Indicizza documenti nel vector store
├── query.py           ← Interroga il knowledge base (chat + singola domanda)
├── manage.py          ← Gestione vector store (stats, reset, list, delete)
├── logging_config.py  ← Setup logging centralizzato
├── data/
│   ├── uploads/       ← Metti qui i tuoi PDF
│   └── chroma_db/     ← Vector store (auto-generato)
├── logs/              ← File di log (auto-generato)
├── .env               ← API key (crea da .env.example)
├── .env.example       ← Template variabili d'ambiente
└── requirements.txt
```

## Setup rapido

```bash
# 1. Crea un virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Configura le variabili d'ambiente
copy .env.example .env
# Modifica .env e inserisci la tua GROQ_API_KEY
```

> **Nota:** gli embeddings usano `all-MiniLM-L6-v2` in locale tramite `sentence-transformers`. Non serve nessuna API key per indicizzare documenti. La `GROQ_API_KEY` serve solo per fare query (generazione risposte).

## Utilizzo

### Indicizzare documenti

```bash
# Un PDF
python ingest.py --pdf data/uploads/mio_documento.pdf

# Una pagina web
python ingest.py --url https://example.com/articolo

# Un file di testo
python ingest.py --text data/uploads/note.txt

# Tutti i PDF in una cartella
python ingest.py --dir data/uploads/

# Forzare re-indicizzazione (salta controllo duplicati)
python ingest.py --pdf doc.pdf --force
```

### Interrogare il knowledge base

```bash
# Chat interattiva
python query.py

# Domanda singola
python query.py --question "Di cosa parla il documento?"

# Con fonti visibili
python query.py --question "Spiega X" --show-sources

# Override del numero di chunk recuperati
python query.py --question "Spiega X" --top-k 10
```

### Gestire il vector store

```bash
# Statistiche del database
python manage.py stats

# Lista delle fonti indicizzate
python manage.py list

# Svuota il vector store
python manage.py reset

# Rimuovi una fonte specifica
python manage.py delete --source "mio_documento.pdf"
```

## Comandi chat interattiva

Nella modalità chat, puoi usare comandi speciali:

| Comando | Descrizione |
|---------|-------------|
| `/help` | Mostra i comandi disponibili |
| `/sources on` | Attiva visualizzazione fonti |
| `/sources off` | Disattiva visualizzazione fonti |
| `/clear` | Pulisci cronologia conversazione |
| `exit` / `quit` / `esci` | Esci dalla chat |

## Parametri configurabili (config.py)

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `GROQ_API_KEY` | — | API key Groq (obbligatoria per query) |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Modello LLM per la generazione |
| `HF_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Modello embeddings locale |
| `CHUNK_SIZE` | 1000 | Caratteri per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap tra chunk |
| `TOP_K` | 5 | Chunk recuperati per query |
| `COLLECTION_NAME` | `rag_docs` | Nome della collection Chroma |
