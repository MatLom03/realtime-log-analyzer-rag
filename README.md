# RAG Project 🔍

Pipeline **Retrieval-Augmented Generation** con Python, LangChain, Chroma e OpenAI.

## Struttura

```
rag-project/
├── config.py        ← Configurazione centralizzata
├── ingest.py        ← Indicizza documenti nel vector store
├── query.py         ← Interroga il knowledge base
├── data/
│   ├── uploads/     ← Metti qui i tuoi PDF
│   └── chroma_db/   ← Vector store (auto-generato)
├── .env             ← API key (crea da .env.example)
├── .env.example     ← Template variabili d'ambiente
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
# Modifica .env e inserisci la tua OPENAI_API_KEY
```

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
```

### Interrogare il knowledge base

```bash
# Chat interattiva
python query.py

# Domanda singola
python query.py --question "Di cosa parla il documento?"

# Con fonti visibili
python query.py --question "Spiega X" --show-sources
```

## Parametri configurabili (config.py)

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Caratteri per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap tra chunk |
| `TOP_K` | 5 | Chunk recuperati per query |
| `OPENAI_MODEL` | gpt-4o | Modello di generazione |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | Modello di embedding |
