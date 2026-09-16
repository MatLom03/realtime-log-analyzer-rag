"""
logging_config.py — Setup logging centralizzato per il progetto RAG.

Ogni modulo fa:
    from logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config

# ── Directory logs ──────────────────────────────────────────────────────────
LOG_DIR = config.BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag.log"

# ── Formato ─────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Setup (eseguito una sola volta all'import) ─────────────────────────────
_configured = False


def _setup():
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # File handler — rotazione a 5 MB, 3 backup
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(file_handler)

    # Console handler — solo WARNING+ per non inquinare l'output rich
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console_handler)

    # Silenzia i logger rumorosi di terze parti
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Restituisce un logger configurato per il modulo specificato."""
    _setup()
    return logging.getLogger(name)
