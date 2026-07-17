"""
Islamic AI Assistant — Logging Configuration

Setup Loguru untuk logging terstruktur di seluruh aplikasi.
Log level dapat dikonfigurasi melalui env var LOG_LEVEL.
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Hapus handler default Loguru
logger.remove()

# Ambil log level dari environment variable (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Format log terstruktur dengan timestamp, level, dan pesan
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Tambahkan handler untuk stdout dengan format terstruktur
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# Tambahkan handler untuk file log (rotasi otomatis setiap 10MB)
_log_file = os.getenv("LOG_FILE", "logs/app.log")
try:
    Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        _log_file,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )
except (PermissionError, OSError) as e:
    logger.warning(f"File logging disabled ({_log_file}): {e}")

# Export logger agar bisa diimport di module lain
__all__ = ["logger"]

def get_logger(name: str = None):
    """
    Dapatkan logger instance dengan nama tertentu.

    Args:
        name: Nama module/component yang menggunakan logger

    Returns:
        Logger instance yang sudah dikonfigurasi

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Aplikasi dimulai")
    """
    if name:
        return logger.bind(name=name)
    return logger
