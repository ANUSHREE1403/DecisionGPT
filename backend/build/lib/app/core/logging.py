"""
app/core/logging.py
Structured logging via loguru — call setup_logging() once at startup.
"""
import sys
from loguru import logger
from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()  # remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=fmt,
        level=settings.log_level,
        colorize=True,
    )

    logger.add(
        "logs/decisiongpt.log",
        format=fmt,
        level="INFO",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
    )

    logger.info("Logging initialised | env={} | level={}", settings.app_env, settings.log_level)
