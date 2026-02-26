import logging
import os

# Global flag to ensure we only configure handlers once
_LOGGING_CONFIGURED = False

def get_logger(name: str, log_file: str = "logs/app.log", level=logging.INFO):
    """
    Returns a logger with a shared global configuration.
    All loggers write to the same file.
    """
    global _LOGGING_CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only configure handlers once (global)
    if not _LOGGING_CONFIGURED:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # File handler (shared)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s"
        ))

        # Console handler (shared)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        # Attach handlers to the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(fh)
        root_logger.addHandler(ch)

        _LOGGING_CONFIGURED = True

    return logger
