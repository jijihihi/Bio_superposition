# ==============================================================================
# Logging
# ==============================================================================

import logging
import re

_LOGGING_CONFIGURED = False


def get_logger(name: str):
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _LOGGING_CONFIGURED = True
    return logging.getLogger(name)


# ==============================================================================
# Constants / mappings (same as your training code)
# ==============================================================================

DEFAULT_SHARD_ROOT = "/content/wds_shards"

LINE_FOLDERS = [
    "Control_C4", "Control_C18", "Control_C19",
    "SNCA", "GBA", "LRRK2"
]

SUPERCLASS_MAP = {
    "Control_C4":  "Control",
    "Control_C18": "Control",
    "Control_C19": "Control",
    "SNCA":        "SNCA",
    "GBA":         "GBA",
    "LRRK2":       "LRRK2",
}

CLASS_TO_LABEL = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}

PLATE_DIR_RE = re.compile(r"plate=(\d{6})")

OUT_DIM = 512  # encoder output channels for feature maps
