# ==============================================================================
# Tar offset index -> SampleRef
# ==============================================================================

import os
import glob
import time
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sae_project.logging_utils import get_logger, PLATE_DIR_RE, SUPERCLASS_MAP, CLASS_TO_LABEL

logger = get_logger("data_shards")


@dataclass(frozen=True)
class SampleRef:
    tar_path: str
    prefix: str
    tif_off: int
    tif_size: int
    js_off: int
    js_size: int
    line: str
    superclass: str
    label: int
    plate: str


def _infer_line_and_plate_from_tarpath(tar_path: str) -> Tuple[str, str]:
    parts = tar_path.replace("\\", "/").split("/")
    line = parts[-3]
    m = PLATE_DIR_RE.search(parts[-2])
    plate = m.group(1) if m else "UNKNOWN"
    return line, plate


def build_tar_index_if_needed(tar_path: str):
    idx_path = tar_path + ".pkl"
    if os.path.exists(idx_path):
        return

    t0 = time.time()
    items = {}

    import tarfile
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            name = m.name
            if name.endswith(".tif"):
                pref = name[:-4]
                it = items.get(pref, {})
                it["tif_off"] = m.offset_data
                it["tif_size"] = m.size
                items[pref] = it
            elif name.endswith(".json"):
                pref = name[:-5]
                it = items.get(pref, {})
                it["js_off"] = m.offset_data
                it["js_size"] = m.size
                items[pref] = it

    pairs = []
    for pref, it in items.items():
        if "tif_off" in it and "js_off" in it:
            pairs.append((pref, it["tif_off"], it["tif_size"], it["js_off"], it["js_size"]))

    with open(idx_path, "wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"[tar-index] built {len(pairs)} pairs: {os.path.basename(tar_path)} ({time.time()-t0:.1f}s)")


def load_all_sample_refs(shard_root: str) -> List[SampleRef]:
    tar_paths = sorted(glob.glob(os.path.join(shard_root, "*", "plate=*", "*.tar")))
    if len(tar_paths) == 0:
        raise FileNotFoundError(f"No tar shards found under: {shard_root}")

    for tp in tar_paths:
        build_tar_index_if_needed(tp)

    refs: List[SampleRef] = []
    for tp in tar_paths:
        line, plate = _infer_line_and_plate_from_tarpath(tp)
        superclass = SUPERCLASS_MAP.get(line, line)
        label = CLASS_TO_LABEL[superclass]

        with open(tp + ".pkl", "rb") as f:
            pairs = pickle.load(f)

        for pref, tif_off, tif_size, js_off, js_size in pairs:
            refs.append(SampleRef(
                tar_path=tp,
                prefix=pref,
                tif_off=int(tif_off),
                tif_size=int(tif_size),
                js_off=int(js_off),
                js_size=int(js_size),
                line=line,
                superclass=superclass,
                label=label,
                plate=plate
            ))

    logger.info(f"Loaded sample refs: {len(refs)}")
    return refs


def build_uid_to_refidx(refs: List[SampleRef]) -> Dict[str, int]:
    return {f"{r.tar_path}:{r.prefix}": i for i, r in enumerate(refs)}
