#!/usr/bin/env python3
"""
TIF 파일들을 WebDataset tar shards 형식으로 변환
Lambda Labs에서 실행

사용법:
    python convert_tif_to_tar_shards.py

입력: /home/ubuntu/wds_shards/<LINE>/*.tif
출력: /home/ubuntu/wds_shards/<LINE>/plate=<PLATE>/<LINE>_plate=<PLATE>-<SHARD>.tar
"""

import os
import re
import json
import tarfile
import io
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ============== 설정 ==============
INPUT_ROOT = "/home/ubuntu/wds_shards"
OUTPUT_ROOT = "/home/ubuntu/model-east3/wds_shards_tar"  # 파일시스템에 저장 (인스턴스 종료해도 보존)

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

# 파일명 패턴: 003004_r05c02f01_Composite_RGB_x0_y0.tif
# plate = 003004
FILENAME_RE = re.compile(r"^(\d{6})_(.+)\.tif$")

SAMPLES_PER_TAR = 1000  # tar당 샘플 수


def parse_filename(filename: str):
    """파일명에서 plate 추출"""
    m = FILENAME_RE.match(filename)
    if m:
        plate = m.group(1)
        return plate
    return None


def write_tar_shard(tar_path: str, samples: list):
    """
    samples: [(tif_path, key, metadata_dict), ...]
    """
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    
    with tarfile.open(tar_path, "w") as tf:
        for tif_path, key, meta in samples:
            # Add TIF file
            tf.add(tif_path, arcname=f"{key}.tif")
            
            # Add JSON metadata
            json_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            tf.addfile(json_info, io.BytesIO(json_bytes))


def convert_line(line: str):
    """특정 LINE 폴더의 tif 파일들을 tar shards로 변환"""
    input_dir = os.path.join(INPUT_ROOT, line)
    if not os.path.isdir(input_dir):
        print(f"[SKIP] {line}: 폴더 없음")
        return
    
    # 모든 tif 파일 스캔
    tif_files = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
    print(f"[{line}] {len(tif_files)} tif files found")
    
    if len(tif_files) == 0:
        return
    
    # plate별로 그룹화
    plate_to_files = defaultdict(list)
    for f in tif_files:
        plate = parse_filename(f)
        if plate:
            plate_to_files[plate].append(f)
        else:
            print(f"[WARN] 파일명 파싱 실패: {f}")
    
    superclass = SUPERCLASS_MAP.get(line, line)
    
    # plate별로 tar shards 생성
    for plate, files in tqdm(plate_to_files.items(), desc=f"{line} plates"):
        plate_dir = os.path.join(OUTPUT_ROOT, line, f"plate={plate}")
        os.makedirs(plate_dir, exist_ok=True)
        
        # 파일들을 SAMPLES_PER_TAR 단위로 나누기
        shard_idx = 0
        samples = []
        
        for i, fname in enumerate(sorted(files)):
            tif_path = os.path.join(input_dir, fname)
            key = fname[:-4]  # .tif 제거
            
            meta = {
                "class": superclass,
                "line": line,
                "plate": plate,
                "filename": fname,
            }
            
            samples.append((tif_path, key, meta))
            
            # tar 파일 쓰기
            if len(samples) >= SAMPLES_PER_TAR or i == len(files) - 1:
                tar_name = f"{line}_plate={plate}-{shard_idx:06d}.tar"
                tar_path = os.path.join(plate_dir, tar_name)
                write_tar_shard(tar_path, samples)
                shard_idx += 1
                samples = []
    
    print(f"[{line}] 완료")


def main():
    print(f"입력: {INPUT_ROOT}")
    print(f"출력: {OUTPUT_ROOT}")
    print()
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    for line in LINE_FOLDERS:
        convert_line(line)
    
    print("\n=== 변환 완료 ===")
    print(f"학습 시 --shard_root={OUTPUT_ROOT} 로 설정하세요")


if __name__ == "__main__":
    main()
