import argparse
import io
import os
import tarfile


def convert_to_wds(source_dir, dest_root, limit):
    os.makedirs(dest_root, exist_ok=True)
    tar_files = [f for f in os.listdir(source_dir) if f.endswith(".tar.gz")]

    if not tar_files:
        print(f"❌ {source_dir} 경로에 .tar.gz 파일이 없습니다.")
        return

    for tar_file in tar_files:
        tar_path = os.path.join(source_dir, tar_file)
        class_name = tar_file.replace(".tar.gz", "")

        print(f"📦 [{class_name}] 바로 WebDataset 샤드로 변환 시작... (최대 {limit}장)")
        class_count = 0
        images_per_shard = 5000

        # 각 plate별 상태 관리: {'count': 0, 'shard_idx': 0, 'out_tar': None}
        plates_state = {}

        def process_image(filename, img_data):
            nonlocal class_count

            # 파일 이름에서 앞 6자리를 추출하여 plate 번호로 사용 (숫자가 아니면 000000)
            base_name = os.path.basename(filename)
            plate_id = base_name[:6]
            if not plate_id.isdigit() or len(plate_id) < 6:
                plate_id = "000000"

            if plate_id not in plates_state:
                plates_state[plate_id] = {"count": 0, "shard_idx": 0, "out_tar": None}

            state = plates_state[plate_id]

            # WDS 요구 폴더 구조: /wds_shards_tar/{class_name}/plate={plate_id}/
            wds_class_dir = os.path.join(dest_root, class_name, f"plate={plate_id}")
            os.makedirs(wds_class_dir, exist_ok=True)

            # 5000장 단위로 샤드 교체
            if state["out_tar"] is None or state["count"] >= (
                state["shard_idx"] * images_per_shard
            ):
                if state["out_tar"] is not None:
                    state["out_tar"].close()
                shard_path = os.path.join(
                    wds_class_dir, f"{state['shard_idx']:04d}.tar"
                )
                state["out_tar"] = tarfile.open(shard_path, "w")
                state["shard_idx"] += 1

            prefix = f"{state['count']:06d}"

            # 1. 이미지 파일 기록
            img_info = tarfile.TarInfo(name=f"{prefix}.tif")
            img_info.size = len(img_data)
            state["out_tar"].addfile(img_info, io.BytesIO(img_data))

            # 2. 가짜 JSON 기록
            json_data = b"{}"
            json_info = tarfile.TarInfo(name=f"{prefix}.json")
            json_info.size = len(json_data)
            state["out_tar"].addfile(json_info, io.BytesIO(json_data))

            state["count"] += 1
            class_count += 1

            if class_count % 2000 == 0:
                print(f"   ... 전체 {class_count}장 WDS 샤드로 압축 완료 ...")

        with tarfile.open(tar_path, "r:gz") as outer_tar:
            for outer_member in outer_tar:
                if class_count >= limit:
                    break

                # Case 1: .tar.gz 안의 .tar 파일 (2중 구조)
                if outer_member.isfile() and outer_member.name.lower().endswith(".tar"):
                    outer_stream = outer_tar.extractfile(outer_member)
                    if outer_stream is not None:
                        tar_data = outer_stream.read()
                        with tarfile.open(
                            fileobj=io.BytesIO(tar_data), mode="r"
                        ) as inner_tar:
                            for inner_member in inner_tar:
                                if class_count >= limit:
                                    break
                                if (
                                    inner_member.isfile()
                                    and inner_member.name.lower().endswith(
                                        (".tif", ".tiff")
                                    )
                                ):
                                    with inner_tar.extractfile(inner_member) as src:
                                        process_image(inner_member.name, src.read())

                # Case 2: .tar.gz 안에 바로 .tif가 있는 구조
                elif outer_member.isfile() and outer_member.name.lower().endswith(
                    (".tif", ".tiff")
                ):
                    with outer_tar.extractfile(outer_member) as src:
                        process_image(outer_member.name, src.read())

        # 남아있는 열려있는 모든 tar 파일 핸들 닫기
        for state in plates_state.values():
            if state["out_tar"] is not None:
                state["out_tar"].close()

        print(
            f"  ✅ [{class_name}] 총 {class_count}장 변환 완벽 완료. (총 {len(plates_state)}개 Plate 탐지됨)\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tar.gz directly to WebDataset shards with JSON padding"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/home/ubuntu/model-east3/wds_shards_tar",
        help="Directory with .tar.gz",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="/home/ubuntu/model-east3/wds_shards_tar",
        help="WDS Target Root",
    )
    parser.add_argument("--limit", type=int, default=27000)
    args = parser.parse_args()

    convert_to_wds(args.source_dir, args.dest_dir, args.limit)
