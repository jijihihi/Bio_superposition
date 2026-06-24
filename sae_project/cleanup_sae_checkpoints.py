import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old SAE checkpoints, keeping only ep008.pt"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/ubuntu/model-east3/outputs",
        help="Base directory containing MoCo_seed* folders",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only prints what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    print(f"Scanning directory: {args.base_dir}")

    # SAE 폴더 내부의 모든 .pt 파일 검색
    # 패턴: base_dir/MoCo_seed*/SAE_dim*/*.pt
    pattern = os.path.join(args.base_dir, "MoCo_seed*", "SAE_dim*", "*.pt")
    all_pt_files = glob.glob(pattern)

    if not all_pt_files:
        print("No SAE checkpoint files found.")
        return

    deleted_count = 0
    kept_count = 0
    total_size_freed = 0

    for pt_file in all_pt_files:
        # ep008.pt로 끝나는 파일은 보존
        if pt_file.endswith("_ep008.pt"):
            kept_count += 1
            continue

        # 그 외의 .pt 파일은 삭제 대상
        try:
            file_size = os.path.getsize(pt_file)
            if not args.dry_run:
                os.remove(pt_file)

            deleted_count += 1
            total_size_freed += file_size
            # 너무 많이 출력되는 것을 방지하기 위해 100개마다 하나씩 출력하거나, 그냥 카운트만
            # print(f"Deleted: {pt_file}")
        except Exception as e:
            print(f"Failed to delete {pt_file}: {e}")

    print("\n" + "=" * 50)
    if args.dry_run:
        print("💡 DRY RUN MODE: No files were actually deleted.")
    else:
        print("✅ CLEANUP COMPLETE")
    print("=" * 50)
    print(f"▶ Kept (_ep008.pt) : {kept_count} files")
    print(f"▶ Deleted (others) : {deleted_count} files")

    # 바이트를 기가바이트로 변환
    gb_freed = total_size_freed / (1024**3)
    if args.dry_run:
        print(f"▶ Space to free    : {gb_freed:.2f} GB")
    else:
        print(f"▶ Space freed      : {gb_freed:.2f} GB")
    print("=" * 50)


if __name__ == "__main__":
    main()
