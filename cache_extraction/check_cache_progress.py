import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check the progress of cache extraction.")
    parser.add_argument("--cache_dir", type=str, default="/home/ubuntu/model-east3/caches",
                        help="Root directory where caches are being extracted.")
    args = parser.parse_args()

    # 모든 .npz 파일 검색
    pattern = os.path.join(args.cache_dir, "**", "*.npz")
    npz_files = glob.glob(pattern, recursive=True)
    
    if not npz_files:
        print(f"[{args.cache_dir}] 폴더에 아직 추출된 캐시 파일이 없습니다.")
        return

    # CNN 캐시와 SAE 캐시 분류
    # SAE 캐시는 보통 "SAE_dim" 폴더 안에 있거나 이름에 "sae_"가 포함됨
    sae_files = [f for f in npz_files if "SAE_dim" in f or "sae_" in os.path.basename(f)]
    cnn_files = [f for f in npz_files if f not in sae_files]

    total_extracted = len(npz_files)
    total_sae = len(sae_files)
    total_cnn = len(cnn_files)

    # 목표치 (run_unified_extractions.sh 기준)
    # CNN: 8 seeds * 3 layers = 24개
    # SAE: 8 seeds * 8 configs = 64개
    # 총합: 88개
    target_cnn = 24
    target_sae = 64
    target_total = target_cnn + target_sae

    print("=" * 50)
    print("📊 캐시 추출 진행 상황 (Cache Extraction Progress)")
    print("=" * 50)
    print(f"📂 대상 경로: {args.cache_dir}")
    print("-" * 50)
    print(f"✅ 전체 진행률 : {total_extracted} / {target_total} 완료 ({(total_extracted/target_total)*100:.1f}%)")
    print(f"   ▶ CNN 캐시: {total_cnn} / {target_cnn} 완료 ({(total_cnn/target_cnn)*100:.1f}%)")
    print(f"   ▶ SAE 캐시: {total_sae} / {target_sae} 완료 ({(total_sae/target_sae)*100:.1f}%)")
    print("=" * 50)
    
    # 세부 리스트를 원할 경우 (최근 생성된 파일 5개 보여주기)
    if total_extracted > 0:
        print("\n최근 추출된 파일 (Top 5):")
        # 수정 시간 기준으로 정렬
        sorted_files = sorted(npz_files, key=os.path.getmtime, reverse=True)
        for i, f in enumerate(sorted_files[:5]):
            print(f" {i+1}. {os.path.basename(f)}")

if __name__ == "__main__":
    main()
