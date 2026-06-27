import argparse
import glob
import os
import tarfile
import json
from pathlib import Path
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Package QC-passed patches into WebDataset Tar shards.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing QC passed patches (e.g., Output/patches)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the tar shards (e.g., wds_shards_tar)")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of images per tar shard")
    parser.add_argument(
        "--classes", 
        nargs="+", 
        default=["Control_C4", "Control_C18", "Control_C19", "SNCA", "GBA", "LRRK2"],
        help="List of classes (lines) to package. By default, only the 6 CNN training classes are packaged to save space."
    )
    return parser.parse_args()

def main():
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # To group by line and plate, we'll scan the input_dir
    # Structure expected: input_dir / <line> / plate=XXXXXX / <files>.tif
    
    target_classes = set(args.classes) if args.classes else None
    
    # Collect all tif files
    print(f"Scanning {input_dir} for patches...")
    all_tif_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".tif") or f.endswith(".tiff"):
                all_tif_files.append(Path(root) / f)
                
    if not all_tif_files:
        print("No TIFF files found.")
        return
        
    print(f"Found {len(all_tif_files)} total TIFF files.")
    
    # Group by (line, plate)
    groups = {}
    for filepath in all_tif_files:
        rel_path = filepath.relative_to(input_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            line = parts[0]
            plate = parts[1]
            
            if target_classes and line not in target_classes:
                continue
                
            key = (line, plate)
            if key not in groups:
                groups[key] = []
            groups[key].append(filepath)

    if not groups:
        print("No files matched the target classes.")
        return

    # Create tar files
    total_tars = 0
    total_images_packed = 0
    
    print("\nPackaging into Tar Shards...")
    for (line, plate), files in tqdm(groups.items(), desc="Processing Plates"):
        save_root = output_dir / line / plate
        save_root.mkdir(parents=True, exist_ok=True)
        
        # Split into chunks
        chunks = [files[i:i + args.shard_size] for i in range(0, len(files), args.shard_size)]
        
        for chunk_idx, chunk_files in enumerate(chunks):
            tar_name = f"shard-{chunk_idx:04d}.tar"
            tar_path = save_root / tar_name
            
            with tarfile.open(tar_path, "w") as tar:
                for file_path in chunk_files:
                    # Add TIF
                    arcname_tif = file_path.name
                    tar.add(file_path, arcname=arcname_tif)
                    
                    # Add Dummy JSON
                    # InMemoryTarBank expects a corresponding .json file for each .tif to pair them
                    json_name = file_path.stem + ".json"
                    dummy_data = json.dumps({"source": "preprocessing.sh", "line": line, "plate": plate}).encode('utf-8')
                    
                    tarinfo = tarfile.TarInfo(name=json_name)
                    tarinfo.size = len(dummy_data)
                    import io
                    tar.addfile(tarinfo, io.BytesIO(dummy_data))
                    
                    total_images_packed += 1
            total_tars += 1

    print("\n" + "=" * 60)
    print("✅ TAR SHARD GENERATION COMPLETED")
    print("=" * 60)
    print(f"  - Target Classes        : {', '.join(target_classes) if target_classes else 'ALL'}")
    print(f"  - Total Images Packed   : {total_images_packed}")
    print(f"  - Total Tar Shards      : {total_tars}")
    print(f"  - Output Directory      : {output_dir}")
    print("=" * 60)
    print("Now you can set --shard_root in your train.py to this output directory.")

if __name__ == "__main__":
    main()
