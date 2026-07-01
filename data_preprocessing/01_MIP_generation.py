import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

# --- Configuration ---


# INPUT_DIRS = [
#     r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003004__2022-05-16T14_57_25-Measurement 1_C19_GBA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003005__2022-05-16T16_25_46-Measurement 1_C19_GBA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003006__2022-05-16T17_13_39-Measurement 1_C19_GBA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004001__2022-06-27T11_22_54-Measurement 1_C4_LRRK2\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004002__2022-06-27T12_12_52-Measurement 1_C4_LRRK2\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004003__2022-06-27T13_02_42-Measurement 1_C4_LRRK2\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004004__2022-06-27T14_07_07-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004005__2022-06-27T14_56_55-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004006__2022-06-27T15_46_53-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004007__2022-06-27T17_09_02-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\Minee_160822_plate1__2022-08-16T10_54_01-Measurement 1_C18\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\Minee_160822_plate2__2022-08-16T11_43_12-Measurement 1_C18\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\Minee_160822_plate3__2022-08-16T12_31_25-Measurement 1_C18\Images",
# ]

EXPERIMENTS = {
    "SNCA_AST18": {
        "rows": [6, 7],
        "input_dirs": [
            r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
        ]
    },
    ########################################################
    # "SNCA_AST18_isogenic": {
    #     "rows": [4, 5],
    #     "input_dirs": [
    #         r"D:\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
    #         r"D:\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
    #         r"D:\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
    #         r"D:\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
    #     ]
    # },
    # "SNCA-G51D": {
    #     "rows": [4, 5],
    #     "input_dirs": [
    #         r"D:\160822\005001_2022-08-16T10_54_01-Measurement 1\Images",
    #         r"D:\160822\005002_2022-08-16T11_43_12-Measurement 1\Images",
    #         r"D:\160822\005003_2022-08-16T12_31_25-Measurement 1\Images",
    #     ]
    # },
    # "SNCA-G51D_isogenic": {
    #     "rows": [6, 7],
    #     "input_dirs": [
    #         r"D:\160822\005001_2022-08-16T10_54_01-Measurement 1\Images",
    #         r"D:\160822\005002_2022-08-16T11_43_12-Measurement 1\Images",
    #         r"D:\160822\005003_2022-08-16T12_31_25-Measurement 1\Images",
    #     ]
    # },
    #############################################################
    "GBA": {
        "rows": [5, 6, 7],
        "input_dirs": [
            r"C:\Users\admin\Desktop\professor_data\New_data\003004__2022-05-16T14_57_25-Measurement 1_C19_GBA\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\003005__2022-05-16T16_25_46-Measurement 1_C19_GBA\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\003006__2022-05-16T17_13_39-Measurement 1_C19_GBA\Images",
        ]
    },
    #######################################  "D:\003008__2022-05-16T18_49_33-Measurement 1_GBA_346"
    # "GBA_346": {
    #     "rows": [5, 6, 7],
    #     "input_dirs": [
    #         r"D:\003007__2022-05-16T18_01_40-Measurement 1_GBA_346\Images",
    #         r"D:\003008__2022-05-16T18_49_33-Measurement 1_GBA_346\Images",
    #         r"D:\003009__2022-05-16T19_37_25-Measurement 1_GBA_346\Images",
    #     ]
    # },
    # "GBA_WIMP4": {
    #     "rows": [5, 6, 7],
    #     "input_dirs": [
    #         r"D:\003014__2022-05-17T18_48_31-Measurement 1\Images",
    #         r"D:\003016__2022-05-17T20_24_27-Measurement 1\Images",
    #     ]
    # },
    ######################################
    # "Diverse_GBA": {
    #     "rows": [4],
    #     "input_dirs": [
    
    #         r"C:\Users\admin\Desktop\professor_data\005005__2022-08-16T14_22_56-Measurement 1_diverse\Images",
    
    #         r"C:\Users\admin\Desktop\professor_data\005007__2022-08-16T16_06_39-Measurement 1_diverse\Images",
    #     ]
    # },
    # "PINK1": {
    #     "rows": [2, 3, 4],
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004004__2022-06-27T14_07_07-Measurement 1_PINK1\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004005__2022-06-27T14_56_55-Measurement 1_PINK1\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004006__2022-06-27T15_46_53-Measurement 1_PINK1\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004007__2022-06-27T17_09_02-Measurement 1_PINK1\Images",
    #     ]
    # },
    "LRRK2": {
        "rows": [5, 6, 7],
        "input_dirs": [
            r"D:\professor_data\New_data\004001__2022-06-27T11_22_54-Measurement 1_C4_LRRK2\Images",
            r"D:\professor_data\New_data\004002__2022-06-27T12_12_52-Measurement 1_C4_LRRK2\Images",
        
        ]
    },
    "Control_C4": {
        "rows": [2, 3, 4],
        "input_dirs": [
            r"C:\Users\admin\Desktop\professor_data\New_data\004001__2022-06-27T11_22_54-Measurement 1_C4_LRRK2\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\004002__2022-06-27T12_12_52-Measurement 1_C4_LRRK2\Images",
         
        ]
    },
    "Control_C18": {
        "rows": [2, 3],
        "input_dirs": [
            r"C:\Users\admin\Desktop\professor_data\New_data\005001__2022-08-16T10_54_01-Measurement 1_C18\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\005002__2022-08-16T11_43_12-Measurement 1_C18\Images",
            r"C:\Users\admin\Desktop\professor_data\New_data\005003__2022-08-16T12_31_25-Measurement 1_C18\Images",
        ]
    },
    # "Diverse_C18": {
    #     "rows": [3],
    #     "input_dirs": [
    
    #         r"C:\Users\admin\Desktop\professor_data\005005__2022-08-16T14_22_56-Measurement 1_diverse\Images",
    
    #         r"C:\Users\admin\Desktop\professor_data\005007__2022-08-16T16_06_39-Measurement 1_diverse\Images",
    #     ]
    # },
    # "Control_SNCA_C19": {
    #     "rows": [2, 3],
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
    #     ]
    # },
    # "Control_GBA_C19": {
    #     "rows": [2, 3, 4],
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003004__2022-05-16T14_57_25-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003005__2022-05-16T16_25_46-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003006__2022-05-16T17_13_39-Measurement 1_C19_GBA\Images",
    #     ]
    # },
    # "alpha_syn_1day": {
    #     "rows": [2, 3, 4, 5, 6, 7],
    #     "cols": [8],
    #     "input_dirs": [
    #         r"D:\090822\001001__2022-08-09T16_31_17-Measurement 1\Images",
    #     ],
    # },
    # "alpha_syn_3day": {
    #     "rows": [2, 3, 4, 5, 6, 7],
    #     "cols": [3],
    #     "input_dirs": [
    #         r"D:\090822\001001__2022-08-09T16_31_17-Measurement 1\Images",
    #     ],
    # },
    # "alpha_syn_7day": {
    #     "rows": [2, 3, 4, 5, 6, 7],
    #     "cols": [3],
    #     "input_dirs": [
    #         r"D:\090822\001002__2022-08-09T17_35_08-Measurement 1\Images",
    #     ],
    # },
}

# --- 1. Data structure ---


@dataclass(frozen=True)
class ImageMetadata:
    group_name: str
    folder_id: str
    row: int
    col: int
    fraction: int
    plane: int
    channel: int
    full_path: str

    @property
    def group_key(self) -> Tuple[str, str, int, int, int, int]:
        return (
            self.group_name,
            self.folder_id,
            self.row,
            self.col,
            self.fraction,
            self.channel,
        )


DEFAULT_COLS_RANGE = range(2, 12)
Z_STACK_COUNT = 5


def get_args():
    parser = argparse.ArgumentParser(description="MIP Generation Pipeline")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to a single input directory. Overrides built-in config if provided.",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default="CustomGroup",
        help="Group name when using --input_dir",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7],
        help="Rows to process when using --input_dir (e.g., --rows 2 3 4)",
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        type=int,
        default=list(range(2, 12)),
        help="Columns to process when using --input_dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"./Output/MIP",
        help="Base directory for RGB MIP output",
    )
    parser.add_argument(
        "--cytox_dir",
        type=str,
        default=r"./Output/MIP_cytox",
        help="Base directory for Cytox MIP output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file containing EXPERIMENTS dictionary.",
    )
    return parser.parse_args()


# --- Logic functions ---
def extract_folder_id(path: str) -> str:
    parent_folder = os.path.basename(os.path.dirname(path))
    match = re.match(r"(\d+)", parent_folder)
    return match.group(1) if match else "Unknown"


def parse_files(group_name: str, config: dict) -> List[ImageMetadata]:
    file_pattern = re.compile(r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)")
    deduplicated_files = {}

    target_rows = config["rows"]
    target_cols = config.get("cols", DEFAULT_COLS_RANGE)
    directories = config["input_dirs"]

    print(
        f"[{group_name}] Scanning files... (Rows: {list(target_rows)}, Cols: {list(target_cols)})"
    )

    for folder in directories:
        if not os.path.exists(folder):
            print(f"Warning: Path not found - {folder}")
            continue

        f_id = extract_folder_id(folder)
        filenames = os.listdir(folder)

        for filename in tqdm(filenames, desc=f"Scanning {f_id}", leave=False):
            match = file_pattern.search(filename)
            if match:
                r, c, f, p, ch = map(int, match.groups())

                if r in target_rows and c in target_cols:
                    file_key = (f_id, r, c, f, p, ch)
                    deduplicated_files[file_key] = ImageMetadata(
                        group_name=group_name,
                        folder_id=f_id,
                        row=r,
                        col=c,
                        fraction=f,
                        plane=p,
                        channel=ch,
                        full_path=os.path.join(folder, filename),
                    )
    return list(deduplicated_files.values())


def run_pipeline(args):
    summary_data = []

    # Load configuration
    if args.input_dir:
        experiments_config = {
            args.group_name: {
                "rows": args.rows,
                "cols": args.cols,
                "input_dirs": [args.input_dir]
            }
        }
        print(f"Using provided input_dir: {args.input_dir}")
    elif args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            experiments_config = json.load(f)
    else:
        experiments_config = EXPERIMENTS

    for group_name, config in experiments_config.items():
        print(f"\n🚀 Processing Group: {group_name}")

        save_dir_rgb = os.path.join(args.output_dir, group_name)
        save_dir_cytox = os.path.join(args.cytox_dir, group_name)
        os.makedirs(save_dir_rgb, exist_ok=True)
        os.makedirs(save_dir_cytox, exist_ok=True)


        images = parse_files(group_name, config)
        if not images:
            print(f"No files found for group: {group_name}. Check configuration.")
            continue

 
        grouped_data = defaultdict(list)
        for img in images:
            grouped_data[img.group_key].append(img)

        final_results = defaultdict(dict)
        process_loop = tqdm(list(grouped_data.items()), desc="Calculating MIP")

        for key, stack in process_loop:
            group, f_id, r, c, f, ch = key

            sorted_stack = sorted(stack, key=lambda x: x.plane)[:Z_STACK_COUNT]
            if len(sorted_stack) < Z_STACK_COUNT:
                continue

            img_data = np.array(
                [tifffile.imread(m.full_path) for m in sorted_stack]
            )
            mip_img = np.max(img_data, axis=0).astype(np.uint16)
            final_results[(f_id, r, c, f)][ch] = mip_img

    
        save_count = 0
        for (f_id, r, c, f), channels in tqdm(
            final_results.items(), desc=f"Saving {group_name}"
        ):
            filename_prefix = f"{f_id}_r{r:02d}c{c:02d}f{f:02d}"

            summary_data.append(
                {
                    "Group": group_name,
                    "Folder_ID": f_id,
                    "Row": r,
                    "Col": c,
                    "Fraction": f,
                    "Has_RGB": all(ch in channels for ch in [1, 3, 4]),
                    "Has_Cytox": 2 in channels,
                }
            )

            if all(ch in channels for ch in [1, 3, 4]):
                rgb = np.stack([channels[3], channels[4], channels[1]], axis=-1)
                tifffile.imwrite(
                    os.path.join(save_dir_rgb, f"{filename_prefix}_Composite_RGB.tif"),
                    rgb,
                )
                save_count += 1

            if 2 in channels:
                tifffile.imwrite(
                    os.path.join(
                        save_dir_cytox, f"{filename_prefix}_MIP_ch2_Cytox.tif"
                    ),
                    channels[2],
                )

        print(f"Finished {group_name}! {save_count} images generated.")

    if summary_data:
        df = pd.DataFrame(summary_data)
        # df.to_csv(os.path.join(args.output_dir, "processing_summary.csv"), index=False)


if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
