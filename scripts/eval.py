import argparse
import os
import json
from pathlib import Path
import time
from dotenv import load_dotenv

from core.eval import process_predictions_from_dir
from core.get_score import get_score


load_dotenv()

if "DATA_DIR" not in os.environ:
    raise Exception("DATA_DIR environment variable not found.  Set DATA_DIR environment variable via .env")

DATA_DIR = Path(os.environ["DATA_DIR"])

OCRBench_V2_IMAGE_BASE_PATH = DATA_DIR / "OCRBench_v2"
OCRBench_V2_JSON_PATH = OCRBench_V2_IMAGE_BASE_PATH / "OCRBench_v2.json"

if not OCRBench_V2_IMAGE_BASE_PATH.is_dir():
    raise Exception(f"OCRBenchV2 data not found.  Download from https://drive.google.com/file/d/1Hk1TMu--7nr5vJ7iaNwMQZ_Iw9W_KI3C/view?usp=sharing and place at {OCRBench_V2_IMAGE_BASE_PATH}")

if not OCRBench_V2_JSON_PATH.is_file():
    raise Exception(f"OCRBenchV2 json not found.  Download from https://drive.google.com/file/d/1Hk1TMu--7nr5vJ7iaNwMQZ_Iw9W_KI3C/view?usp=sharing and place at {OCRBench_V2_IMAGE_BASE_PATH}")

RESULTS_INPUT_DIR = DATA_DIR / "ocr_results"

if not RESULTS_INPUT_DIR.is_dir():
    raise Exception("OCR Results not found.  Run OCR with ' just run ocr ' command")

RESULTS_OUTPUT_DIR = DATA_DIR / "output_results"
RESULTS_OUTPUT_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":

    allowed_task_types_to_evaluate = [
        "document classification en", 
        "document parsing en", 
        "full-page OCR en",
        "key information extraction en", 
        "reasoning VQA en",
    ]

    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    process_predictions_from_dir(
        RESULTS_INPUT_DIR,
        RESULTS_OUTPUT_DIR,
        allowed_task_types=allowed_task_types_to_evaluate
    )

    print(f"\nProcessing finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    output_files = list(RESULTS_OUTPUT_DIR.glob("*.json"))
    print(output_files)
    print(f"Got {len(output_files)} Files:\n{(output_files)}")
    all_results = {}
    for file in output_files:
        all_results[file.stem] = get_score(file)

    with open(RESULTS_OUTPUT_DIR.parent / "final_results.json", "w") as fp:
        json.dump(all_results, fp, indent=True)

