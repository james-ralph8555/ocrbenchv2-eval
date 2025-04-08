import os
import re
import ast
import json
import argparse
import numpy as np
from tqdm import tqdm
import concurrent.futures
import glob # For finding files
import time
from collections import defaultdict
import functools # For partial
import pandas as pd

# --- Import your evaluation metric functions ---
from OCRBenchv2.vqa_metric import vqa_evaluation, cn_vqa_evaluation, math_expression_evaluation, vqa_evaluation_case_sensitive, counting_evaluation, cn_math_expression_evaluation
from OCRBenchv2.IoUscore_metric import vqa_with_position_evaluation, calculate_iou, extract_coordinates
from OCRBenchv2.TEDS_metric import TEDS, convert_markdown_table_to_html, convert_str_to_dict, convert_str_to_multi_dict, generate_combinations, dict_to_html, compute_f1_score, doc_parsing_evaluation, wrap_html_table
from OCRBenchv2.page_ocr_metric import cal_per_metrics
from OCRBenchv2.spotting_metric import extract_bounding_boxes_robust, spotting_evaluation
# --- End Imports ---

# --- Helper functions ---
def is_nan_value(value):
    if value is None: return True
    if isinstance(value, str) and value.lower() == 'nan': return True
    if pd.isna(value): 
        return True
    return False

def get_value_or_zero(value):
    # Ensure score is float, handle potential non-numeric gracefully
    try:
        if isinstance(value, (int, float)):
             return float(value)
        return float(value) if value is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def extract_prediction_content(prediction_value):
    """Removes potential markdown code block fences from prediction strings."""
    if not isinstance(prediction_value, str):
        return prediction_value # Return as is if not string
    match = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", prediction_value, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return prediction_value.strip()

def sanitize_filename(name):
    """Removes or replaces characters invalid for filenames."""
    name = name.strip()
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name

# --- Worker Function: Processes a single JSON file ---
def evaluate_single_file(filepath, allowed_task_types_set):
    """
    Loads a JSON file, evaluates specific predictions if the task type is allowed,
    and returns a list of result dictionaries, one for each VALID evaluated prediction.
    Skips predictions that are None or become empty after extraction.
    """
    teds_instance = None
    teds_helpers_available = True

    evaluated_predictions = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        record_id = data.get('record_id', os.path.basename(filepath))
        task_type = data.get("type", "Unknown")

        should_evaluate_this_type = (allowed_task_types_set is None or task_type in allowed_task_types_set)

        ground_truth = data.get("ground_truth_answers", [])
        # Only print warning if evaluation was expected for this type
        if not ground_truth and should_evaluate_this_type:
             print(f"Warning: Missing ground_truth_answers for record {record_id}, cannot evaluate.")

        original_common_data = {
            "id": record_id,
            "image_path": data.get("image_path"),
            "question": data.get("question"),
            "type": task_type,
            "answers": ground_truth,
            "dataset_name": data.get("dataset_name")
        }
        original_common_data = {k: v for k, v in original_common_data.items() if v is not None}

        for key, raw_prediction_value in data.items():
            # --- Identify target prediction fields ---
            # Check BOTH ocr_qa and vqa prefixes now based on category mapping logic
            if key.startswith("ocr_qa") or key.startswith("vqa"):

                # --- *** NEW: Skip if prediction value is None/Null *** ---
                if raw_prediction_value is None:
                    # print(f"Debug: Skipping None prediction for key '{key}' in record {record_id}") # Optional debug log
                    continue # Skip to the next key in the file

                # --- Evaluate only if type is allowed and GT exists ---
                if should_evaluate_this_type and ground_truth:
                    prediction_for_eval = extract_prediction_content(raw_prediction_value)

                    # --- *** NEW: Skip if prediction is empty after extraction *** ---
                    if not prediction_for_eval: # Checks for None or empty string ""
                        # print(f"Debug: Skipping empty prediction after extraction for key '{key}' in record {record_id}") # Optional debug log
                        continue # Skip to the next key

                    score = 0.0
                    eval_error = None
                    try:
                        # --- Evaluation Logic (Same extensive if/elif block as previous version) ---
                        # ... (Assume the block is here and correct) ...
                        if task_type in ["APP agent en", "ASCII art classification en", "math QA en", "reasoning VQA en", "science QA en", "text recognition en", "document classification en", "cognition VQA en", "diagram QA en"]:
                            if data.get("eval") == "case sensitive": score = vqa_evaluation_case_sensitive(prediction_for_eval, ground_truth)
                            else: score = vqa_evaluation(prediction_for_eval, ground_truth)
                        elif task_type in ["cognition VQA cn", "reasoning VQA cn"]:
                            if data.get("eval") == "case sensitive": score = vqa_evaluation_case_sensitive(prediction_for_eval, ground_truth)
                            else: score = cn_vqa_evaluation(prediction_for_eval, ground_truth)
                        elif task_type == "handwritten answer extraction cn":
                            if "简答" in str(data.get("question", "")):
                                ocr_metric = cal_per_metrics(prediction_for_eval, ground_truth[0])
                                score = ( get_value_or_zero(ocr_metric.get("bleu")) + get_value_or_zero(ocr_metric.get("meteor")) + get_value_or_zero(ocr_metric.get("f_measure")) + (1 - get_value_or_zero(ocr_metric.get("edit_dist"))) ) / 4
                            else: score = 1 if ground_truth and ground_truth[0] in prediction_for_eval else 0
                        elif task_type == "formula recognition cn":
                            if is_nan_value(prediction_for_eval): score = 0
                            else: score = cn_math_expression_evaluation(prediction_for_eval, ground_truth)
                        elif task_type == "text counting en":
                            score = counting_evaluation(prediction_for_eval, ground_truth, data.get("eval"))
                        elif task_type == "formula recognition en":
                            score = math_expression_evaluation(prediction_for_eval, ground_truth)
                        elif task_type in ["table parsing en", "table parsing cn", "chart parsing en", "document parsing en", "document parsing cn"]:
                             try:
                                 if teds_instance is None: 
                                     teds_instance = TEDS(n_jobs=1)
                                 if isinstance(ground_truth, list) and ground_truth:
                                     gt_table_str = ground_truth[0]
                                     score = doc_parsing_evaluation(prediction_for_eval, gt_table_str)
                                 else: 
                                     score = 0
                             except Exception as teds_err:
                                 eval_error = f"TEDS Error: {teds_err}"
                                 raise Exception(eval_error)
                        elif task_type == "key information extraction en" or task_type == "key information extraction cn":
                            if not teds_helpers_available: eval_error = "KIE helpers not available"
                            else:
                                try:
                                    answers_str = ground_truth[0] if ground_truth else "{}"
                                    answers_dict = ast.literal_eval(answers_str) if isinstance(answers_str, str) else {}
                                    answers_dict = {k: v if isinstance(v, list) else [v] for k, v in answers_dict.items()}
                                    possible_answers = generate_combinations(answers_dict)
                                    max_score = 0; pred_kie_dict = convert_str_to_dict(prediction_for_eval)
                                    for ans_combo in possible_answers: f1 = compute_f1_score(pred_kie_dict, ans_combo); max_score = max(max_score, f1)
                                    score = max_score
                                except Exception as kie_err: eval_error = f"KIE Error: {kie_err}"
                        elif task_type == "VQA with position en":
                            if not teds_helpers_available: eval_error = "VQA Position helpers not available"
                            else:
                                try: pred_dict = convert_str_to_dict(prediction_for_eval); score = vqa_with_position_evaluation(pred_dict, data)
                                except Exception as vqa_pos_err: eval_error = f"VQA Pos Error: {vqa_pos_err}"
                        elif task_type == "text translation cn":
                            gt = ground_truth[0] if ground_truth else ""; score = 0
                            if prediction_for_eval and gt:
                                ocr_metric = cal_per_metrics(prediction_for_eval, gt)
                                score = ( get_value_or_zero(ocr_metric.get("bleu")) + get_value_or_zero(ocr_metric.get("meteor")) + get_value_or_zero(ocr_metric.get("f_measure")) + (1 - get_value_or_zero(ocr_metric.get("edit_dist"))) ) / 4
                        elif task_type == "fine-grained text recognition en":
                            gt = ground_truth[0] if ground_truth else ""; score = 0
                            if prediction_for_eval and gt:
                                ocr_metric = cal_per_metrics(prediction_for_eval, gt)
                                score = ( get_value_or_zero(ocr_metric.get("bleu")) + get_value_or_zero(ocr_metric.get("meteor")) + get_value_or_zero(ocr_metric.get("f_measure")) + (1 - get_value_or_zero(ocr_metric.get("edit_dist"))) ) / 4
                        elif task_type == "full-page OCR en" or task_type == "full-page OCR cn":
                            gt = ground_truth[0] if ground_truth else ""; score = 0
                            if prediction_for_eval and gt:
                                ocr_metric = cal_per_metrics(prediction_for_eval, gt)
                                score = ( get_value_or_zero(ocr_metric.get("bleu")) + get_value_or_zero(ocr_metric.get("meteor")) + get_value_or_zero(ocr_metric.get("f_measure")) + (1 - get_value_or_zero(ocr_metric.get("edit_dist"))) ) / 4
                        elif task_type == "text grounding en":
                            predict_bbox = extract_coordinates(prediction_for_eval); score = 0
                            if predict_bbox: score = calculate_iou(predict_bbox, ground_truth)
                        elif task_type == "text spotting en":
                            predict_bbox_dict = extract_bounding_boxes_robust(prediction_for_eval); score = 0
                            if predict_bbox_dict: score = spotting_evaluation(predict_bbox_dict, data)
                        else:
                             eval_error = f"No evaluation logic defined for task type: {task_type}"
                             score = 0.0
                        # --- End Evaluation Logic ---

                    except Exception as eval_err:
                         eval_error = f"Evaluation function error: {eval_err}"
                         score = 0.0
                         raise Exception(eval_error)

                    # --- Create result object ONLY if evaluation was attempted ---
                    result_obj = {
                        "original_data": original_common_data,
                        "prediction_key": key,
                        "prediction_value": raw_prediction_value, # Keep raw value for output record
                        "score": score,
                        "error": eval_error
                    }
                    evaluated_predictions.append(result_obj)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {filepath}: {e}")
        return [{"error": f"JSON Decode Error: {e}", "filepath": filepath}]
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return [{"error": f"General Processing Error: {e}", "filepath": filepath}]

    return evaluated_predictions
# --- End Worker Function ---


# --- Main Processing Function ---
# (No changes needed in process_predictions_from_dir itself,
# as the filtering happens in the worker and aggregation/saving already handle the results)
def process_predictions_from_dir(input_dir, output_dir, allowed_task_types=None):
    print(f"Searching for prediction JSON files in: {input_dir}")
    filepaths = glob.glob(os.path.join(input_dir, '*.json'))
    if not filepaths:
        print("Error: No JSON files found in the input directory.")
        return
    print(f"Found {len(filepaths)} JSON files.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    allowed_set = set(allowed_task_types) if allowed_task_types else None
    if allowed_set:
        print(f"Filtering evaluations FOR task types: {allowed_task_types}")
    else:
        print("Evaluating ALL task types found in files.")

    all_results_flat = []
    max_workers = os.cpu_count()
    print(f"Starting parallel evaluation using up to {max_workers} workers...")

    worker_func = functools.partial(evaluate_single_file, allowed_task_types_set=allowed_set)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(
            worker_func,
            tqdm(filepaths, desc="Processing Files")
        )
        for result_list in results_iterator:
            if result_list: all_results_flat.extend(result_list)

    print(f"Parallel processing complete. Collected {len(all_results_flat)} individual prediction results (ignoring None/empty predictions).")

    # --- Aggregate results by OUTPUT CATEGORY ---
    results_by_category = defaultdict(list)
    file_processing_errors = []
    unmapped_keys = set()

    for res in all_results_flat:
        if "error" in res and "prediction_key" not in res:
            file_processing_errors.append(res)
        elif "prediction_key" in res:
            category = res['prediction_key']
            if category:
                results_by_category[category].append(res)
            else:
                unmapped_keys.add(res['prediction_key'])

    if file_processing_errors:
        print(f"\nWARNING: Encountered errors processing {len(file_processing_errors)} files:")
        for err_info in file_processing_errors[:10]: print(f"  - File: {err_info.get('filepath', 'Unknown')}, Error: {err_info.get('error', 'Unknown')}")
        if len(file_processing_errors) > 10: print("  ...")
    if unmapped_keys:
        print(f"\nWARNING: Could not map {len(unmapped_keys)} prediction keys to an output category:")
        for key in sorted(list(unmapped_keys))[:10]: print(f"  - {key}")
        if len(unmapped_keys) > 10: print("  ...")

    # --- Save Output Files Per Category ---
    print(f"\nAggregating and saving results for {len(results_by_category)} categories...")
    saved_files_count = 0

    for category_name in tqdm(results_by_category, desc="Saving Category Files"):
        category_results = results_by_category.get(category_name, [])
        output_records = []
        if not category_results:
            raise Exception(f"Got no results for {category_name}")

        for res in category_results:
            record = res['original_data'].copy()
            record['predict'] = res['prediction_value']
            record['score'] = res['score']
            if res.get('error'): record['eval_error'] = res['error']
            output_records.append(record)

        output_filename = f"{category_name.replace('/','_')}.json"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(output_records, f, ensure_ascii=False, indent=4)
            # Count file as saved even if empty, as per requirement to create all 6
            saved_files_count += 1
        except Exception as e:
            print(f"Error saving file {output_filepath}: {e}")
            raise e

    print(f"Finished saving results. Created/Updated {saved_files_count} category result files.")


    # --- Score Aggregation and Reporting ---
    print("\n--- Final Scores ---")
    scores_by_type = defaultdict(lambda: {'total_score': 0.0, 'count': 0})
    processed_types_found = set()

    for res in all_results_flat:
        if "error" in res or "original_data" not in res: continue
        task_type = res['original_data'].get('type')
        score = res.get('score')
        category = res.get('prediction_key', '')

        if category and task_type and score is not None:
             processed_types_found.add(task_type)
             if allowed_set is None or task_type in allowed_set:
                scores_by_type[task_type]['total_score'] += get_value_or_zero(score)
                scores_by_type[task_type]['count'] += 1

    report_task_types = allowed_task_types if allowed_task_types else sorted(list(processed_types_found))
    overall_total_score = 0.0
    overall_total_predictions = 0

    for task_name in report_task_types:
        task_stats = scores_by_type.get(task_name, {'count': 0})
        if task_stats['count'] > 0:
            mean_score = task_stats['total_score'] / task_stats['count']
            print(f"Task '{task_name}', Total Evaluated Predictions (in target categories): {task_stats['count']}, Average Score: {mean_score:.3f}")
            overall_total_score += task_stats['total_score']
            overall_total_predictions += task_stats['count']
        else:
             if allowed_task_types and task_name in allowed_task_types:
                 print(f"Task '{task_name}', Total Evaluated Predictions: 0 (Type allowed, but no results/scores found in target categories)")

    if overall_total_predictions > 0:
         overall_avg = overall_total_score / overall_total_predictions
         print(f"\nOverall Average Score ({overall_total_predictions} evaluated predictions across allowed types and target categories): {overall_avg:.3f}")
    else:
         print("\nNo predictions were evaluated for the allowed task types in the target categories.")
# --- End Main Processing Function ---
