import json
import os
from pathlib import Path
# Removed docling_core imports as they are not needed for raw DocTags generation
from typing import List

from litellm import completion, acompletion # For LLM calls
import asyncio # For async litellm calls
import base64 # To potentially encode images for LLM OCR
from dotenv import load_dotenv # Import dotenv
import logging # Import logging module
import sys # To access stdout for StreamHandler
import time
import functools

import aiometer

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

RESULTS_OUTPUT_DIR = DATA_DIR / "ocr_results"
RESULTS_OUTPUT_DIR.mkdir(exist_ok=True)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_level = logging.INFO # Set the desired log level (e.g., INFO, DEBUG)
log_file = DATA_DIR / "ocr_pipeline.log"

# Get the root logger
logger = logging.getLogger()
logger.setLevel(log_level)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# --- System Prompts (as provided by the user) ---

LLM_OCR_PROMPT = """
**You are an expert Optical Character Recognition (OCR) and Markdown formatting assistant.** Your primary task is to process a document image provided by the user, accurately extract all textual content, and format this content as a well-structured Markdown document.

**Please adhere to the following instructions when processing the image and generating the Markdown output:**

* **Accurate Text Extraction:** Ensure that all text present in the image is extracted as accurately as possible, capturing all words, characters, and numbers. Pay attention to clarity and legibility in the source image.
* **Logical Reading Order:** Organize the extracted text in a logical reading order, typically from top to bottom and left to right, as it appears in the document.
* **Line Breaks:** Preserve line breaks present in the original document by using single line breaks in the Markdown output.
* **Markdown Formatting for Structural Elements:** Identify and represent the following structural elements using standard Markdown syntax:
    * **Headings:** Detect headings of different levels and use the appropriate number of `#` symbols at the beginning of the line (e.g., `# Heading 1`, `## Subheading`).
    * **Paragraphs:** Separate distinct blocks of text with blank lines to represent paragraphs.
    * **Unordered Lists:** Identify bulleted or dash-started lists and format each item using `-` or `*` at the beginning of the line.
    * **Ordered Lists:** Recognize numbered lists and format each item with the corresponding number followed by a period (e.g., `1. First item`, `2. Second item`).
    * **Code Blocks:** If code snippets are present, enclose them within triple backticks (```) at the beginning and end of the code block. If you can identify the programming language, specify it after the opening backticks (e.g., ```python).
    * **Tables:** Detect tabular data and format it using pipe (`|`) symbols to separate columns and hyphens (`---`) to create the header row separator. Ensure consistent alignment within the table cells. For example:

        ```markdown
        | Header 1 | Header 2 |
        |---|---|
        | Cell 1   | Cell 2   |
        | Cell 3   | Cell 4   |
        ```
* **Handling Unreadable Text:** If any portion of the text in the image is unclear or cannot be accurately recognized, indicate it within the Markdown output using the placeholder `[UNREADABLE]`.
* **Document Type (Optional Guidance):** If the user provides information about the type of document (e.g., "This is a research paper" or "This is an invoice"), please take this into consideration to better identify structural elements.

**The user will now provide the document image.** Please process it according to these instructions and output the full extracted text in Markdown format.
"""

# --- Helper Functions ---

def load_ocrbench_data(json_path: Path) -> list:
    """Loads the OCRBench v2 JSON data."""
    if not json_path.exists():
        logger.error(f"OCRBench JSON file not found at {json_path}")
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} records from {json_path}")
        return data
    except json.JSONDecodeError:
        logger.exception(f"Could not decode JSON from {json_path}")
        return []
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading {json_path}: {e}")
        return []

def get_full_image_path(relative_path: str) -> Path | None:
    """Constructs the full image path using the base path."""
    # Basic security check to prevent path traversal
    if ".." in relative_path:
        logger.warning(f"Potentially unsafe relative path detected: {relative_path}. Skipping.")
        return None
    try:
        full_path = OCRBench_V2_IMAGE_BASE_PATH.joinpath(relative_path).resolve()
        # Extra check: ensure the resolved path is still within the base directory
        if OCRBench_V2_IMAGE_BASE_PATH.resolve() not in [full_path] + list(full_path.parents):
             logger.warning(f"Resolved path {full_path} is outside the base directory {OCRBench_V2_IMAGE_BASE_PATH}. Skipping.")
             return None
        return full_path
    except Exception as e:
        logger.exception(f"Error resolving path for {relative_path} within {OCRBench_V2_IMAGE_BASE_PATH}: {e}")
        return None


def encode_image_base64(image_path: Path) -> str | None:
    """Encodes an image file into base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path} for base64 encoding.")
        return None
    except Exception as e:
        logger.exception(f"Error encoding image {image_path} to base64: {e}")
        return None

async def perform_llm_ocr(image_path: Path, model_name: str) -> str | None:
    """
    Performs OCR using a specified multimodal LLM via LiteLLM and returns Markdown.
    Requires the chosen model to support base64 image input in the standard message format.
    """
    if not image_path or not image_path.exists():
        logger.warning(f"Image not found: {image_path}. Skipping LLM OCR.")
        return None

    logger.info(f"Performing LLM OCR ({model_name}) on: {image_path.name}")

    # Encode the image to base64
    base64_image = encode_image_base64(image_path)
    if not base64_image:
        return None # Error message handled in encode_image_base64

    # Construct messages for multimodal input (common format for OpenAI, Anthropic, Google)
    # Check LiteLLM docs for provider-specific variations if needed.

    mime_type = image_path.suffix.lower().strip('.').replace('jpg', 'jpeg')
    messages = [
        {
            "role": "system",
            "content": LLM_OCR_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Perform OCR on this image and provide the output in Markdown format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # Determine image MIME type (common types)
                        "url": f"data:image/{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        # Use LiteLLM's async completion function
        response = await acompletion(
            model=model_name,
            messages=messages,
            # Add any model-specific parameters here (e.g., max_tokens, temperature)
            # max_tokens=4096 # Example: Set max tokens for the response
        )
        # Extract the Markdown output from the response
        markdown_output = response.choices[0].message.content
        logger.info(f"LLM OCR ({model_name}) completed for: {image_path.name}")
        return markdown_output
    except Exception as e:
        logger.exception(f"Error during LiteLLM OCR ({model_name}) for {image_path.name}: {e}")
        # You might want more specific error handling here depending on LiteLLM exceptions
        if "authentication" in str(e).lower():
             logger.error("Authentication error: Please check your API keys in the .env file or environment variables.")
        elif "rate limit" in str(e).lower():
             logger.error("Rate limit exceeded. Please check your plan or wait.")
        # Add more specific checks if needed
        return None

async def perform_llm_vqa(image_path: Path, model_name: str, question: str) -> str | None:
    """
    Requires the chosen model to support base64 image input in the standard message format.
    """
    if not image_path or not image_path.exists():
        logger.warning(f"Image not found: {image_path}. Skipping LLM VQA.")
        return None

    logger.info(f"Performing LLM VQA ({model_name}) on: {image_path.name}")

    # Encode the image to base64
    base64_image = encode_image_base64(image_path)
    if not base64_image:
        return None # Error message handled in encode_image_base64

    # Construct messages for multimodal input (common format for OpenAI, Anthropic, Google)
    # Check LiteLLM docs for provider-specific variations if needed.
    mime_type = image_path.suffix.lower().strip('.').replace('jpg', 'jpeg')
    messages = [
        {
            "role": "system",
            "content": LLM_OCR_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # Determine image MIME type (common types)
                        "url": f"data:image/{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        # Use LiteLLM's async completion function
        response = await acompletion(
            model=model_name,
            messages=messages,
            # Add any model-specific parameters here (e.g., max_tokens, temperature)
            # max_tokens=4096 # Example: Set max tokens for the response
        )
        # Extract the Markdown output from the response
        markdown_output = response.choices[0].message.content
        logger.info(f"LLM VQA ({model_name}) completed for: {image_path.name}")
        return markdown_output
    except Exception as e:
        logger.exception(f"Error during LiteLLM VQA ({model_name}) for {image_path.name}: {e}")
        # You might want more specific error handling here depending on LiteLLM exceptions
        if "authentication" in str(e).lower():
             logger.error("Authentication error: Please check your API keys in the .env file or environment variables.")
        elif "rate limit" in str(e).lower():
             logger.error("Rate limit exceeded. Please check your plan or wait.")
        # Add more specific checks if needed
        return None


async def answer_ocr_question_litellm(context: str, question: str, model_name: str) -> str | None:
    """
    Answers a question based on the provided context using LiteLLM,
    with a separate system prompt.
    """
    if not context or not question:
        logger.warning("Missing context or question for QA. Skipping.")
        return None

    logger.info(f"Answering question using {model_name}...")
    # Format the prompt using the appropriate template
    # Construct the user prompt containing the context and question
    # Determine context type based on system prompt content (heuristic)
    context_label = "Context"
    context_wrapper = "```Context"

    user_prompt = f"""
    Answer the question using the following context:

    {context_label}:
    {context_wrapper}
    {context}
    {context_wrapper}

    Question:
    {question}

    Answer:
    """


    # Prepare messages list with system and user roles
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Call LiteLLM's async completion
        response = await acompletion(
            model=model_name,
            messages=messages
            # Add any model-specific parameters here (e.g., max_tokens, temperature)
        )
        # Extract the answer from the response
        answer = response.choices[0].message.content.strip() # Strip leading/trailing whitespace
        logger.info("Question answering completed.")
        return answer
    except Exception as e:
        logger.exception(f"Error during LiteLLM QA ({model_name}): {e}")
        if "authentication" in str(e).lower():
             logger.error("Authentication error: Please check your API keys in the .env file or environment variables.")
        elif "rate limit" in str(e).lower():
             logger.error("Rate limit exceeded. Please check your plan or wait.")
        return None

def save_results(record_id: int | str, results: dict, output_dir: Path):
    """Saves the processing results for a single record to a JSON file."""
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define the output file path
    output_path = output_dir / f"result_{record_id}.json"
    try:
        # Write the results dictionary to the JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.exception(f"Error saving results to {output_path}: {e}")

# --- Main Workflow ---

async def process_record(record: dict, ocr_models: List[str], ocr_qa_model: str, qa_models: List[str]):
    """Processes a single record from the OCRBench v2 dataset."""
    # Extract necessary information from the record
    record_id = record.get("id", "unknown")

    # TODO RESIZE FOR GROQ {"error":{"message":"Image too large - images can contain at most 33177600 pixels. but image contained 34806376","type":"invalid_request_error"}}
    # TODO CHANGE TIF TO PNG - currently done manually with imagemagiick
    image_relative_path = record.get("image_path").replace(".tif", ".png") # mogrified all tifs
    question = record.get("question")
    ground_truth_answers = record.get("answers", [])
    mime_type = Path(image_relative_path).suffix.lower().strip('.').replace('jpg', 'jpeg')
    if mime_type not in ["png", "jpeg", "gif", "webp"]:
        logger.info(f"Skipping {image_relative_path} due to mime type {mime_type}")
        return

    logger.info(f"--- Processing Record ID: {record_id} ---")

    # Validate essential data
    if not image_relative_path or not question:
        logger.warning(f"Record {record_id} missing image path or question. Skipping.")
        return
    if record_id == "unknown":
         logger.warning(f"Record missing 'id'. Using fallback name for output file.")


    # 1. Image Retrieval (Construct full path)
    full_path = get_full_image_path(image_relative_path)
    if not full_path: # Handle case where path construction failed (e.g., unsafe path)
        logger.error(f"Could not get valid image path for record {record_id}. Skipping.")
        return
    # Check if image file actually exists before proceeding
    if not full_path.exists():
        logger.error(f"Image file not found at resolved path: {full_path}. Skipping record {record_id}.")
        return

    # Initialize results dictionary OR load existing if resuming
    result_file_path = RESULTS_OUTPUT_DIR / f"result_{record_id}.json"
    if result_file_path.exists():
        logger.info(f"Found existing result file for {record_id}, loading previous results.")
        try:
            with open(result_file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            results.setdefault("errors", [])
        except Exception as e:
            logger.exception(f"Error loading existing result file {result_file_path}. Starting fresh.")
            results = {
                "record_id": record_id,
                "image_path": str(full_path),
                "question": question,
                "ground_truth_answers": ground_truth_answers,
                "type": record.get("type"),
                "errors": [] # List to store any errors encountered
            }
            for ocr_model in ocr_models:
                results[f"md_ocr_{ocr_model}"] = None
                results[f"ocr_qa_{ocr_model}_{ocr_qa_model}"] = None
            for qa_model in qa_models:
                results[f"vqa_{qa_model}"] = None
    else:
        # Initialize results dictionary
        results = {
            "record_id": record_id,
            "image_path": str(full_path),
            "question": question,
            "ground_truth_answers": ground_truth_answers,
            "type": record.get("type"),
            "errors": [] # List to store any errors encountered
        }
        for ocr_model in ocr_models:
            results[f"md_ocr_{ocr_model}"] = None
            results[f"ocr_qa_{ocr_model}_{ocr_qa_model}"] = None
        for qa_model in qa_models:
            results[f"vqa_{qa_model}"] = None

    results["errors"] = []

    logger.info(f"Image path: {full_path}")

    # 1. OCR Inference

    for ocr_model in ocr_models:
        if results.get(f"md_ocr_{ocr_model}") is not None:
            logger.info(f"Skipping Markdown OCR For Model {ocr_model} for {record_id} - already found in results.")
            llm_markdown_output = results.get(f"md_ocr_{ocr_model}")
        else:
            # 2b. LLM OCR -> Markdown (using LiteLLM)
            llm_markdown_output = await perform_llm_ocr(full_path, model_name=ocr_model)
            if llm_markdown_output is None:
                 results["errors"].append(f"LLM OCR ({ocr_model}) failed or was skipped.")
            results[f"md_ocr_{ocr_model}"] = llm_markdown_output
            save_results(record_id, results, RESULTS_OUTPUT_DIR)

            if results.get("markdown_qa_answer") is not None:
                logger.info(f"Skipping Markdown OCR QA For Model {ocr_model} for {record_id} - already found in results.")
        if results.get(f"ocr_qa_{ocr_model}_{ocr_qa_model}") is not None:
            logger.info(f"Skipping OCR QA For Model {ocr_model} for {record_id} - already found in results.")
        else:
            if llm_markdown_output:
                markdown_answer = await answer_ocr_question_litellm(
                    context=llm_markdown_output,
                    question=question,
                    model_name=ocr_qa_model
                )
                if markdown_answer is None:
                    results["errors"].append(f"Markdown QA ({ocr_model}/{ocr_qa_model}) failed.")
                results[f"ocr_qa_{ocr_model}_{ocr_qa_model}"] = markdown_answer
                save_results(record_id, results, RESULTS_OUTPUT_DIR)
            else:
                logger.info("Skipping Markdown QA as LLM OCR failed or was skipped.")

    for qa_model in qa_models:
        if results.get(f"vqa_{qa_model}") is not None:
            logger.info(f"Skipping Markdown VQA For Model {qa_model} for {record_id} - already found in results.")
            continue
        # 2b. LLM OCR -> Markdown (using LiteLLM)
        qa_output = await perform_llm_vqa(full_path, question=question, model_name=qa_model)
        if qa_output is None:
             results["errors"].append(f"LLM OCR ({qa_model}) failed or was skipped.")
        results[f"vqa_{qa_model}"] = qa_output
        save_results(record_id, results, RESULTS_OUTPUT_DIR)

    # 4. Save Results
    save_results(record_id, results, RESULTS_OUTPUT_DIR)
    logger.info(f"--- Finished Processing Record ID: {record_id} ---")


async def main(ocr_models: List[str], ocr_qa_model: str, qa_models: List[str], question_types: List[str], max_workers: int):
    """Main function to load data and process records."""

    logger.info(f"Using LLM OCR Models: {ocr_models}")
    logger.info(f"Using LLM OCR QA Model: {ocr_qa_model}")
    logger.info(f"Using LLM Visual QA Models: {qa_models}")

    # --- Verify Data Path ---
    if not OCRBench_V2_IMAGE_BASE_PATH.exists() or not OCRBench_V2_JSON_PATH.exists():
         logger.critical(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         logger.critical(f"!!! ERROR: OCRBench data not found at the specified path:                !!!")
         logger.critical(f"!!!        {OCRBench_V2_IMAGE_BASE_PATH.resolve()} ")
         logger.critical(f"!!!        Ensure the path is correct and the data exists.               !!!")
         logger.critical(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         return # Exit if data path is incorrect

    # --- Load Data ---
    ocr_data = [r for r in load_ocrbench_data(OCRBench_V2_JSON_PATH) if r["type"] in question_types]

    if not ocr_data:
        logger.error("Exiting due to data loading issues (JSON file might be empty or corrupt).")
        return

    # --- Process Records ---
    # Limit the number of records for testing if needed
    # records_to_process = ocr_data[:5] # Process only the first 5 records
    #records_to_process = ocr_data[:2] # Using subset from user's file
    records_to_process = ocr_data

    logger.info(f"Starting processing for {len(records_to_process)} out of {len(ocr_data)} records...")
    # Create asynchronous tasks for each record
    tasks = [functools.partial(process_record, record, ocr_models, ocr_qa_model, qa_models) for record in records_to_process]
    # Run tasks concurrently
    await aiometer.run_all(tasks, max_at_once=max_workers)

    logger.info("\n--- All specified records processed ---")

if __name__ == "__main__":
    # Model definitions from user's file
    MAX_WORKERS = 1  # Concurrent model calls
    LLM_OCR_MODELS = ["gemini/gemini-2.0-flash", "groq/meta-llama/llama-4-scout-17b-16e-instruct", "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"]
    OCR_QA_MODEL = "gemini/gemini-2.0-flash"
    VQA_MODELS = ["gemini/gemini-2.0-flash", "groq/meta-llama/llama-4-scout-17b-16e-instruct",  "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"] # Updated from user file (was 2.0-flash)
    question_types = [
        #"APP agent en",
        #"ASCII art classification en",
        #"chart parsing en"
        #"chart parsing en",
        #"cognition VQA cn"
        #"cognition VQA en",
        #"cognition VQA en",
        #"diagram QA en",
        #"diagram QA en",
        "document classification en",
        #"document parsing cn",
        "document parsing en",
        #"fine-grained text recognition en",
        #"formula recognition cn",
        #"formula recognition en",
        #"full-page OCR cn"
        "full-page OCR en",
        #"handwritten answer extraction cn",
        #"key information extraction cn",
        "key information extraction en",
        #"key information mapping en",
        #"math QA en",
        #"reasoning VQA cn",
        "reasoning VQA en",
        #"science QA en",
        #"table parsing cn",
        #"table parsing en"
        #"table parsing en",
        #"text counting en",
        #"text grounding en",
        #"text recognition en",
        #"text spotting en",
        #"text translation cn",
        #"VQA with position en",
    ]

    # Crucial check for the base path configuration - Using path from user's file
    if not OCRBench_V2_IMAGE_BASE_PATH or str(OCRBench_V2_IMAGE_BASE_PATH) == "/path/to/your/OCRBench_v2/": # Keep original check placeholder logic
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.critical("!!! ERROR: Please set the OCRBench_V2_IMAGE_BASE_PATH variable correctly !!!")
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not OCRBench_V2_IMAGE_BASE_PATH.exists():
         logger.critical(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         logger.critical(f"!!! ERROR: The specified OCRBench path does not exist:                 !!!")
         logger.critical(f"!!!        {OCRBench_V2_IMAGE_BASE_PATH.resolve()} ")
         logger.critical(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        asyncio.run(main(LLM_OCR_MODELS, OCR_QA_MODEL, VQA_MODELS, question_types, MAX_WORKERS))


