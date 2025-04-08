# OCRBenchv2 Eval

Official OCRBenchv2 Repo:
https://github.com/Yuliang-Liu/MultimodalOCR

Make sure you download the OCRBench2 from the dataset and extract in the DATA_DIR folder specified in .env:

https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2#data

# Usage

### Install basic dependencies
```
pip install -r requirements.txt
```

### Install needed dependencies
```
just install
```

### Run OCR
```
just ocr
```

### Run Eval
```
just eval
```

### Change Models/Eval Metrics

As configured, the following metrics are run:

     - document classification en
     - document parsing en
     - full-page OCR en
     - key information extraction en
     - reasoning VQA en

The following are disabled:

     - APP agent en
     - ASCII art classification en
     - chart parsing en
     - chart parsing en
     - cognition VQA cn
     - cognition VQA en
     - cognition VQA en
     - diagram QA en
     - diagram QA en
     - document parsing cn
     - fine-grained text recognition en
     - formula recognition cn
     - formula recognition en
     - full-page OCR cn
     - handwritten answer extraction cn
     - key information extraction cn
     - key information mapping en
     - math QA en
     - reasoning VQA cn
     - science QA en
     - table parsing cn
     - table parsing en
     - table parsing en
     - text counting en
     - text grounding en
     - text recognition en
     - text spotting en
     - text translation cn
     - VQA with position en

The following models are currently chosen:


```python
LLM_OCR_MODELS = ["gemini/gemini-2.0-flash", "groq/meta-llama/llama-4-scout-17b-16e-instruct", "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"]
OCR_QA_MODEL = "gemini/gemini-2.0-flash"
VQA_MODELS = ["gemini/gemini-2.0-flash", "groq/meta-llama/llama-4-scout-17b-16e-instruct",  "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"] 
```

Change eval metrics in the main area of scripts/ocrbench_run.py and scripts/eval.py

Change models to any model supported by litellm in scripts/ocrbench_run.py

**Make sure you setup your dotenv**

Sample format at .env:
```
DATA_DIR=.data

GEMINI_API_KEY=
GROQ_API_KEY=

AWS_REGION=us-east-1


# AWS Credentials fetched at Tue Apr  8 03:22:16 AM EDT 2025 - Expires: 2025-04-08T19:22:16+00:00
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=
```
