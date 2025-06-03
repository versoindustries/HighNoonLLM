import os
import re
import logging
import importlib
import sys
from PyPDF2 import PdfReader
from datasets import Dataset
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('owasp_preprocess.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def check_dependencies():
    """Check if required libraries are installed and log environment details."""
    logging.info(f"Python executable: {sys.executable}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Site-packages paths: {sys.path}")
    
    required = ['PyPDF2', 'datasets']
    for module in required:
        try:
            importlib.import_module(module)
            logging.info(f"Module '{module}' is installed and importable")
        except ImportError as e:
            logging.error(f"Required module '{module}' is not installed or not found. Install it using 'pip install {module}'")
            logging.error(f"Import error details: {e}")
            raise ImportError(f"Missing module: {module}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2 and save for inspection."""
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return ""
    try:
        logging.info(f"Extracting text from {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        logging.info(f"Extracted {len(text)} characters from {pdf_path}")
        
        # Save extracted text for debugging
        output_text_file = pdf_path.replace('.pdf', '_extracted.txt')
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Saved extracted text to {output_text_file}")
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def parse_owasp_2021(text):
    """Parse the 2021 OWASP Top 10 text to extract risk sections and subsections."""
    if not text:
        logging.warning("Empty text provided for 2021 OWASP parsing")
        return []

    # Relaxed pattern to match A01:2021 - Risk Name with variations
    risk_pattern = r"(A\d{2}\s*:\s*2021\s*[-–—]?\s*.+?)(?=(A\d{2}\s*:\s*2021|$))"
    risks = re.findall(risk_pattern, text, re.DOTALL | re.MULTILINE)
    parsed_data = []
    logging.info(f"Found {len(risks)} risk sections in 2021 OWASP document")

    # Log first 500 characters of text for debugging
    logging.debug(f"First 500 chars of 2021 text: {text[:500]}")
    
    for risk in risks:
        match = re.match(r"A\d{2}\s*:\s*2021\s*[-–—]?\s*(.+)", risk, re.MULTILINE)
        if not match:
            logging.debug(f"Skipping malformed risk section: {risk[:100]}...")
            continue
        risk_name = match.group(1).strip()

        # Log matched risk for debugging
        logging.debug(f"Matched risk: {risk_name}")

        # Extract subsections with relaxed patterns
        description_match = re.search(
            r"(?<=Example Scenario:.*?)(.+?)(?=Notable Common Weakness Enumerations|\Z)",
            risk, re.DOTALL
        )
        examples_match = re.search(
            r"(?<=Example Scenario:)(.+?)(?=Notable Common Weakness Enumerations|\Z)",
            risk, re.DOTALL
        )
        prevention_match = re.search(
            fr"{re.escape(risk_name)}\s*Prevention(.+?)(?=A\d{{2}}\s*:\s*2021|\Z)",
            text, re.DOTALL
        )

        parsed_data.append({
            "risk_name": risk_name,
            "description": description_match.group(1).strip() if description_match else "",
            "examples": examples_match.group(1).strip() if examples_match else "",
            "prevention": prevention_match.group(1).strip() if prevention_match else ""
        })
        logging.debug(f"Parsed risk: {risk_name}")

    return parsed_data

def parse_owasp_2023(text):
    """Parse the 2023 OWASP Top 10 for LLMs to extract risk sections and subsections."""
    if not text:
        logging.warning("Empty text provided for 2023 OWASP parsing")
        return []

    # Relaxed pattern to match LLM01:2025 - Risk Name with variations
    risk_pattern = r"(LLM\d{2}\s*[:–—]?\s*202[0-5]?\s*[-–—]?\s*.+?)(?=(LLM\d{2}\s*[:–—]?\s*202[0-5]?|$))"
    risks = re.findall(risk_pattern, text, re.DOTALL | re.MULTILINE)
    parsed_data = []
    logging.info(f"Found {len(risks)} risk sections in 2023 OWASP document")

    # Log first 500 characters of text for debugging
    logging.debug(f"First 500 chars of 2023 text: {text[:500]}")
    
    for risk in risks:
        match = re.match(r"LLM\d{2}\s*[:–—]?\s*202[0-5]?\s*[-–—]?\s*(.+)", risk, re.MULTILINE)
        if not match:
            logging.debug(f"Skipping malformed risk section: {risk[:100]}...")
            continue
        risk_name = match.group(1).strip()

        # Log matched risk for debugging
        logging.debug(f"Matched risk: {risk_name}")

        # Extract subsections with relaxed patterns
        description_match = re.search(
            r"(?<=Description)(.+?)(?=Types of|Common Examples of|Prevention and Mitigation|\Z)",
            risk, re.DOTALL
        )
        examples_match = re.search(
            r"(?<=Example Attack Scenarios)(.+?)(?=Reference Links|\Z)",
            risk, re.DOTALL
        )
        prevention_match = re.search(
            r"(?<=Prevention and Mitigation Strategies)(.+?)(?=Example Attack Scenarios|Reference Links|\Z)",
            risk, re.DOTALL
        )

        parsed_data.append({
            "risk_name": risk_name,
            "description": description_match.group(1).strip() if description_match else "",
            "examples": examples_match.group(1).strip() if examples_match else "",
            "prevention": prevention_match.group(1).strip() if prevention_match else ""
        })
        logging.debug(f"Parsed risk: {risk_name}")

    return parsed_data

def generate_qa_pairs(parsed_data, year):
    """Generate question-answer pairs from parsed data."""
    qa_pairs = []
    for risk in parsed_data:
        risk_name = risk["risk_name"]
        if risk["description"]:
            qa_pairs.append({
                "question": f"What is {risk_name} according to the {year} OWASP Top 10?",
                "answer": risk["description"]
            })
        if risk["examples"]:
            qa_pairs.append({
                "question": f"What are some examples of {risk_name} in the {year} OWASP Top 10?",
                "answer": risk["examples"]
            })
        if risk["prevention"]:
            qa_pairs.append({
                "question": f"How can {risk_name} be prevented as per the {year} OWASP Top 10?",
                "answer": risk["prevention"]
            })
    logging.info(f"Generated {len(qa_pairs)} Q&A pairs for {year} OWASP Top 10")
    return qa_pairs

def create_owasp_dataset(pdf_2021_path, pdf_2023_path, output_json="owasp_dataset.json"):
    """Create a dataset from OWASP Top 10 PDFs."""
    # Check dependencies
    check_dependencies()

    # Verify PDF files exist
    for pdf_path in [pdf_2021_path, pdf_2023_path]:
        if not os.path.exists(pdf_path):
            logging.error(f"PDF file missing: {pdf_path}")
            return None

    # Extract text
    text_2021 = extract_text_from_pdf(pdf_2021_path)
    text_2023 = extract_text_from_pdf(pdf_2023_path)

    if not text_2021 or not text_2023:
        logging.error("Failed to extract text from one or both PDFs")
        return None

    # Parse text
    parsed_2021 = parse_owasp_2021(text_2021)
    parsed_2023 = parse_owasp_2023(text_2023)

    # Generate Q&A pairs
    qa_2021 = generate_qa_pairs(parsed_2021, 2021)
    qa_2023 = generate_qa_pairs(parsed_2023, 2023)

    # Combine and format dataset
    owasp_data = [
        {"context": item["question"], "target": item["answer"], "task": "chat"}
        for item in qa_2021 + qa_2023
    ]

    if not owasp_data:
        logging.error("No data generated. Check PDF content and parsing logic.")
        return None

    # Save to JSON
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(owasp_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Dataset saved to '{output_json}' with {len(owasp_data)} items")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")
        return None

    # Convert to Hugging Face Dataset
    try:
        dataset = Dataset.from_list(owasp_data)
        logging.info(f"Created Hugging Face Dataset with {len(dataset)} entries")
        return dataset
    except Exception as e:
        logging.error(f"Failed to create Dataset: {e}")
        return None

if __name__ == "__main__":
    # Define PDF paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_2021_path = os.path.normpath(os.path.join(script_dir, "owasp-top-10.pdf"))
    pdf_2023_path = os.path.normpath(os.path.join(script_dir, "LLMAll_en-US_FINAL.pdf"))
    output_json = os.path.normpath(os.path.join(script_dir, "owasp_dataset.json"))

    logging.info("Starting OWASP dataset preprocessing")
    dataset = create_owasp_dataset(pdf_2021_path, pdf_2023_path, output_json)
    if dataset:
        logging.info(f"Dataset creation complete. {len(dataset)} entries generated.")
        print(f"Dataset created with {len(dataset)} entries. Saved to '{output_json}'")
    else:
        logging.error("Dataset creation failed.")
        print("Failed to create dataset. Check logs for details.")