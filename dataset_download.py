import os
import json
import shutil
from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import login, hf_hub_download
import logging
from typing import Optional, List, Dict

def download_datasets(save_dir: str = "datasets", hf_token: Optional[str] = None) -> bool:
    """
    Download specified Hugging Face datasets to the working directory if not already present.
    Optimized for ~9.9GB VRAM by using Python lists and subsampling large datasets.
    
    Args:
        save_dir (str): Directory to save datasets.
        hf_token (Optional[str]): Hugging Face API token for private datasets.
    
    Returns:
        bool: True if processing completes successfully, False otherwise.
    """
    # Input validation
    if not isinstance(save_dir, str) or not save_dir:
        logging.error("Invalid save_dir: must be a non-empty string")
        return False
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Suppress symlink warnings on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Securely handle Hugging Face token
    if hf_token:
        try:
            login(hf_token)
            logging.info("Logged in to Hugging Face")
        except Exception as e:
            logging.error(f"Failed to log in to Hugging Face: {e}")
            return False
    else:
        logging.info("No Hugging Face token provided. Public datasets should still work.")
    
    # Create save directory
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create save directory {save_dir}: {e}")
        return False
    
    # Define dataset configurations with nq_open replacing natural_questions
    dataset_configs: List[Dict] = [
        {"name": "code_search_net", "repo": "code_search_net", "split": "train", "trust_remote_code": True},
        {"name": "human_eval", "repo": "openai_humaneval", "split": "test"},
        {"name": "mbpp", "repo": "google-research-datasets/mbpp", "split": "train"},
        {"name": "mmlu", "repo": "cais/mmlu", "split": "test", "config": "all"},
        {"name": "open_r1_math", "repo": "open-r1/OpenR1-Math-220k", "split": "train", "trust_remote_code": True},
        {"name": "sciq", "repo": "allenai/sciq", "split": "train"},
        {"name": "gsm8k", "repo": "openai/gsm8k", "split": "train", "config": "main"},
        {"name": "arc_easy", "repo": "allenai/ai2_arc", "split": "train", "config": "ARC-Easy"},
        {"name": "arc_challenge", "repo": "allenai/ai2_arc", "split": "train", "config": "ARC-Challenge"},
        {"name": "stem", "repo": "stemdataset/STEM", "split": "train"},
        {"name": "daily_dialog", "repo": "daily_dialog", "split": "train", "trust_remote_code": True},
        {"name": "personachat", "repo": "bavard/personachat_truecased", "split": "train", "trust_remote_code": True},
        {"name": "open_assistant", "repo": "OpenAssistant/oasst1", "split": "train"},
        {"name": "code_alpaca", "repo": "sahil2801/CodeAlpaca-20k", "split": "train"},
        {"name": "alpaca", "repo": "tatsu-lab/alpaca", "split": "train"},
        {"name": "dolly", "repo": "databricks/databricks-dolly-15k", "split": "train"},
        {"name": "truthfulqa", "repo": "truthfulqa/truthful_qa", "split": "validation", "config": "generation", "trust_remote_code": True},
        {"name": "kjv_bible", "repo": "JDRJ/kjv-bible", "split": "train"},
        {"name": "hellaswag", "repo": "Rowan/hellaswag", "split": "train"},
        {"name": "apps", "repo": "codeparrot/apps", "split": "train", "trust_remote_code": True},
        {"name": "mathqa", "repo": "allenai/math_qa", "split": "train", "trust_remote_code": True},
        {"name": "strategyqa", "repo": "tau/strategyqa", "split": "train"},
        {"name": "codecontests", "repo": "deepmind/code_contests", "split": "train"},
        {"name": "nq_open", "repo": "google-research-datasets/nq_open", "split": "train", "trust_remote_code": True},  # Replaced natural_questions
        {"name": "aqua_rat", "repo": "deepmind/aqua_rat", "split": "train"},
        {"name": "ultrafeedback", "repo": "HuggingFaceH4/ultrafeedback_binarized", "split": "train_sft"},
    ]
    
    # BBH configurations (subset to manage memory)
    bbh_configs = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "logical_deduction_five_objects",
        "object_counting",
        "word_sorting"
    ]
    
    logging.info(f"Checking datasets in {os.path.abspath(save_dir)}...")
    
    for config in dataset_configs:
        dataset_name = config["name"]
        repo = config["repo"]
        split = config["split"]
        dataset_config = config.get("config")
        trust_remote_code = config.get("trust_remote_code", False)
        save_path = os.path.join(save_dir, dataset_name)
        
        # Handle BBH specially due to multiple configs
        if dataset_name == "bbh":
            for bbh_config in bbh_configs:
                bbh_save_path = os.path.join(save_dir, f"bbh_{bbh_config}")
                if os.path.exists(bbh_save_path):
                    try:
                        dataset = load_from_disk(bbh_save_path)
                        logging.info(f"Dataset bbh/{bbh_config} already exists at {bbh_save_path}, skipping download.")
                        continue
                    except Exception as e:
                        logging.error(f"Error loading bbh/{bbh_config} from disk: {e}. Redownloading...")
                        shutil.rmtree(bbh_save_path, ignore_errors=True)
                
                try:
                    logging.info(f"Downloading bbh/{bbh_config} ({repo}, {split})...")
                    dataset = load_dataset(
                        repo,
                        bbh_config,
                        split=split,
                        cache_dir=os.path.join(save_dir, "cache"),
                        trust_remote_code=trust_remote_code
                    )
                    # Subsample BBH to reduce memory footprint
                    dataset = dataset.shuffle(seed=42).select(range(min(5000, len(dataset))))
                    dataset.save_to_disk(bbh_save_path)
                    logging.info(f"Successfully downloaded bbh/{bbh_config} to {bbh_save_path}")
                except Exception as e:
                    logging.error(f"Error downloading bbh/{bbh_config}: {e}")
                    continue
            continue
        
        # Check if dataset already exists
        if os.path.exists(save_path):
            try:
                dataset = load_from_disk(save_path)
                logging.info(f"Dataset {dataset_name} already exists at {save_path}, skipping download.")
                continue
            except Exception as e:
                logging.error(f"Error loading {dataset_name} from disk: {e}. Redownloading...")
                shutil.rmtree(save_path, ignore_errors=True)
        
        try:
            logging.info(f"Downloading {dataset_name} ({repo}, {split})...")
            
            if dataset_name == "strategyqa":
                # Special handling for StrategyQA: Download and preprocess JSON directly
                cache_dir = os.path.join(save_dir, "cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                # Download the JSON file
                filename = f"strategyqa_{split}.json"
                json_file = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    repo_type="dataset",
                    cache_dir=cache_dir
                )
                logging.info(f"Downloaded {filename} for {dataset_name}")
                
                # Load JSON into memory as a list
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Robust preprocessing of evidence column
                for i, item in enumerate(data):
                    if "evidence" in item:
                        evidence = item["evidence"]
                        if isinstance(evidence, list):
                            item["evidence"] = [str(e) for e in evidence]
                        elif evidence is None:
                            item["evidence"] = []
                        else:
                            logging.debug(f"Converting non-list evidence at index {i}: {evidence}")
                            item["evidence"] = [str(evidence)]
                
                # Create dataset from list to optimize memory
                dataset = Dataset.from_list(data)
                
                # Save to disk
                dataset.save_to_disk(save_path)
                logging.info(f"Successfully processed and saved {dataset_name} to {save_path}")
            
            else:
                # Standard dataset loading with subsampling for large datasets
                if dataset_name in ["nq_open", "ultrafeedback", "aqua_rat", "apps"]:
                    # Subsample large datasets to manage VRAM (~9.9GB)
                    dataset = load_dataset(
                        repo,
                        dataset_config,
                        split=split,
                        cache_dir=os.path.join(save_dir, "cache"),
                        trust_remote_code=trust_remote_code
                    )
                    dataset = dataset.shuffle(seed=42).select(range(min(5000, len(dataset))))
                elif dataset_name == "codecontests":
                    # Subsample codecontests due to longer sequences
                    dataset = load_dataset(
                        repo,
                        dataset_config,
                        split=split,
                        cache_dir=os.path.join(save_dir, "cache"),
                        trust_remote_code=trust_remote_code
                    )
                    dataset = dataset.shuffle(seed=42).select(range(min(3000, len(dataset))))
                else:
                    # Load full dataset for smaller or non-subsampled datasets
                    if dataset_config:
                        dataset = load_dataset(
                            repo,
                            dataset_config,
                            split=split,
                            cache_dir=os.path.join(save_dir, "cache"),
                            trust_remote_code=trust_remote_code
                        )
                    else:
                        dataset = load_dataset(
                            repo,
                            split=split,
                            cache_dir=os.path.join(save_dir, "cache"),
                            trust_remote_code=trust_remote_code
                        )
                
                # Save to disk
                dataset.save_to_disk(save_path)
                logging.info(f"Successfully downloaded {dataset_name} to {save_path}")
        except Exception as e:
            logging.error(f"Error downloading {dataset_name}: {e}")
            continue
    
    # Clean up cache directory
    cache_path = os.path.join(save_dir, "cache")
    shutil.rmtree(cache_path, ignore_errors=True)
    
    logging.info("Dataset processing complete!")
    return True

if __name__ == "__main__":
    # Specify your Hugging Face token here
    HF_TOKEN = "token_here"  # Replace with your token from huggingface.co/settings/tokens
    
    # Download datasets
    success = download_datasets(save_dir="datasets", hf_token=HF_TOKEN)
    if not success:
        logging.error("Dataset downloading failed.")