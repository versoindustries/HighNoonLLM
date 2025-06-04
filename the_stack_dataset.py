# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datasets import load_dataset
from huggingface_hub import login

def stream_the_stack(save_subset=False, subset_size=1000, save_dir="datasets/the_stack_subset", language="python", hf_token=None):
    """Stream The Stack dataset and optionally save a subset."""
    # Suppress symlink warnings on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Log in to Hugging Face (required for The Stack)
    if not hf_token:
        raise ValueError("Hugging Face token required for The Stack. Get a token from huggingface.co/settings/tokens")
    login(hf_token)
    print("Logged in to Hugging Face")
    
    # Create save directory if saving a subset
    if save_subset:
        os.makedirs(save_dir, exist_ok=True)
    
    print("Streaming The Stack dataset...")
    # Load dataset in streaming mode
    dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)
    
    # Process dataset
    subset = []
    count = 0
    for item in dataset:
        # Filter by language if specified (e.g., Python)
        if language and item.get("lang", "").lower() != language.lower():
            continue
        
        print(f"Processing item {count + 1}: {item.get('path', 'unknown')}")
        # Example processing: print code snippet
        print(f"Code:\n{item.get('content', '')[:200]}...")
        
        if save_subset:
            subset.append(item)
        
        count += 1
        if count >= subset_size and save_subset:
            print(f"Saving subset of {len(subset)} items to {save_dir}")
            from datasets import Dataset
            subset_dataset = Dataset.from_list(subset)
            subset_dataset.save_to_disk(save_dir)
            print(f"Subset saved to {save_dir}")
            break
    
    if not save_subset:
        print(f"Processed {count} items from The Stack (streaming mode, no save)")

if __name__ == "__main__":
    # Specify your Hugging Face token here
    HF_TOKEN = "your_token_here"  # Replace with your token from huggingface.co/settings/tokens
    
    # Stream dataset and save a small subset
    stream_the_stack(
        save_subset=True,
        subset_size=1000,
        save_dir="datasets/the_stack_subset",
        language="python",
        hf_token=HF_TOKEN
    )