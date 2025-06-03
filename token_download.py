from transformers import RobertaTokenizer
import os

# Download CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Save tokenizer vocabulary to the working directory
output_path = os.path.join(os.getcwd(), "codebert_tokenizer")
tokenizer.save_pretrained(output_path)
print(f"Saved CodeBERT tokenizer to {output_path}")