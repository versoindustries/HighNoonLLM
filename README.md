# HighNoon LLM: Hierarchical Spatial Neural Memory

HighNoon LLM is an innovative large language model leveraging Hierarchical Spatial Neural Memory (HSMN) to efficiently process long sequences. By reducing the computational complexity of traditional transformers from $O(n^2)$ to $O(n \cdot c)$—where $c$ is a fixed chunk size (e.g., 128 tokens)—HighNoon enhances scalability and performance for natural language processing (NLP) tasks, including document translation, summarization, and code generation.

## Key Features

- **Efficient Sequence Processing**: Utilizes a hierarchical binary memory tree to manage long contexts with reduced computational overhead.
- **Multi-Task Training**: Supports diverse datasets like SciQ, GSM8K, MMLU, and CodeSearchNet for robust performance across NLP and code generation tasks.
- **Optimized for Resource Constraints**: Targets ~6.3GB VRAM usage with gradient accumulation and 50% model pruning for efficiency.
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting during continual learning across multiple datasets.
- **High Performance**: Achieved 100% accuracy on STEM and SciQ as a classification architecture (reproducible upon request).

## Project Status

- **Training in Progress**: Expected completion by September 2025 due to compute constraints.
- **Code Available**: Open-sourced under the Apache 2.0 License.
- **Model Weights**: Intermediate checkpoints will be released starting July 2025 under the Creative Commons Attribution-NonCommercial 4.0 License for non-commercial use. Commercial use requires a paid license (see `COMMERCIAL-LICENSE.md`).

## Licensing

HighNoon LLM adopts a dual licensing model to balance community access with commercial opportunities:

### Code

- **License**: Apache 2.0 License
- **Usage**: Free for all users, including commercial use. Permits modification, distribution, and contributions with attribution.

### Model Weights

- **Non-Commercial Use**: Available under the Creative Commons Attribution-NonCommercial 4.0 License for researchers, hobbyists, and academics.
- **Commercial Use**: Requires a paid commercial license. See `COMMERCIAL-LICENSE.md` for details.

## Installation

### Prerequisites

- **Python**: 3.10.11
- **TensorFlow**: 2.10.0 (see notes below for GPU-specific requirements)
- **Hardware**: 32GB RAM minimum; NVIDIA or AMD GPU with at least 8GB VRAM recommended
- **System**: Linux, Windows

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/versoindustries/HighNoonLLM.git
   cd highnoon-llm
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### TensorFlow Notes

- **NVIDIA GPU Users**: Use `tensorflow==2.10.0` for optimal performance.
- **AMD GPU Users**: Use `tensorflow-cpu==2.10.1` with `tensorflow-directml-plugin` due to DirectML compatibility. Note that DirectML introduces unlogged overhead (likely from lists/dictionaries in the plugin). We plan to optimize this in future releases.
- **CPU Users**: Install `tensorflow-cpu==2.10.1` for non-GPU setups.

### Download Datasets

Run the dataset download script to fetch training datasets (e.g., SciQ, GSM8K):

```bash
python dataset_download.py
```

### Download Tokenizers

Obtain the pre-trained RobertaTokenizer from the `transformers` library:

```bash
python token_download.py
```

See `docs/SETUP.md` for detailed troubleshooting and advanced setup options.

## Usage

### Training the Model

1. **Prepare the Environment**: Ensure dependencies, datasets, and tokenizers are installed/downloaded successfully.
2. **Launch Training**: Start the training process with the main script:
   ```bash
   python batch_train.py
   ```
3. **Monitor Training**:
   - **Logs**: Real-time training logs are saved to `training_log.log` for monitoring and debugging.
   - **Checkpoints**: Model weights are saved as `hsmn_model_<dataset>_best_epoch_XX.h5` in the specified model path.

### Example

To train on the MMLU dataset:

```bash
python batch_train.py --dataset mmlu
```

Check `batch_train.py` for additional command-line options (e.g., batch size, epochs).

## Project Structure

- `Owasp/`: Contains OWASP guidelines and a preprocess script to convert PDFs into `.txt` files. Planned for conversion into an OWASP dataset to instruct inference for coding.
- `Research/`: Contains research papers on the HSMN architecture and HighNoon Model.
- `batch_train.py`: Main script for orchestrating multi-dataset training.
- `dataset_download.py`: Script to download configured datasets.
- `token_download.py`: Script to set up the RobertaTokenizer.
- `requirements.txt`: Python dependencies.
- `models/`: Placeholder for model weights (checkpoints released starting July 2025).
- `docs/`: Documentation, including setup and contribution guidelines.

## Model Architecture

- **ChunkEncoder**: Splits input sequences into fixed-size chunks (default: 128 tokens) and encodes them into embeddings.
- **Aggregator**: Builds a hierarchical binary memory tree from chunk embeddings for efficient context representation.
- **ReasoningModule**: Generates outputs autoregressively by attending to the memory tree.

## Training Process

- **Data Preprocessing**: Datasets are split into training, validation, and test sets with custom preprocessors (see `src/preprocessors.py`).
- **Continual Learning**: Uses Elastic Weight Consolidation (EWC) to retain knowledge across tasks, preventing catastrophic forgetting.
- **Optimization**: Employs gradient accumulation for large effective batch sizes and model pruning (50% sparsity) for efficiency.
- **Supported Datasets**:
  - SciQ
  - GSM8K
  - MMLU
  - CodeSearchNet
  - HumanEval
  - MBPP
  - And more (see `batch_train.py` for full list).

## Evaluation

- **Benchmarks**: Performance is compared against top models (e.g., GPT-4, Claude 3.5 Sonnet) on datasets like MMLU, HumanEval, and GSM8K.
- **Results**: Saved to `scorecard.txt` post-training.
- **Notable Achievement**: 100% accuracy on STEM and SciQ as a classification architecture. Contact us to replicate these results (requires reverting some LLM-specific engineering).

## Additional Notes

- **Logging**: Detailed logs in `training_log.log` provide transparency and aid troubleshooting.
- **Memory Management**: Optimized for ~6.3GB VRAM usage with GPU memory growth and CPU offloading for chunking.
- **Known Issues**:
  - Dataset-specific preprocessing errors may occur; check `training_log.log` for details.
  - GPU memory overflows possible with large sequences; reduce batch size or chunk size in `src/hsmn.py`.
  - AMD DirectML plugin overhead impacts performance (under investigation).

## Future Plans

- Expand empirical benchmarks on additional datasets.
- Integrate adaptive memory enhancements for dynamic chunk sizing.
- Optimize DirectML plugin for AMD GPUs.
- Complete Inference Session (Windows Executable) for full model usage upon training completion, giving users control over Chain of Thought and other features not offered by large-scale LLM providers.
- Raise enough funding to create a localized gpu training cluster to train all HSMN models on. 

## Research Papers

- `HSMN-2.pdf`
- HighNoon LLM: Revolutionizing Sequence Processing

## Contributing

We welcome contributions to code, documentation, and testing! Please read:

- `docs/CONTRIBUTING.md` for guidelines.
- `docs/CODE_OF_CONDUCT.md` for community standards.

## Support the Project

- **Contribute**: Submit pull requests or report issues on GitHub Issues.
- **Sponsor**: Support development via GitHub Sponsors or Patreon.
- **Share**: Promote HighNoon LLM on social media, forums, or X to increase visibility.

## Contributors

- **Michael Zimmerman**: Founder and CEO of Verso Industries, creator of HighNoon LLM and the HSMN architecture, and Lead Developer for the project.
- **Jacob Godina**: President and Co-Founder of Verso Industries, contributor to code, design, and marketing.

## Contact

- **Email**: `zimmermanmb99@gmail.com`
- **Website**: `www.versoindustries.com` (currently not live)
- **Issues**: [GitHub Issues](https://github.com/versoindustries/HighNoonLLM/issues)
- **Commercial Licensing**: See `COMMERCIAL-LICENSE.md`

## Discord Server

https://discord.gg/pBrSPbaMnM

Thank you for your interest in HighNoon LLM! Stay tuned for model checkpoints and updates.

## Sponsorship

Support the development of HighNoon LLM and its innovative Hierarchical Spatial Neural Memory (HSMN) architecture! Your sponsorship helps advance research, optimize performance, and expand applications in NLP tasks like document translation, summarization, and code generation. Choose a tier below to contribute via Stripe:

### Friend of HighNoon

- **Monthly Amount**: $5
- **Benefits**: Listed as a friend in the README and GitHub Sponsors profile.
- [Sponsor Now](https://buy.stripe.com/dRm9ATfiY47c0v45yugfu0y)

### Research Supporter

- **Monthly Amount**: $10
- **Benefits**: All above benefits; early access to research updates via email or private GitHub repository.
- [Sponsor Now](https://buy.stripe.com/4gM00j5IogTYb9I7GCgfu0z)

### HighNoon Patron

- **Monthly Amount**: $50
- **Benefits**: All above benefits; personal thank-you via email or X post; mention in research acknowledgments (if feasible).
- [Sponsor Now](https://buy.stripe.com/aFa7sL4Ek0V02DcaSOgfu0A)

### Collaborator

- **Monthly Amount**: $100
- **Benefits**: All above benefits; opportunity to suggest features or influence project direction; access to exclusive content like early demos.
- [Sponsor Now](https://buy.stripe.com/cNi14nfiY47cdhQaSOgfu0B)

Your contributions drive the future of efficient and ethical NLP innovation. Thank you for supporting HighNoon LLM!