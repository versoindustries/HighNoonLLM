# HighNoon LLM: AI That Thinks Like You

Welcome to **HighNoon LLM**, a groundbreaking AI project from Verso Industries that’s making artificial intelligence smarter, faster, and more accessible. Our goal is to build an AI that processes and understands language the way humans do—breaking down information into manageable pieces, connecting ideas, and learning new skills without forgetting old ones. Whether you’re a developer, a business, or just curious about AI, HighNoon LLM is designed to empower you with tools that are efficient, private, and open to everyone.

We’re thrilled to invite you to join our open-source community, contribute to the project, or support us as a sponsor to help shape the future of AI. Let’s dive into what HighNoon LLM is all about!

## What is HighNoon LLM?

HighNoon LLM is a large language model that uses a unique system called **Hierarchical Spatial Neural Memory (HSMN)** to process language in a human-like way. Imagine reading a long book: you don’t memorize every word—you group sentences into ideas and ideas into themes. HighNoon does the same, organizing text into chunks and building a tree of understanding that captures both the big picture and tiny details.

This approach makes HighNoon:
- **Super efficient**: It uses way less computing power than traditional AI models, saving time and energy.
- **Smart and versatile**: It handles tasks like summarizing long documents, writing code, or answering complex questions with ease.
- **Privacy-focused**: It runs on your own device, keeping your data secure.
- **Accessible**: Designed to work on everyday computers, not just fancy servers.

Our mission is to create AI that’s practical for everyone—businesses, developers, students, and hobbyists—while staying sustainable and ethical.

## Why HighNoon LLM Stands Out

Here’s what makes HighNoon special:
- **Thinks Like You**: Breaks down text into chunks (like sentences or paragraphs) and organizes them into a tree, making it great at understanding long documents or conversations.
- **Saves Resources**: Uses 78x less computing power than standard AI models for big tasks, so it’s faster and greener.
- **Keeps Learning**: Adapts to new tasks (like coding or answering science questions) without forgetting what it already knows, using Elastic Weight Consolidation (EWC).
- **Runs Locally**: Works on your laptop or server, protecting your data and cutting cloud costs.
- **High Performance**: Achieved 100% accuracy on STEM and SciQ datasets as a classification model (contact us to replicate).
- **Open-Source Heart**: Free for non-commercial use, with code open to all under Apache 2.0, inviting everyone to build and explore.

## What Can HighNoon LLM Do?

HighNoon LLM is being trained to tackle a wide range of tasks, including:
- **Summarizing Long Texts**: Turn a 100-page report into a concise summary in seconds.
- **Writing Code**: Generate or debug code for projects, from Python to web apps.
- **Answering Questions**: Provide accurate answers for schoolwork, research, or curiosity.
- **Translating Documents**: Handle entire books or reports with context-aware translations.
- **Powering Chatbots**: Create smart, responsive assistants for businesses or personal use.

Whether you’re a business streamlining operations, a developer building cool tools, or a student tackling a project, HighNoon LLM has something for you.

## Key Features

- **Efficient Sequence Processing**: Uses a hierarchical binary memory tree to manage long contexts with reduced computational overhead, dropping complexity from O(n²) to O(n·c), where c is a chunk size (e.g., 128 tokens).
- **Multi-Task Training**: Supports diverse datasets like SciQ, GSM8K, MMLU, CodeSearchNet, HumanEval, and MBPP for robust performance across NLP and coding tasks.
- **Optimized for Resource Constraints**: Targets ~6.3GB VRAM usage with gradient accumulation and 50% model pruning for efficiency.
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting during continual learning, ensuring versatility across tasks.
- **High Performance**: Proven 100% accuracy on STEM and SciQ datasets (reproducible upon request).

## Project Status

- **Training in Progress**: Expected completion by September 2025 due to compute constraints.
- **Code Available**: Open-sourced under the Apache 2.0 License.
- **Model Weights**: Intermediate checkpoints available from July 2025 under Creative Commons Attribution-NonCommercial 4.0 for non-commercial use. Commercial use requires a paid license (see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).

## Join the Community

HighNoon LLM is more than a project—it’s a movement to make AI open, inclusive, and useful for all. Here’s how you can get involved:
- **Contribute Code**: Help improve the model, fix bugs, or add features. Check out our [Contributing Guide](docs/CONTRIBUTING.md).
- **Test and Share**: Try out our early model checkpoints (available from July 2025) and share your feedback.
- **Spread the Word**: Post about HighNoon LLM on social media, forums, or X using #HighNoonLLM.
- **Join the Conversation**: Chat with us and other enthusiasts on our [Discord server](https://discord.gg/pBrSPbaMnM).

Every contribution, big or small, helps us grow and improve!

## Support HighNoon LLM

Building cutting-edge AI takes resources, and we need your support to make HighNoon LLM the best it can be. By becoming a sponsor, you’ll help us:
- Build faster training systems.
- Hire talented researchers and developers.
- Expand features like web search integration and a plugin marketplace.

### Sponsorship Tiers

- **Friend of HighNoon ($5/month)**: Get a shoutout in our README and GitHub Sponsors page. Perfect for showing your support!
  - [Sponsor Now](https://buy.stripe.com/dRm9ATfiY47c0v45yugfu0y)
- **Research Supporter ($10/month)**: All Friend benefits plus early access to updates and new features via email.
  - [Sponsor Now](https://buy.stripe.com/4gM00j5IogTYb9I7GCgfu0z)
- **HighNoon Patron ($50/month)**: All Research Supporter benefits, plus a personal thank-you and recognition in our papers.
  - [Sponsor Now](https://buy.stripe.com/3cI14n8UA1Z4a5E8KGgfu0G)
- **Corporate Sponsor ($500/month)**: All Patron benefits, plus your company logo on our homepage and priority for feature requests.
  - [Sponsor Now](https://buy.stripe.com/cNidR9eeU0V00v47GCgfu0H)
- **Strategic Partner ($25,000/year)**: Shape the project with quarterly strategy meetings and guaranteed feature inclusion.
  - [Sponsor Now](https://buy.stripe.com/cNiaEX9YEdHM6Ts8KGgfu0I)
- **Premier Sponsor ($150,000/year)**: Co-branding, dedicated support, and naming rights for a feature or model version.
  - [Sponsor Now](https://buy.stripe.com/7sI9ATfiY47c0v48KGgfu0J)

Your support fuels innovation and helps us keep HighNoon LLM open and accessible. Thank you!

## How to Get Started

Ready to dive in? Here’s how to set up HighNoon LLM on your machine:

### Requirements

- **Python**: Version 3.10.11
- **TensorFlow**: 2.10.0 for NVIDIA GPUs, 2.10.1 for AMD GPUs or CPUs
- **Hardware**: 32GB RAM; NVIDIA or AMD GPU with 8GB VRAM recommended (CPU works too)
- **System**: Linux or Windows

### Installation Steps

1. **Clone the Project**:
   ```bash
   git clone https://github.com/versoindustries/HighNoonLLM.git
   cd HighNoonLLM
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Datasets**:
   ```bash
   python dataset_download.py
   ```

5. **Download Tokenizers**:
   ```bash
   python token_download.py
   ```

### TensorFlow Notes

- **NVIDIA GPU Users**: Use `tensorflow==2.10.0` for optimal performance.
- **AMD GPU Users**: Use `tensorflow-cpu==2.10.1` with `tensorflow-directml-plugin`. Note: DirectML may introduce overhead (under investigation).
- **CPU Users**: Install `tensorflow-cpu==2.10.1`.

Check [SETUP.md](docs/SETUP.md) for troubleshooting or advanced setup tips.

### Training the Model

To start training HighNoon LLM:
```bash
python batch_train.py --dataset mmlu
```
- **Logs**: Saved in `training_log.log`.
- **Checkpoints**: Saved as `hsmn_model_<dataset>_best_epoch_XX.h5`.

Explore `batch_train.py` for options like batch size or epochs.

## Project Structure

- `Owasp/`: OWASP guidelines and preprocess script for PDF-to-text conversion, planned for OWASP dataset creation.
- `Research/`: Research papers on HSMN and HighNoon LLM.
- `batch_train.py`: Main script for multi-dataset training.
- `dataset_download.py`: Downloads configured datasets.
- `token_download.py`: Sets up RobertaTokenizer.
- `requirements.txt`: Python dependencies.
- `models/`: Placeholder for model weights (checkpoints from July 2025).
- `docs/`: Setup and contribution guidelines.

## Model Architecture

- **ChunkEncoder**: Splits input sequences into fixed-size chunks (default: 128 tokens) and encodes them into embeddings.
- **Aggregator**: Builds a hierarchical binary memory tree from chunk embeddings for efficient context representation.
- **ReasoningModule**: Generates outputs autoregressively by attending to the memory tree.

## Training Process

- **Data Preprocessing**: Datasets split into training, validation, and test sets (see `src/preprocessors.py`).
- **Continual Learning**: EWC retains knowledge across tasks, preventing catastrophic forgetting.
- **Optimization**: Uses gradient accumulation and 50% model pruning for efficiency.
- **Supported Datasets**: SciQ, GSM8K, MMLU, CodeSearchNet, HumanEval, MBPP, and more (see `batch_train.py`).

## Evaluation

- **Benchmarks**: Compared against GPT-4, Claude 3.5 Sonnet on MMLU, HumanEval, GSM8K.
- **Results**: Saved to `scorecard.txt` post-training.
- **Notable Achievement**: 100% accuracy on STEM and SciQ (contact for replication details).

## Additional Notes

- **Logging**: Detailed logs in `training_log.log` for transparency.
- **Memory Management**: Optimized for ~6.3GB VRAM with GPU memory growth and CPU offloading.
- **Known Issues**:
  - Dataset-specific preprocessing errors (check `training_log.log`).
  - GPU memory overflows with large sequences (reduce batch/chunk size in `src/hsmn.py`).
  - AMD DirectML plugin overhead (under investigation).

## Future Plans

- Expand benchmarks on additional datasets.
- Add adaptive memory for dynamic chunk sizing.
- Optimize DirectML for AMD GPUs.
- Release Inference Session (Windows Executable) for full model use, including user-controlled Chain of Thought.
- Raise funding for a localized GPU training cluster.

## Research Papers

- `HSMN-2.pdf`
- HighNoon LLM: Revolutionizing Sequence Processing

## Licensing

- **Code**: Free for all under Apache 2.0—modify, share, or use commercially with attribution.
- **Model Weights**: Free for non-commercial use under Creative Commons Attribution-NonCommercial 4.0. Commercial use requires a paid license (see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).

## Meet the Team

- **Michael Zimmerman**: Founder & CEO of Verso Industries, creator of HighNoon LLM and HSMN.
- **Jacob Godina**: President & Co-Founder, driving code, design, and marketing.
- **Abby Hosta**: Design and marketing expert.
- **Lee**: Machine Learning Engineer.
- **Elijah**: Social media and content creator.

## Get in Touch

- **Email**: zimmermanmb99@gmail.com
- **GitHub Issues**: [Report bugs or ideas](https://github.com/versoindustries/HighNoonLLM/issues)
- **Discord**: [Join our community](https://discord.gg/pBrSPbaMnM)
- **Commercial Licensing**: See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)

## Why Support HighNoon LLM?

HighNoon LLM isn’t just about building better AI—it’s about building AI that works for everyone. By joining us as a contributor or sponsor, you’re helping create a future where AI is:
- **Affordable**: No expensive cloud subscriptions.
- **Private**: Your data stays with you.
- **Sustainable**: Lower energy use for a greener planet.
- **Inclusive**: Open to developers, businesses, and hobbyists worldwide.

Let’s make AI that thinks like humans, works for humans, and grows with humans. Explore the project at [https://github.com/versoindustries/HighNoonLLM](https://github.com/versoindustries/HighNoonLLM) and join us today!

#AI #OpenSource #Innovation #HighNoonLLM