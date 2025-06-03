# Contributing to HighNoon LLM

Thank you for your interest in contributing to **HighNoon LLM**, a large language model leveraging Hierarchical Spatial Neural Memory (HSMN) for efficient sequence processing. We welcome contributions from the community to enhance the code, documentation, testing, and more. This guide outlines the process for contributing and ensures a smooth collaboration experience.

## Ways to Contribute
You can contribute in several ways:
- **Code**: Improve the model architecture, optimize performance, or add new features (e.g., in `src/hsmn.py`, `src/chunk_encoder.py`).
- **Documentation**: Enhance `README.md`, `docs/SETUP.md`, or create tutorials in `examples/`.
- **Testing**: Write tests for model components or validate performance on datasets like MMLU or HumanEval.
- **Issues**: Report bugs, suggest features, or propose optimizations via GitHub Issues.
- **Feedback**: Share insights on model performance, usability, or potential improvements.

## Getting Started
1. **Fork the Repository**:
   ```bash
   git clone https://github.com/yourusername/highnoon-llm.git
   cd highnoon-llm
   git remote add upstream https://github.com/yourusername/highnoon-llm.git
   ```
2. **Create a Branch**:
   Name your branch descriptively (e.g., `feature/add-dynamic-chunking`, `fix/log-error`):
   ```bash
   git checkout -b your-branch-name
   ```
3. **Set Up the Environment**:
   Follow [docs/SETUP.md](SETUP.md) to install dependencies and download datasets/tokenizers.

## Contribution Guidelines
### Code Contributions
- **Coding Standards**:
  - Follow [PEP 8](https://pep8.org/) for Python code.
  - Use clear, descriptive variable and function names.
  - Add docstrings for new functions and classes (NumPy or Google style).
- **Testing**:
  - Include tests for new features or bug fixes (place in `tests/` if available).
  - Ensure existing tests pass locally: `python -m unittest discover tests`.
- **Commits**:
  - Write concise, meaningful commit messages (e.g., "Add dynamic chunk size support to StreamingChunker").
  - Group related changes into a single commit where possible.
- **Pull Requests (PRs)**:
  - Submit PRs to the `main` branch.
  - Provide a clear title and description, linking to relevant issues (e.g., "Fixes #123").
  - Explain the purpose, changes, and any testing performed.
  - Ensure the PR passes any automated checks (e.g., linting, tests).

### Documentation Contributions
- Update or create Markdown files in `docs/` or `examples/`.
- Ensure clarity, correct grammar, and consistent formatting.
- Link to relevant sections (e.g., `[README.md](../README.md)`).

### Issue Reporting
- Check [GitHub Issues](https://github.com/yourusername/highnoon-llm/issues) for duplicates before opening a new issue.
- Use the provided templates for bug reports or feature requests.
- Include:
  - A clear title and description.
  - Steps to reproduce (for bugs).
  - Environment details (e.g., Python version, TensorFlow version, OS).
  - Logs from `training_log.log` if applicable.

## Development Workflow
1. **Make Changes**:
   Modify code, tests, or documentation in your branch.
2. **Test Locally**:
   Run the model or scripts to verify changes:
   ```bash
   python batch_train.py --dataset sciq --epochs 1
   ```
3. **Commit Changes**:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```
4. **Sync with Upstream**:
   Ensure your branch is up-to-date:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
   Resolve conflicts if any.
5. **Push to Your Fork**:
   ```bash
   git push origin your-branch-name
   ```
6. **Open a Pull Request**:
   - Go to your fork on GitHub and create a PR.
   - Tag maintainers for review (e.g., @yourusername).
   - Respond to feedback and make requested changes.

## Licensing
- **Code Contributions**: All code contributions are licensed under the [Apache 2.0 License](../LICENSE-APACHE-2.0.txt). By submitting a PR, you agree to license your contributions under this license.
- **Model Weights**: Contributions involving model weights (e.g., training scripts) must respect the [Creative Commons Attribution-NonCommercial 4.0 License](../LICENSE-CC-BY-NC-4.0.txt) for non-commercial use or the [Commercial License](../COMMERCIAL-LICENSE.md) for commercial use.

## Code of Conduct
Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a respectful and inclusive environment for all contributors.

## Review Process
- PRs are reviewed by maintainers within 3-5 days.
- We may request changes to align with project standards.
- Once approved, your PR will be merged into `main`.

## Recognition
Contributors are acknowledged in:
- The `README.md` contributors section.
- Release notes for significant contributions.
- Project documentation or website (if applicable).

## Contact
For questions or assistance:
- **Email**: zimmermanmb99@gmail.com
- **GitHub Issues**: [https://github.com/yourusername/highnoon-llm/issues](https://github.com/yourusername/highnoon-llm/issues)
- **Website**: www.versoindustries.com

Thank you for helping make HighNoon LLM better!