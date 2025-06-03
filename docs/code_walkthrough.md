HSMN Implementation Guide

This document provides an overview of the Hierarchical Spatial Neural Memory (HSMN) model implementation, as presented in the research paper "HighNoon LLM: Advancing Sequence Processing with Hierarchical Spatial Neural Memory" (included in the repository as hsmn-2.pdf). The HSMN model is designed to efficiently process long sequences in natural language processing (NLP) tasks, addressing the quadratic complexity limitations of traditional transformers by using a hierarchical memory tree. This guide explains the code's architecture, data preprocessing, training procedures, evaluation methods, and usage instructions.

Introduction

The HSMN model, part of the HighNoon LLM framework, enhances sequence processing by dividing input sequences into chunks, constructing a hierarchical memory tree, and generating outputs with reduced computational complexity—$O(n \cdot c)$ where $n$ is the sequence length and $c$ is a fixed chunk size (e.g., 128). This contrasts with the $O(n^2)$ complexity of standard transformers. The implementation supports continual learning via Elastic Weight Consolidation (EWC) and is optimized for both efficiency and scalability. For theoretical details, refer to the research paper.
Model Architecture

The HSMN model comprises several key components, each implemented as a TensorFlow Keras layer or model:

1. StreamingChunker

Purpose: Divides input sequences into overlapping chunks for processing.
Details:
Configurable chunk_size (default: 128) and stride (default: 64).
Ensures a minimum number of chunks (min_chunks) with padding if necessary.
Processes on CPU to reduce GPU memory load.


Code Location: StreamingChunker class.
Paper Reference: Section 3.1, ChunkEncoder description.

2. ChunkEncoder

Purpose: Encodes each chunk into a fixed-dimensional embedding.
Details:
Uses a transformer-based encoder with multiple layers (default: 2) of multi-head self-attention and feed-forward networks.
Adds a [CLS] token to each chunk, whose embedding is extracted as output.
Complexity per chunk is $O(c^2)$, total $O(n \cdot c)$.


Code Location: ChunkEncoder class.
Paper Reference: Section 3.1.

3. Aggregator

Purpose: Builds a hierarchical memory tree from chunk embeddings.
Details:
Constructs a binary tree by iteratively aggregating pairs of child embeddings using a low-rank dense layer (LowRankDense) with pruning.
Employs TensorFlow Model Optimization Toolkit (TF-MOT) for sparsity (50% target).
Output is a set of memory nodes, with the root representing the entire sequence.


Code Location: Aggregator class.
Paper Reference: Section 3.2.

4. ReasoningModule

Purpose: Generates output sequences autoregressively using the memory tree.
Details:
Uses a transformer decoder with self-attention and cross-attention to memory nodes.
Supports chunked processing for long targets, caching past key-value pairs.
Final layer is pruned for efficiency.


Code Location: ReasoningModule class.
Paper Reference: Section 3.3.

5. HSMN

Purpose: Integrates all components into a cohesive model.
Details:
Manages the flow: input → chunking → encoding → aggregation → reasoning.
Includes compressor and decompressor layers (pruned) for dimensionality reduction and restoration.
Optimized for GPU/CPU hybrid execution to manage memory.


Code Location: HSMN class.
Paper Reference: Section 3.

Additional Features:

Pruning is applied using TF-MOT to reduce parameters (see pruning_params).
Gradient recomputation (tf.recompute_grad) enhances memory efficiency during training.

Data Preprocessing

The code includes preprocessing functions for numerous datasets, stored in the PREPROCESSORS dictionary. Each function transforms raw data into context-target pairs:

Examples:
preprocess_sciq: Combines support text and questions as context, with answers as targets.
preprocess_gsm8k: Uses math questions as context and final answers as targets.
preprocess_code_search_net: Pairs docstrings (context) with code snippets (targets).


Output Format: Dictionaries with keys 'context', 'target', and 'task' ('chat' or 'code').
Usage: Applied lazily via generators to minimize memory usage.

Supported Datasets: Includes sciq, gsm8k, mmlu, human_eval, open_assistant, and more (see benchmark_datasets).
Training Process

Training is implemented in two main functions:

1. train_on_dataset

Purpose: Trains the model on a single dataset.
Details:
Uses gradient accumulation (accum_steps) to simulate larger batches.
Employs EWC to prevent catastrophic forgetting when previous_fishers and previous_optimal_weights are provided.
Saves best weights based on validation loss with early stopping.


Key Features:
Lazy data loading via tf.data.Dataset with prefetching.
Fisher Information Matrix computation for EWC (compute_fisher).
Extensive logging and gradient debugging.



2. train_cumulative

Purpose: Trains the model sequentially across multiple datasets.
Details:
Maintains a single model instance, updating it with each dataset.
Uses EWC to retain knowledge from prior tasks.
Evaluates performance on benchmark datasets post-training.


Output: Final model and a scorecard comparing HSMN to top models (e.g., GPT-4, Claude 3.5).

Optimization:

Adam optimizer with a learning rate of 5e-5 and gradient clipping.
Mixed precision (float32) enforced for stability.

Evaluation
Evaluation is handled by evaluate_dataset and train_cumulative:

Metrics: Accuracy for datasets like mmlu, arc, gsm8k; placeholder for others (e.g., human_eval requires custom metrics).
Process:
Generates predictions using model.generate.
Compares predictions to ground truth.


Scorecard: Compares HSMN performance to top models (see top_models_performance).

Benchmarks: Evaluated on datasets like MMLU, GSM8K, HumanEval, etc., with results saved to scorecard.txt.

Notes:

Adjust chunk_size, max_seq_len, and layer counts based on your hardware.
Use the --main-- block to run cumulative training with predefined configs.

Conclusion
The HSMN implementation provides a scalable and efficient solution for long-sequence NLP tasks, leveraging a hierarchical memory structure. This guide has outlined its architecture, preprocessing, training, evaluation, and usage, enabling developers to adapt and extend the code. For a deeper understanding, consult the research paper hsmn-2.pdf.
