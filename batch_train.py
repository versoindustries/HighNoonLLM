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

import tensorflow as tf
import numpy as np
import time
import os
import sys
import io
import logging
from tensorflow.keras.mixed_precision import set_global_policy
from transformers import RobertaTokenizer
from datasets import load_from_disk, load_dataset
import faulthandler
import psutil
import gc
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, ConstantSparsity, UpdatePruningStep
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
import tensorflow_model_optimization as tfmot
from tensorflow.compat.v1 import recompute_grad
import traceback
import random
import numpy
from typing import Dict, Generator, List, Optional, Tuple, Any
import psutil
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import Counter

# Define pruning parameters (consistent with your original code)
pruning_params = {
    'pruning_schedule': ConstantSparsity(target_sparsity=0.5, begin_step=0, frequency=100)
}

# Enable fault handler to catch segmentation faults
faulthandler.enable()

# Clear any existing handlers to avoid duplicates
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel(logging.INFO)

# Define the log format
log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# File handler with UTF-8 encoding
file_handler = logging.FileHandler('training_log.log', mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler with UTF-8 encoding and error replacement
console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
console_handler = logging.StreamHandler(stream=console_stream)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configure TensorFlow's logger (optional, but good practice if you want to control TF's own logs)
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.INFO) # You can set this to logging.DEBUG, logging.WARN, etc.

# Set memory growth for GPUs before any other operations
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
logging.info(f"Available GPUs: {physical_devices}")

# Enforce float32 precision
set_global_policy('float32')

# Load CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
VOCAB_SIZE = tokenizer.vocab_size

MAX_CHUNK_SIZE = 128

def get_pruned_layers(model):
    """
    Recursively find all pruned layers in the model using model.submodules.
    
    Args:
        model: The HSMN model instance.
    
    Returns:
        List of pruned layers.
    """
    pruned_layers_set = set()
    for submodule in model.submodules:
        if hasattr(submodule, 'pruning_step') and isinstance(submodule.pruning_step, tf.Variable):
            pruned_layers_set.add(submodule)
    return list(pruned_layers_set)

# Helper function for ChunkEncoder transformer block
def apply_transformer_block(x, block, training):
    attn_output = block['mha'](query=x, value=x, key=x, training=training)
    x = block['norm1'](x + block['dropout1'](attn_output, training=training))
    ff_output = block['dense_out'](block['dense_ff'](x))
    x = block['norm2'](x + block['dropout2'](ff_output, training=training))
    return x

# Helper function for ReasoningModule transformer block
def apply_reasoning_block(x, memory_embeddings, block, training, causal_mask, cross_attention_mask):
    """
    Apply a single transformer block in the ReasoningModule, with shape assertions and proper mask handling.

    Args:
        x: Input tensor of shape (batch_size, seq_len, embedding_dim).
        memory_embeddings: Memory embeddings for cross-attention, shape (batch_size, num_memory_nodes, embedding_dim).
        block: Dictionary containing layers for the transformer block (mha_self, mha_cross, dropout1, norm1, etc.).
        training: Boolean indicating training mode.
        causal_mask: Combined causal and padding mask for self-attention, shape (batch_size, seq_len, seq_len).
        cross_attention_mask: Mask for cross-attention, shape (batch_size, seq_len, num_memory_nodes).

    Returns:
        Transformed tensor after applying the block, shape (batch_size, seq_len, embedding_dim).
    """
    logging.info(f"Input x shape={x.shape}")
    tf.debugging.assert_shapes([(x, (None, None, 512))], message="Input x has unexpected shape")

    # Apply self-attention with combined causal and padding mask
    attn_output_self = block['mha_self'](
        query=x,
        value=x,
        key=x,
        attention_mask=causal_mask,
        training=training
    )
    logging.info(f"attn_output_self shape={attn_output_self.shape}")
    tf.debugging.assert_shapes([(attn_output_self, (None, None, 512))], message="Self-attention output shape mismatch")

    x_plus_attn = x + block['dropout1'](attn_output_self, training=training)
    x = block['norm1'](x_plus_attn)
    logging.info(f"After norm1 x shape={x.shape}")

    # Apply cross-attention if memory_embeddings are non-empty
    if tf.shape(memory_embeddings)[1] > 0:
        attn_output_cross = block['mha_cross'](
            query=x,
            value=memory_embeddings,
            key=memory_embeddings,
            attention_mask=cross_attention_mask,
            training=training
        )
        logging.info(f"attn_output_cross shape={attn_output_cross.shape}")
        tf.debugging.assert_shapes([(attn_output_cross, (None, None, 512))], message="Cross-attention output shape mismatch")
        x_plus_cross = x + block['dropout2'](attn_output_cross, training=training)
    else:
        logging.info("Skipping cross-attention as memory_embeddings sequence length is 0")
        x_plus_cross = x

    x = block['norm2'](x_plus_cross)
    logging.info(f"After norm2 x shape={x.shape}")

    # Feed-forward network
    ff_intermediate = block['dense_ff'](x)
    logging.info(f"ff_intermediate shape={ff_intermediate.shape}")
    tf.debugging.assert_shapes([(ff_intermediate, (None, None, 2048))], message="FF intermediate shape mismatch")

    ff_output = block['dense_out'](ff_intermediate)
    logging.info(f"ff_output shape={ff_output.shape}")
    tf.debugging.assert_shapes([(ff_output, (None, None, 512))], message="FF output shape mismatch")

    x_plus_ff = x + block['dropout3'](ff_output, training=training)
    x = block['norm3'](x_plus_ff)
    logging.info(f"After norm3 x shape={x.shape}")

    return x

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            value_dim=embedding_dim // num_heads,
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense_ff = tf.keras.layers.Dense(
            ff_dim,
            activation='relu',
            bias_initializer=tf.keras.initializers.Constant(0.01)
        )
        self.dense_out = tf.keras.layers.Dense(
            embedding_dim,
            bias_initializer=tf.keras.initializers.Constant(0.01)
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, attention_mask=None, training=False):
        @tf.recompute_grad
        def forward_fn(x, attention_mask):
            attn_output = self.mha(
                query=x,
                value=x,
                key=x,
                attention_mask=attention_mask,
                training=training  # Captured from outer scope
            )
            x = self.norm1(x + self.dropout1(attn_output, training=training))
            ff_output = self.dense_out(self.dense_ff(x))
            x = self.norm2(x + self.dropout2(ff_output, training=training))
            return x

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32)  # Ensure float32 for consistency
        return forward_fn(x, attention_mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.mha.num_heads,
            'ff_dim': self.dense_ff.units,
            'dropout': self.dropout1.rate
        })
        return config
    
# StreamingChunker with dynamic chunk_size
class StreamingChunker(tf.keras.layers.Layer):
    def __init__(self, chunk_size=128, stride=64, min_chunks=2, tokenizer=None, process_on_cpu=False, **kwargs):
        super(StreamingChunker, self).__init__(**kwargs)
        self.chunk_size = chunk_size
        self.stride = stride
        self.min_chunks = min_chunks
        self.tokenizer = tokenizer
        self.process_on_cpu = process_on_cpu

    @tf.function(reduce_retracing=True)
    def call(self, inputs, is_target=False):
        """
        Processes pre-tokenized input token IDs into chunks, ensuring at least min_chunks.
        
        Args:
            inputs: Tensor of shape (batch_size, seq_length) with token IDs.
            is_target: Boolean indicating if this is a target sequence (affects output shape).
        
        Returns:
            If is_target: List of tensors, each of shape (batch_size, chunk_size).
            Else: Tensor of shape (num_chunks, batch_size, chunk_size).
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        tf.debugging.assert_greater(self.chunk_size, 0, "Chunk size must be positive")
        tf.debugging.assert_greater_equal(self.stride, 0, "Stride must be non-negative")
        
        logging.info(f"StreamingChunker: Processing {seq_length} tokens, chunk_size={self.chunk_size}, stride={self.stride}, is_target={is_target}")
        
        device = '/CPU:0' if self.process_on_cpu else '/GPU:0'
        with tf.device(device):
            # Ensure minimum length for at least min_chunks
            min_length_needed = self.chunk_size + (self.min_chunks - 1) * self.stride  # e.g., 128 + (2-1)*64 = 192
            padding_needed = tf.maximum(0, min_length_needed - seq_length)
            padded_inputs = tf.pad(inputs, [[0, 0], [0, padding_needed]], constant_values=self.tokenizer.pad_token_id)
            effective_seq_length = seq_length + padding_needed
            
            # Calculate number of chunks
            num_chunks_float = (tf.cast(effective_seq_length - self.chunk_size, tf.float32) / self.stride) + 1
            num_chunks = tf.cast(tf.math.ceil(num_chunks_float), tf.int32)
            num_chunks = tf.maximum(num_chunks, self.min_chunks)  # Enforce min_chunks
            
            logging.info(f"StreamingChunker: seq_length={seq_length}, effective_seq_length={effective_seq_length}, num_chunks={num_chunks}")
            
            def gather_chunks(i):
                start = i * self.stride
                end = tf.minimum(start + self.chunk_size, effective_seq_length)
                chunk_length = end - start
                begin = tf.stack([0, start])
                size = tf.stack([batch_size, chunk_length])
                chunk = tf.slice(padded_inputs, begin, size)
                padding_length = self.chunk_size - chunk_length
                padding_shape = tf.stack([batch_size, padding_length])
                padding = tf.fill(padding_shape, self.tokenizer.pad_token_id)
                chunk = tf.concat([chunk, padding], axis=1)
                return chunk
            
            chunks = tf.map_fn(
                gather_chunks,
                tf.range(num_chunks),
                fn_output_signature=tf.TensorSpec([None, self.chunk_size], dtype=tf.int32)
            )
            
            if is_target:
                chunk_list = [chunks[i] for i in range(num_chunks)]
                logging.info(f"StreamingChunker: Returned {len(chunk_list)} target chunks")
                return chunk_list
            else:
                logging.info(f"StreamingChunker: Generated {tf.shape(chunks)[0]} context chunks")
                tf.debugging.assert_rank(chunks, 3, message="Chunks must have three dimensions")
                return chunks

    def get_config(self):
        config = super(StreamingChunker, self).get_config()
        config.update({
            'chunk_size': self.chunk_size,
            'stride': self.stride,
            'min_chunks': self.min_chunks,
            'tokenizer': None,
            'process_on_cpu': self.process_on_cpu
        })
        return config

# ChunkEncoder with transformer layers
class ChunkEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=512, num_layers=2, num_heads=8, ff_dim=2048, dropout=0.1, max_seq_len=512, pad_token_id=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = tf.keras.layers.Embedding(max_seq_len + 1, embedding_dim)
        self.pad_token_id = pad_token_id
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.transformer_blocks = [
            TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        logging.info(f"Initialized ChunkEncoder with {len(self.transformer_blocks)} transformer blocks")

    def call(self, inputs, training=False):
        tf.debugging.assert_type(inputs, tf.int32, message="ChunkEncoder: Inputs must be int32 token IDs")
        # Expect inputs with shape [batch_size, num_chunks, chunk_size]
        batch_size = tf.shape(inputs)[0]
        num_chunks = tf.shape(inputs)[1]
        seq_len = tf.shape(inputs)[2]
        tf.print("ChunkEncoder.call: Input shape=", [batch_size, num_chunks, seq_len])

        # Create attention mask for padded tokens
        mask = tf.not_equal(inputs, self.pad_token_id)  # [batch_size, num_chunks, seq_len]
        cls_token = tf.zeros([batch_size, num_chunks, 1], dtype=tf.int32)  # CLS token
        inputs_with_cls = tf.concat([cls_token, inputs], axis=2)  # [batch_size, num_chunks, seq_len + 1]
        extended_mask = tf.concat([tf.ones([batch_size, num_chunks, 1], dtype=tf.bool), mask], axis=2)  # [batch_size, num_chunks, seq_len + 1]
        attention_mask = tf.expand_dims(extended_mask, axis=2) & tf.expand_dims(extended_mask, axis=3)  # [batch_size, num_chunks, seq_len + 1, seq_len + 1]

        # Positional embeddings
        positions = tf.range(start=0, limit=seq_len + 1, delta=1)
        positions = tf.expand_dims(positions, axis=0)  # [1, seq_len + 1]
        positions = tf.expand_dims(positions, axis=0)  # [1, 1, seq_len + 1]
        positions = tf.tile(positions, [batch_size, num_chunks, 1])  # [batch_size, num_chunks, seq_len + 1]

        # Embed inputs and add positional embeddings
        embedded = self.embedding(inputs_with_cls)  # [batch_size, num_chunks, seq_len + 1, embedding_dim]
        pos_embed = self.positional_embedding(positions)  # [batch_size, num_chunks, seq_len + 1, embedding_dim]
        x = embedded + pos_embed
        tf.debugging.assert_shapes([(x, (None, None, None, self.embedding_dim))], message="Embedded input shape mismatch")

        # Store the sequence length after CLS token addition
        output_seq_len = seq_len + 1  # e.g., 128 + 1 = 129

        # Reshape to [batch_size * num_chunks, output_seq_len, embedding_dim] once
        x_flat = tf.reshape(x, [batch_size * num_chunks, output_seq_len, self.embedding_dim])
        mask_flat = tf.reshape(attention_mask, [batch_size * num_chunks, output_seq_len, output_seq_len])

        # Apply all transformer blocks on x_flat
        if not self.transformer_blocks:
            logging.error("ChunkEncoder: transformer_blocks is empty")
            raise ValueError("No transformer blocks available")
        for idx, block in enumerate(self.transformer_blocks):
            logging.info(f"Applying transformer block {idx + 1}/{len(self.transformer_blocks)}")
            x_flat = block(x_flat, attention_mask=mask_flat, training=training)  # Uses checkpointed_call internally during training

        # Extract CLS embedding from x_flat after all blocks
        cls_embedding_flat = x_flat[:, 0, :]  # [batch_size * num_chunks, embedding_dim]
        cls_embedding = tf.reshape(cls_embedding_flat, [batch_size, num_chunks, self.embedding_dim])  # [batch_size, num_chunks, embedding_dim]
        tf.debugging.assert_shapes([(cls_embedding, (None, None, self.embedding_dim))], message="CLS embedding shape mismatch")
        tf.print("ChunkEncoder.call: Output shape=", cls_embedding.shape)
        return cls_embedding
    
class LowRankDense(tf.keras.layers.Layer, prunable_layer.PrunableLayer):
    def __init__(self, input_dim, output_dim, rank, activation=None, use_bias=True, **kwargs):
        super(LowRankDense, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.use_bias = use_bias
        self.activation_fn = tf.keras.activations.get(activation)

        self.U = self.add_weight(
            shape=(input_dim, rank),
            initializer='glorot_uniform',  # Consider 'he_uniform' for ReLU, but keep for now
            trainable=True,
            name='U'
        )
        self.V = self.add_weight(
            shape=(rank, output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='V'
        )
        if self.use_bias:
            if isinstance(self.activation_fn, tf.keras.layers.ReLU) or \
               (hasattr(self.activation_fn, '__name__') and 'relu' in self.activation_fn.__name__.lower()) or \
               isinstance(self.activation_fn, tf.keras.layers.LeakyReLU):
                bias_init_val = 0.01
            else:
                bias_init_val = 0.0
            self.bias_weight = self.add_weight(
                shape=(output_dim,),
                initializer=tf.keras.initializers.Constant(bias_init_val),
                trainable=True,
                name='bias_weight'
            )
        else:
            self.bias_weight = None

    def call(self, inputs, training=False):
        product_UV = tf.matmul(self.U, self.V)
        x = tf.matmul(inputs, product_UV)
        if self.use_bias and self.bias_weight is not None:
            x = x + self.bias_weight
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x

    def get_prunable_weights(self):
        weights = [self.U, self.V]
        if self.use_bias and self.bias_weight is not None:
            weights.append(self.bias_weight)
        return weights

    def get_config(self):
        config = super(LowRankDense, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'rank': self.rank,
            'activation': tf.keras.activations.serialize(self.activation_fn),
            'use_bias': self.use_bias
        })
        return config

class Aggregator(tf.keras.Model):
    def __init__(self, compressed_dim=128, rank=64):
        super(Aggregator, self).__init__()
        self.dense1 = tfmot.sparsity.keras.prune_low_magnitude(
            LowRankDense(
                input_dim=256,
                output_dim=256,
                rank=rank,
                activation='relu'
            ),
            **pruning_params
        )
        self.dense2 = tfmot.sparsity.keras.prune_low_magnitude(
            LowRankDense(
                input_dim=256,
                output_dim=compressed_dim,
                rank=rank
            ),
            **pruning_params
        )

    @tf.function
    def call(self, child1_embedding, child2_embedding):
        concatenated = tf.concat([child1_embedding, child2_embedding], axis=-1)
        x = self.dense1(concatenated)
        output_embedding = self.dense2(x)
        return output_embedding

class ReasoningTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"

        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.output_dense = tf.keras.layers.Dense(embedding_dim)

        self.mha_cross = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense_ff = tf.keras.layers.Dense(
            ff_dim,
            activation='relu',
            bias_initializer=tf.keras.initializers.Constant(0.01)
        )
        self.dense_out = tf.keras.layers.Dense(
            embedding_dim,
            bias_initializer=tf.keras.initializers.Constant(0.01)
        )
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.supports_masking = True

    def call(self, x, memory_embeddings, causal_mask, cross_attention_mask, past_key=None, past_value=None, training=False):
        logging.debug(f"call: x shape={x.shape}, memory_embeddings shape={memory_embeddings.shape}, "
                      f"causal_mask shape={causal_mask.shape}, cross_attention_mask shape={cross_attention_mask.shape}, "
                      f"past_key shape={'None' if past_key is None else past_key.shape}, "
                      f"past_value shape={'None' if past_value is None else past_value.shape}, training={training}")

        # Ensure inputs are tensors
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        memory_embeddings = tf.convert_to_tensor(memory_embeddings, dtype=tf.float32)
        causal_mask = tf.convert_to_tensor(causal_mask, dtype=tf.float32)
        cross_attention_mask = tf.convert_to_tensor(cross_attention_mask, dtype=tf.float32)
        if past_key is not None:
            past_key = tf.convert_to_tensor(past_key, dtype=tf.float32)
        if past_value is not None:
            past_value = tf.convert_to_tensor(past_value, dtype=tf.float32)

        # Define the forward pass as a nested function
        @tf.recompute_grad
        def forward_fn(x, memory_embeddings, causal_mask, cross_attention_mask, past_key, past_value):
            batch_size = tf.shape(x)[0]
            current_seq_len = tf.shape(x)[1]
            past_length = tf.shape(past_key)[1] if past_key is not None else 0
            total_length = past_length + current_seq_len

            query = self.query_dense(x)
            key_current = self.key_dense(x)
            value_current = self.value_dense(x)

            if past_key is not None:
                key = tf.concat([past_key, key_current], axis=1)
                value = tf.concat([past_value, value_current], axis=1)
            else:
                key = key_current
                value = value_current

            query = tf.reshape(query, [batch_size, current_seq_len, self.num_heads, self.head_dim])
            query = tf.transpose(query, [0, 2, 1, 3])
            key = tf.reshape(key, [batch_size, total_length, self.num_heads, self.head_dim])
            key = tf.transpose(key, [0, 2, 1, 3])
            value = tf.reshape(value, [batch_size, total_length, self.num_heads, self.head_dim])
            value = tf.transpose(value, [0, 2, 1, 3])

            scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
            i = tf.range(current_seq_len)
            j = tf.range(total_length)
            mask = tf.cast(j[None, :] <= (past_length + i)[:, None], tf.float32)
            mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
            scores = scores + (1.0 - mask) * -1e9

            attn_weights = tf.nn.softmax(scores, axis=-1)
            attn_output = tf.matmul(attn_weights, value)
            attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
            attn_output = tf.reshape(attn_output, [batch_size, current_seq_len, self.embedding_dim])
            attn_output = self.output_dense(attn_output)

            x_plus_attn = x + self.dropout1(attn_output, training=training)
            x = self.norm1(x_plus_attn)

            if tf.shape(memory_embeddings)[1] > 0:
                attn_output_cross = self.mha_cross(
                    query=x,
                    value=memory_embeddings,
                    key=memory_embeddings,
                    attention_mask=cross_attention_mask,
                    training=training
                )
                x_plus_cross = x + self.dropout2(attn_output_cross, training=training)
            else:
                x_plus_cross = x
            x = self.norm2(x_plus_cross)

            ff_output = self.dense_out(self.dense_ff(x))
            x_out = self.norm3(x + self.dropout3(ff_output, training=training))

            logging.debug(f"forward_fn outputs: x_out shape={x_out.shape}, key_current shape={key_current.shape}, value_current shape={value_current.shape}")
            return x_out, key_current, value_current

        # Handle batch dimension adjustments
        if len(x.shape) == 2:
            x = tf.expand_dims(x, 0)
        if len(memory_embeddings.shape) == 2:
            memory_embeddings = tf.expand_dims(memory_embeddings, 0)
        if len(causal_mask.shape) == 2:
            causal_mask = tf.expand_dims(causal_mask, 0)
        if len(cross_attention_mask.shape) == 2:
            cross_attention_mask = tf.expand_dims(cross_attention_mask, 0)
        if past_key is not None and len(past_key.shape) == 2:
            past_key = tf.expand_dims(past_key, 0)
        if past_value is not None and len(past_value.shape) == 2:
            past_value = tf.expand_dims(past_value, 0)

        batch_size = tf.shape(x)[0]
        embedding_dim = self.embedding_dim
        if past_key is None:
            past_key = tf.zeros([batch_size, 0, embedding_dim], dtype=tf.float32)
        if past_value is None:
            past_value = tf.zeros([batch_size, 0, embedding_dim], dtype=tf.float32)

        # Call the decorated function
        outputs = forward_fn(x, memory_embeddings, causal_mask, cross_attention_mask, past_key, past_value)

        # Shape assertions
        tf.debugging.assert_shapes([
            (outputs[0], (None, None, self.embedding_dim)),
            (outputs[1], (None, None, self.embedding_dim)),
            (outputs[2], (None, None, self.embedding_dim)),
        ], message="Output shapes mismatch in ReasoningTransformerBlock")

        return outputs[0], outputs[1], outputs[2]

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.dense_ff.units,
            'dropout': self.dropout1.rate
        })
        return config

class ReasoningModule(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=512, num_layers=7, num_heads=8, ff_dim=2048, dropout=0.3, max_seq_len=512, chunk_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = tf.keras.layers.Embedding(max_seq_len, embedding_dim)
        self.transformer_blocks = [
            ReasoningTransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.final_dense = tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(vocab_size, bias_initializer=tf.keras.initializers.Constant(0.01)),
            **pruning_params
        )

    def call(self, target_tokens, memory_embeddings, cross_attention_mask, training=False):
        batch_size = tf.shape(target_tokens)[0]
        seq_len = tf.shape(target_tokens)[1]

        if seq_len <= self.chunk_size:
            positions = tf.range(0, seq_len, dtype=tf.int32)
            positions = tf.expand_dims(positions, axis=0)
            positions = tf.tile(positions, [batch_size, 1])
            embedded = self.embedding(target_tokens)
            pos_embed = self.positional_embedding(positions)
            x = embedded + pos_embed
            causal_mask = tf.linalg.band_part(tf.ones([seq_len, seq_len], dtype=tf.float32), -1, 0)
            causal_mask = tf.expand_dims(causal_mask, axis=0)
            for block in self.transformer_blocks:
                x, _, _ = block(x, memory_embeddings, causal_mask, cross_attention_mask, training=training)
            logits = self.final_dense(x)
            return logits
        else:
            with tf.device('/CPU:0'):
                num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
                logits_list = []
                past_key_values = [None] * len(self.transformer_blocks)
                for i in range(num_chunks):
                    start = i * self.chunk_size
                    end = tf.minimum(start + self.chunk_size, seq_len)
                    current_tokens = target_tokens[:, start:end]
                    current_seq_len = end - start
                    positions = tf.range(0, current_seq_len, dtype=tf.int32)
                    positions = tf.expand_dims(positions, axis=0)
                    positions = tf.tile(positions, [batch_size, 1])
                    embedded = self.embedding(current_tokens)
                    pos_embed = self.positional_embedding(positions)
                    x = embedded + pos_embed
                    causal_mask = tf.linalg.band_part(tf.ones([current_seq_len, current_seq_len], dtype=tf.float32), -1, 0)
                    causal_mask = tf.expand_dims(causal_mask, axis=0)
                    current_cross_attention_mask = cross_attention_mask[:, start:end, :] if cross_attention_mask is not None else None
                    with tf.device('/GPU:0'):
                        for block_idx, block in enumerate(self.transformer_blocks):
                            past_key, past_value = past_key_values[block_idx] if past_key_values[block_idx] else (None, None)
                            x, current_key, current_value = block(x, memory_embeddings, causal_mask, current_cross_attention_mask, past_key, past_value, training=training)
                            if past_key is None:
                                past_key_values[block_idx] = (current_key, current_value)
                            else:
                                past_key_values[block_idx] = (
                                    tf.concat([past_key, current_key], axis=1),  # Fix: Changed from axis=2 to axis=1
                                    tf.concat([past_value, current_value], axis=1)  # Fix: Changed from axis=2 to axis=1
                                )
                        current_logits = self.final_dense(x)
                    logits_list.append(current_logits)
                logits = tf.concat(logits_list, axis=1)
            return logits

class HSMN(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len=512, compressed_dim=128, tokenizer=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.compressed_dim = compressed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id if tokenizer else 1  # Fallback to 1 if tokenizer is None
        self.pruning_step_tracker = tf.Variable(0, dtype=tf.int64, trainable=False, name="pruning_step_tracker")
        
        # Pruning parameters for sparsity
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=0.5, begin_step=0, frequency=100
            )
        }
        
        # Initialize components
        self.chunker = StreamingChunker(chunk_size=128, stride=64, min_chunks=2, tokenizer=tokenizer, process_on_cpu=True)
        self.encoder = ChunkEncoder(
            vocab_size=vocab_size,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            ff_dim=2048,
            dropout=0.1,
            max_seq_len=max_seq_len
        )
        self.aggregator = Aggregator(compressed_dim=compressed_dim)
        self.reasoning = ReasoningModule(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len
        )
        self.compressor = tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(
                compressed_dim,
                name=f"compressor_{id(self)}",  # Unique ID-based name
                bias_initializer=tf.keras.initializers.Constant(0.01)
            ),
            **pruning_params
        )
        self.decompressor = tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(
                512,
                name="decompressor",
                bias_initializer=tf.keras.initializers.Constant(0.01)
            ),
            **pruning_params
        )

    @tf.function
    def build_memory_embeddings(self, compressed_embeddings, training=False):
        """Build memory tree by aggregating compressed embeddings."""
        logging.info(f"HSMN.build_memory_embeddings: Starting with input shape: {compressed_embeddings.shape}")
        
        all_embeddings_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        for i in tf.range(tf.shape(compressed_embeddings)[0]):
            all_embeddings_ta = all_embeddings_ta.write(i, compressed_embeddings[i])
            logging.debug(f"Added initial embedding at index {i}, shape: {compressed_embeddings[i].shape}")
        
        current_level = compressed_embeddings
        
        def condition(all_emb, curr):
            return tf.shape(curr)[0] > 1
        
        def body(all_emb, curr):
            num_nodes = tf.shape(curr)[0]
            num_pairs = num_nodes // 2
            logging.info(f"Aggregating level with {num_nodes} nodes, {num_pairs} pairs")
            
            if num_pairs > 0:
                child1 = curr[0:2*num_pairs:2]
                child2 = curr[1:2*num_pairs:2]
                logging.debug(f"Child1 shape: {child1.shape}, Child2 shape: {child2.shape}")
                aggregated = self.aggregator(child1, child2, training=training)
                logging.info(f"Aggregator output shape: {aggregated.shape}")
                if num_nodes % 2 == 1:
                    next_level = tf.concat([aggregated, tf.expand_dims(curr[-1], 0)], axis=0)
                    logging.debug(f"Odd number of nodes, concatenated last node, next_level shape: {next_level.shape}")
                else:
                    next_level = aggregated
            else:
                next_level = curr
                logging.info("No pairs to aggregate, passing current level unchanged")
            
            for i in tf.range(tf.shape(next_level)[0]):
                all_emb = all_emb.write(all_emb.size(), next_level[i])
                logging.debug(f"Added embedding at index {all_emb.size()-1}, shape: {next_level[i].shape}")
            
            return all_emb, next_level
        
        all_embeddings_ta, final_level = tf.while_loop(
            condition,
            body,
            loop_vars=[all_embeddings_ta, current_level],
            shape_invariants=[None, tf.TensorShape([None, self.compressed_dim])],
            swap_memory=True
        )
        
        for i in tf.range(tf.shape(final_level)[0]):
            all_embeddings_ta = all_embeddings_ta.write(all_embeddings_ta.size(), final_level[i])
            logging.debug(f"Added final level embedding at index {all_embeddings_ta.size()-1}, shape: {final_level[i].shape}")
        
        all_embeddings = all_embeddings_ta.stack()
        logging.info(f"HSMN.build_memory_embeddings: Output shape {all_embeddings.shape}")
        return all_embeddings

    def call(self, inputs, training=False):
        context_tokens, target_tokens = inputs
        logging.info("HSMN.call: Starting forward pass")

        # Ensure batch dimension
        if len(context_tokens.shape) == 1:
            context_tokens = tf.expand_dims(context_tokens, 0)
        if len(target_tokens.shape) == 1:
            target_tokens = tf.expand_dims(target_tokens, 0)

        batch_size = tf.shape(context_tokens)[0]
        target_seq_len = tf.shape(target_tokens)[1]
        memory_embeddings_list = []

        # Process each sequence sequentially to minimize VRAM usage
        for i in range(batch_size):
            with tf.device('/CPU:0'):  # Compute on CPU to offload to RAM
                context_i = context_tokens[i:i+1]  # [1, seq_len]
                chunks_i = self.chunker(context_i, is_target=False)  # [num_chunks, 1, chunk_size]
                num_chunks = tf.shape(chunks_i)[0]
                chunks_i = tf.transpose(chunks_i, [1, 0, 2])  # [1, num_chunks, chunk_size]
                logging.info(f"Processing sequence {i+1}/{batch_size}, {num_chunks} context chunks")

                # Move to GPU for ChunkEncoder
                with tf.device('/GPU:0'):
                    embeddings_i = self.encoder(chunks_i, training=training)  # [1, num_chunks, 512]
                    compressed_i = self.compressor(embeddings_i)  # [1, num_chunks, 128]

                # Back to CPU for memory tree construction
                memory_i = self.build_memory_embeddings(compressed_i[0], training=training)  # [num_nodes, 128]
                memory_embeddings_list.append(memory_i)
                logging.debug(f"Stored memory embeddings for sequence {i+1}, nodes={tf.shape(memory_i)[0]}")

                # Log VRAM usage
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                logging.info(f"After sequence {i+1}: VRAM usage: {memory_info['current'] / 1024**2:.2f} MB")

        # Pad and stack memory embeddings on CPU
        with tf.device('/CPU:0'):
            max_nodes = tf.reduce_max([tf.shape(m)[0] for m in memory_embeddings_list])
            memory_embeddings_padded = [
                tf.pad(m, [[0, max_nodes - tf.shape(m)[0]], [0, 0]], constant_values=0.0)
                for m in memory_embeddings_list
            ]
            memory_embeddings_batch = tf.stack(memory_embeddings_padded, axis=0)  # [batch_size, max_nodes, 128]
            logging.info(f"Stacked memory embeddings: shape={memory_embeddings_batch.shape}")

        # Move to GPU for reasoning
        with tf.device('/GPU:0'):
            memory_embeddings_batch_decompressed = self.decompressor(memory_embeddings_batch)  # [batch_size, max_nodes, 512]
            num_nodes = tf.shape(memory_embeddings_batch)[1]
            
            # Create cross-attention mask for full target sequence
            cross_attention_mask = tf.ones([batch_size, target_seq_len, num_nodes], dtype=tf.float32)

            # Process target chunks in ReasoningModule (handled internally)
            logits = self.reasoning(
                target_tokens,
                memory_embeddings_batch_decompressed,
                cross_attention_mask,
                training=training
            )
            logging.info(f"Completed reasoning, logits shape={logits.shape}")

            # Final VRAM check
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            logging.info(f"After reasoning: VRAM usage: {memory_info['current'] / 1024**2:.2f} MB")

        return logits

    def reason(self, target_tokens, memory_embeddings, cross_attention_mask, training=False):
        """Perform reasoning with batched target tokens and memory embeddings."""
        if self.reasoning is None:
            logging.error("Reasoning module is None in HSMN.reason")
            raise ValueError("Reasoning module is None")
        logits = self.reasoning(target_tokens, memory_embeddings, cross_attention_mask, training=training)
        return logits

    def generate(self, context, max_length=128000 , temperature=1.0, task="chat"):
        generated_tokens = [tokenizer.cls_token_id]
        
        # Tokenize full context without truncation
        context_tokens = tokenizer(context, add_special_tokens=False, truncation=False)['input_ids']
        context_tokens_tf = tf.constant([context_tokens], dtype=tf.int32)
        
        # Process context into memory tree
        chunks = self.chunker(context_tokens_tf)
        num_chunks = tf.shape(chunks)[0]
        chunk_size = tf.shape(chunks)[2]
        chunks_flat = tf.reshape(chunks, [num_chunks, chunk_size])
        embeddings = self.encoder(chunks_flat, training=False)
        embeddings = tf.reshape(embeddings, [num_chunks, 1, -1])
        compressed_embeddings = self.compressor(embeddings)
        item_compressed_embeddings = compressed_embeddings[:, 0, :]
        item_memory_embeddings = self.build_memory_embeddings(item_compressed_embeddings, training=False)
        item_memory_embeddings = self.decompressor(item_memory_embeddings)
        
        num_nodes = tf.shape(item_memory_embeddings)[0]
        memory_embeddings_batch = tf.expand_dims(item_memory_embeddings, 0)
        
        # Generate until EOS or max_length
        while len(generated_tokens) < max_length:
            target_tokens_tf = tf.constant([generated_tokens], dtype=tf.int32)
            current_seq_len = len(generated_tokens)
            cross_attention_mask = tf.ones([1, current_seq_len, num_nodes], dtype=tf.float32)
            
            logits = self.reason(target_tokens_tf, memory_embeddings_batch, cross_attention_mask, training=False)
            next_token_logits = logits[0, -1, :] / temperature
            next_token = tf.random.categorical([next_token_logits], num_samples=1)[0, 0].numpy()
            generated_tokens.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
        
        decoded = tokenizer.decode(generated_tokens[1:])
        if task == "code":
            decoded = decoded.strip()
        return decoded

def get_context(msg_id, id_to_msg, max_context_len=1000):
    """
    Build conversation context for a given message ID by traversing parent messages.

    Args:
        msg_id: ID of the current message.
        id_to_msg: Dictionary mapping message IDs to message dictionaries.
        max_context_len: Maximum length of the context string.

    Returns:
        String containing the concatenated context, or empty string if none.
    """
    context = []
    current = msg_id
    while current is not None:
        current_msg = id_to_msg.get(current)
        if current_msg is None or not current_msg.get('text'):
            break
        context.append(current_msg['text'])
        current = current_msg.get('parent_id')
    context.reverse()
    if len(context) > 1:
        context_str = "\n".join(context[:-1])[:max_context_len]
        return context_str
    return ""

def preprocess_bigbench_hard(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        question = item.get("input", "").strip()
        target = item.get("target", "").strip()
        if not question or not target:
            continue
        yield {"context": question, "target": target, "task": "chat"}

def preprocess_hellaswag(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("ctx", "").strip()
        endings = item.get("endings", [])
        label = int(item.get("label", "-1"))  # Convert label to integer, default "-1" to handle missing
        if not context or not endings or label < 0:
            continue
        target = endings[label].strip()
        yield {"context": context, "target": target, "task": "chat"}

def preprocess_kjv_bible(dataset) -> Generator[Dict[str, str], None, None]:
    """
    Preprocess the KJV Bible dataset, grouping by chapter to create context-target pairs.
    
    Args:
        dataset: Hugging Face dataset object containing KJV Bible text with fields "Book", "Chapter", "Text".
    
    Yields:
        Dictionary with 'context', 'target', and 'task' keys for each valid pair within the same chapter.
    """
    current_book = None
    current_chapter = None
    current_verses = []
    for item in dataset:
        book = item["Book"]
        chapter = item["Chapter"]
        text = item["Text"].strip()  # Corrected field name to "Text"
        if book != current_book or chapter != current_chapter:
            if current_verses:
                # Process current_verses for the previous chapter
                context_length = 3
                for i in range(len(current_verses) - context_length):
                    context = " ".join(current_verses[i:i + context_length])
                    target = current_verses[i + context_length]
                    if context and target:
                        yield {"context": context, "target": target, "task": "chat"}
            # Reset for the new chapter
            current_book = book
            current_chapter = chapter
            current_verses = [text]
        else:
            current_verses.append(text)
    # Process the last chapter
    if current_verses:
        context_length = 3
        for i in range(len(current_verses) - context_length):
            context = " ".join(current_verses[i:i + context_length])
            target = current_verses[i + context_length]
            if context and target:
                yield {"context": context, "target": target, "task": "chat"}

def preprocess_sciq(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        support = item.get("support", "").strip()
        question = item.get("question", "").strip()
        correct_answer = item.get("correct_answer", "").strip()
        if not question or not correct_answer:
            logging.debug(f"Skipping invalid item: {item}")
            continue
        context = (support + " " + question).strip() if support else question
        yield {"context": context, "target": correct_answer, "task": "chat"}

def preprocess_gsm8k(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("question", "").strip()
        target = item.get("answer", "").split("#### ")[-1].strip()
        if context and target:
            yield {"context": context, "target": target, "task": "chat"}

def preprocess_mmlu(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("question", "").strip()
        answer_idx = item.get("answer", -1)
        choices = item.get("choices", [])
        if not context or answer_idx < 0 or not choices:
            continue
        target = choices[answer_idx].strip()
        yield {"context": context, "target": target, "task": "chat"}

def preprocess_arc(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("question", "").strip()
        choices = item.get("choices", {}).get("text", [])
        labels = item.get("choices", {}).get("label", [])
        correct_label = item.get("answerKey", "")
        if not context or not choices or not labels or not correct_label:
            continue
        try:
            correct_idx = labels.index(correct_label)
            target = choices[correct_idx].strip()
            yield {"context": context, "target": target, "task": "chat"}
        except ValueError:
            logging.debug(f"Skipping item with invalid answerKey: {item}")

def preprocess_math(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("problem", "").strip()
        target = item.get("solution", "").strip()
        if context and target:
            yield {"context": context, "target": target, "task": "code"}

def preprocess_stem(dataset: list[Dict[str, Any]]) -> Generator[Dict[str, str], None, None]:
    """
    Preprocess the STEM dataset, filtering for text-based questions and extracting context-target pairs.
    
    Args:
        dataset: List of dictionaries containing STEM dataset items with fields like 'problem', 'choices', etc.
    
    Yields:
        Dictionary with 'context', 'target', and 'task' for each valid text-based question.
    """
    for item in dataset:
        # Skip items with image-based choices since we need text targets
        if item.get("pic_choice", False):
            continue
        
        # Extract problem as context
        problem = item.get("problem", "").strip()
        
        # Extract choices and answer index
        choices = item.get("choices", [])
        answer_idx = item.get("answer_idx", -1)
        
        # Validate and construct target
        if problem and choices and 0 <= answer_idx < len(choices):
            target = choices[answer_idx].strip()
            yield {"context": problem, "target": target, "task": "chat"}

def preprocess_code_search_net(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("docstring", "").strip()
        target = item.get("code", "").strip()
        if context and target:
            yield {"context": context, "target": target, "task": "code"}

def preprocess_human_eval(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("prompt", "").strip()
        target = item.get("canonical_solution", "").strip()
        if context and target:
            yield {"context": context, "target": target, "task": "code"}

def preprocess_mbpp(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        context = item.get("text", "").strip()
        target = item.get("code", "").strip()
        if context and target:
            yield {"context": context, "target": target, "task": "code"}

def preprocess_daily_dialog(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        dialogue = item.get("dialog", [])
        if not dialogue or len(dialogue) < 2:
            continue
        for t in range(len(dialogue) - 1):
            context = " ".join(dialogue[:t + 1]).strip()
            target = dialogue[t + 1].strip()
            if context and target:
                yield {"context": context, "target": target, "task": "chat"}

def preprocess_personachat(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        utterances = item.get("utterances", [])
        for utterance in utterances:
            history = utterance.get("history", [])
            candidates = utterance.get("candidates", [])
            if not history or not candidates:
                continue
            context = " ".join(history).strip()
            target = candidates[-1].strip()
            if context and target:
                yield {"context": context, "target": target, "task": "chat"}

def preprocess_openassistant(dataset) -> Generator[Dict[str, str], None, None]:
    logging.info("Preprocessing OpenAssistant dataset...")
    id_to_msg = {msg['message_id']: msg for msg in dataset if msg.get('message_id') and msg.get('text')}
    assistant_msgs = [
        msg for msg in dataset
        if msg.get('role') == 'assistant' and msg.get('text') and msg.get('message_id')
    ]
    if not assistant_msgs:
        logging.warning("No valid assistant messages found in OpenAssistant dataset.")
        return
    logging.info(f"Found {len(assistant_msgs)} valid assistant messages.")
    for idx, assistant_msg in enumerate(assistant_msgs):
        context = get_context(assistant_msg['message_id'], id_to_msg)
        if not context:
            logging.debug(f"Skipping assistant message {idx+1} due to empty context.")
            continue
        target = assistant_msg['text'].strip()
        yield {"context": context, "target": target, "task": "chat"}

def preprocess_truthfulqa(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        question = item.get("question", "").strip()
        best_answer = item.get("best_answer", "").strip()
        if question and best_answer:
            yield {"context": question, "target": best_answer, "task": "chat"}

def preprocess_apps(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        problem = item.get("problem", "").strip()
        solution = item.get("solutions", [""])[0].strip()
        if problem and solution:
            yield {"context": problem, "target": solution, "task": "code"}

def preprocess_instruction_dataset(dataset, context_field="context", target_field="response"):
    """
    Generic preprocessor for instruction-based datasets.
    
    Args:
        dataset: Hugging Face dataset object.
        context_field: Field name for context/input (default: 'context').
        target_field: Field name for target/response (default: 'response').
    
    Yields:
        Dictionary with 'context', 'target', and 'task' keys for each valid pair.
    """
    for i, item in enumerate(dataset):
        if i == 0:
            logging.info(f"Sample item from dataset: {item}")
        instruction = item.get("instruction", "").strip()
        input_text = item.get(context_field, "").strip()
        target = item.get(target_field, "").strip()
        if not instruction or not target:
            logging.debug(f"Skipping item {i} with empty instruction or target")
            continue
        context = (instruction + " " + input_text).strip() if input_text else instruction
        task = "code" if "code" in instruction.lower() or "programming" in instruction.lower() else "chat"
        yield {"context": context, "target": target, "task": task}

def preprocess_mathqa(dataset) -> Generator[Dict[str, str], None, None]:
    """
    Preprocess the mathqa dataset, fixing the key case sensitivity issue.
    
    Args:
        dataset: Input dataset with mathqa items.
    
    Yields:
        Dictionary with 'context', 'target', and 'task' for each valid item.
    """
    import tensorflow as tf
    for item in dataset:
        problem = item.get("Problem", "").strip()  # Changed from "problem" to "Problem"
        options = item.get("options", [])
        correct_option = item.get("correct", "").strip().lower()
        if not problem:
            logging.debug(f"Skipping item due to empty problem: {item}")
            continue
        
        if not options and not correct_option:
            logging.warning(f"No options or correct answer for problem: {problem[:50]}...")
            continue
        
        option_map = {}
        if options:
            for opt in options:
                opt = str(opt).strip()
                if opt and len(opt) > 2:
                    if opt[0].lower() in 'abcde':
                        letter = opt[0].lower()
                        answer_text = opt[2:].strip()
                        option_map[letter] = answer_text
                    else:
                        logging.debug(f"Skipping malformed option: {opt}")
        
        correct_answer = ""
        if correct_option and option_map:
            correct_answer = option_map.get(correct_option, "")
        elif correct_option and not option_map:
            correct_answer = correct_option
        
        if problem and correct_answer:
            yield {"context": problem, "target": correct_answer, "task": "chat"}
        else:
            logging.debug(f"Skipping item, no valid answer for problem: {problem[:50]}..., correct_option={correct_option}")

def preprocess_wizardlm(dataset) -> Generator[Dict[str, str], None, None]:
    return preprocess_instruction_dataset(dataset)

def preprocess_strategyqa(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        question = item.get("question", "").strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            yield {"context": question, "target": answer, "task": "chat"}

def preprocess_codecontests(dataset) -> Generator[Dict[str, str], None, None]:
    empty_solution_count = 0
    for item in dataset:
        problem = item.get("description", "").strip()
        solutions_list = item.get("solutions", {}).get("solution", [])
        if not solutions_list:
            empty_solution_count += 1
        solution = solutions_list[0].strip() if solutions_list else ""
        if problem and solution:
            yield {"context": problem, "target": solution, "task": "code"}
    print(f"Number of items with empty solutions: {empty_solution_count}")

def preprocess_natural_questions(dataset) -> Generator[Dict[str, str], None, None]:
    """
    Preprocess the Natural Questions (NQ Open) dataset to create context-target pairs.

    Args:
        dataset: Input dataset (list or Hugging Face dataset) with question and answer fields.

    Yields:
        Dictionary with 'context', 'target', and 'task' keys for each valid item.
    """
    for item in dataset:
        question = item.get("question", "").strip()
        # Handle cases where answer is a list (as in NQ Open) or a single string
        answer = item.get("answer", [""])[0].strip() if isinstance(item.get("answer"), list) else item.get("answer", "").strip()
        if question and answer:
            yield {"context": question, "target": answer, "task": "chat"}
        else:
            logging.debug(f"Skipping invalid item: question='{question[:50]}...', answer='{answer[:50]}...'")

def preprocess_aqua_rat(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        question = item.get("question", "").strip()
        rationale = item.get("rationale", "").strip()
        answer = item.get("correct", "").strip()
        if question and answer:
            context = f"{question} Rationale: {rationale}" if rationale else question
            yield {"context": context, "target": answer, "task": "chat"}

def preprocess_ultrafeedback(dataset):
    """
    Preprocess the ultrafeedback dataset by iterating over items and yielding context-target pairs.

    Args:
        dataset: An iterable (e.g., list or Hugging Face Dataset) containing ultrafeedback items.

    Yields:
        Dictionary with 'context', 'target', and 'task' keys for each valid item.
    """
    for item in dataset:
        # Extract prompt (instruction)
        instruction = item.get("prompt", "").strip()
        
        # Extract chosen response
        chosen = item.get("chosen", [])
        response = ""
        if isinstance(chosen, list) and chosen:
            # Assume the last message with role 'assistant' contains the response
            for msg in reversed(chosen):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                    response = msg["content"].strip()
                    break
        elif isinstance(chosen, str):
            response = chosen.strip()

        # Validate and yield
        if instruction and response:
            task = "code" if "code" in instruction.lower() or "programming" in instruction.lower() else "chat"
            yield {"context": instruction, "target": response, "task": task}
        else:
            logging.debug(f"Skipping item with invalid instruction or response: {item}")

def preprocess_bbh(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        question = item.get("input", "").strip()
        target = item.get("target", "").strip()
        if question and target:
            yield {"context": question, "target": target, "task": "chat"}

def preprocess_wikiqa(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        question = item.get("question", "").strip()
        answers = item.get("answers", [])
        labels = item.get("labels", [])
        if not question or not answers or not labels:
            continue
        for answer, label in zip(answers, labels):
            if label == 1:
                yield {"context": question, "target": answer, "task": "chat"}
                break

def preprocess_open_r1_math(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        problem = item.get("problem", "").strip()
        solution = item.get("solution", "").strip()
        if problem and solution:
            yield {"context": problem, "target": solution, "task": "code"}

def preprocess_nq_open(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        question = item.get("question", "").strip()
        answer = item.get("answer", [""])[0].strip()
        if question and answer:
            yield {"context": question, "target": answer, "task": "chat"}

def preprocess_owasp(dataset):
    for item in dataset:
        yield {"context": item["question"], "target": item["answer"], "task": "chat"}

def preprocess_dolly(dataset):
    for item in dataset:
        instruction = item.get("instruction", "").strip()
        context_text = item.get("context", "").strip()
        target = item.get("response", "").strip()
        if not instruction or not target:
            logging.debug(f"Skipping item with empty instruction or target: {item}")
            continue
        full_context = (context_text + " " + instruction).strip() if context_text else instruction
        task = "code" if "code" in instruction.lower() or "programming" in instruction.lower() else "chat"
        logging.debug(f"Processed item: context='{full_context[:50]}...', target='{target[:50]}...', task={task}")
        yield {"context": full_context, "target": target, "task": task}

def preprocess_code_alpaca(dataset) -> Generator[Dict[str, str], None, None]:
    for item in dataset:
        output = item.get("output", "").strip()
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if not instruction or not output:
            continue
        context = (instruction + " " + input_text).strip() if input_text else instruction
        task = "code" if "code" in instruction.lower() or "programming" in instruction.lower() else "chat"
        yield {"context": context, "target": output, "task": task}

def preprocess_alpaca(dataset) -> Generator[Dict[str, str], None, None]:
    """
    Preprocess the Alpaca dataset using preprocess_instruction_dataset with specific field names.
    
    Args:
        dataset: Hugging Face dataset object containing Alpaca data.
    
    Yields:
        Dictionary with 'context', 'target', and 'task' keys for each valid pair.
    """
    return preprocess_instruction_dataset(dataset, context_field="input", target_field="output")

def get_context(msg_id, id_to_msg, max_context_len=1000):
    context = []
    current = msg_id
    while current is not None:
        current_msg = id_to_msg.get(current)
        if current_msg is None or not current_msg.get('text'):
            break
        context.append(current_msg['text'])
        current = current_msg.get('parent_id')
    context.reverse()
    if len(context) > 1:
        context_str = "\n".join(context[:-1])[:max_context_len]
        return context_str
    return ""

# Preprocessors Dictionary
PREPROCESSORS = {
    "sciq": preprocess_sciq,
    "gsm8k": preprocess_gsm8k,
    "mmlu": preprocess_mmlu,
    "arc_easy": preprocess_arc,
    "arc_challenge": preprocess_arc,
    "math": preprocess_math,
    "stem": preprocess_stem,
    "code_search_net": preprocess_code_search_net,
    "human_eval": preprocess_human_eval,
    "mbpp": preprocess_mbpp,
    "daily_dialog": preprocess_daily_dialog,
    "personachat": preprocess_personachat,
    "open_assistant": preprocess_openassistant,
    "code_alpaca": preprocess_code_alpaca,
    "dolly": preprocess_dolly,
    "truthfulqa": preprocess_truthfulqa,
    "kjv_bible": preprocess_kjv_bible,
    "hellaswag": preprocess_hellaswag,
    "bigbench_hard": preprocess_bigbench_hard,
    "apps": preprocess_apps,
    "mathqa": preprocess_mathqa,
    "wizardlm": preprocess_wizardlm,
    "strategyqa": preprocess_strategyqa,
    "codecontests": preprocess_codecontests,
    "natural_questions": preprocess_natural_questions,
    "aqua_rat": preprocess_aqua_rat,
    "ultrafeedback": preprocess_ultrafeedback,
    "bbh": preprocess_bbh,
    "wikiqa": preprocess_wikiqa,
    "open_r1_math": preprocess_math,
    "nq_open": preprocess_natural_questions,
    "owasp": preprocess_owasp,
    "alpaca": preprocess_alpaca,
}

def compute_fisher(model, validation_data, batch_size=32):
    """
    Compute the Fisher Information Matrix diagonal for the model's trainable variables using the updated call method.

    Args:
        model: HSMN model instance with updated call method.
        validation_data: List of dictionaries with 'context' and 'target' keys.
        batch_size: Batch size for processing validation data (default: 32).

    Returns:
        Dictionary mapping variable names to their Fisher diagonal values.
    """
    # Validate inputs
    if not validation_data or not isinstance(validation_data, list):
        logging.error("Validation data must be a non-empty list")
        return {}
    
    # Initialize Fisher dictionary
    fisher = {var.name: tf.zeros_like(var) for var in model.trainable_variables}
    num_batches = 0

    # Prepare dataset
    contexts = [item['context'] for item in validation_data if 'context' in item and 'target' in item]
    targets = [item['target'] for item in validation_data if 'context' in item and 'target' in item]
    if not contexts or len(contexts) != len(targets):
        logging.error("Invalid validation data format or empty after filtering")
        return fisher
    
    dataset = tf.data.Dataset.from_tensor_slices((contexts, targets)).batch(batch_size)

    # Process batches
    for batch_idx, (batch_contexts, batch_targets) in enumerate(dataset):
        with tf.GradientTape() as tape:
            losses = []
            for context, target in zip(batch_contexts, batch_targets):
                try:
                    # Tokenize target
                    target_str = target.numpy().decode('utf-8')
                    target_tokens = tokenizer(target_str, add_special_tokens=True,
                                            truncation=True, max_length=model.max_seq_len,
                                            padding='max_length')['input_ids']
                    target_tokens_tf = tf.constant([target_tokens], dtype=tf.int32)

                    # Compute logits using call method
                    logits = model((context, target_tokens_tf), training=False)

                    # Compute per-sample loss
                    labels = target_tokens_tf[:, 1:]
                    logits = logits[:, :-1, :]
                    loss_per_token = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction='none')(labels, logits)
                    mask = tf.cast(labels != tokenizer.pad_token_id, tf.float32)
                    loss = tf.reduce_sum(loss_per_token * mask) / tf.reduce_sum(mask)

                    if tf.math.is_finite(loss):
                        losses.append(loss)
                    else:
                        logging.warning(f"Non-finite loss for context: {context[:50]}...")
                except Exception as e:
                    logging.error(f"Error processing sample: {str(e)}")
                    continue

            # Compute mean loss for the batch
            mean_loss = tf.reduce_mean(losses) if losses else tf.constant(0.0, dtype=tf.float32)

        # Accumulate gradients if loss is finite
        if tf.math.is_finite(mean_loss):
            grads = tape.gradient(mean_loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, grads):
                if grad is not None:
                    fisher[var.name] += tf.square(grad)
            num_batches += 1
        else:
            logging.warning(f"Batch {batch_idx + 1}: Mean loss not finite, skipping")

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Processed {batch_idx + 1} batches")

    # Average Fisher values
    if num_batches > 0:
        for name in fisher:
            fisher[name] /= num_batches
        logging.info(f"Computed Fisher over {num_batches} batches")
    else:
        logging.warning("No batches processed successfully")

    return fisher

def compute_loss(model, context, target, loss_fn, previous_fishers, previous_optimal_weights, lambda_ewc=1000.0):
    """
    Computes the loss tensors for the HSMN model using CategoricalCrossentropy for full sequences.

    Args:
        model: HSMN model instance.
        context: Input context string (scalar tf.string or Python string).
        target: Target string (scalar tf.string or Python string).
        loss_fn: Loss function (unused; replaced with CategoricalCrossentropy).
        previous_fishers: List of Fisher Information Matrices from previous tasks.
        previous_optimal_weights: List of optimal weights from previous tasks.
        lambda_ewc: EWC penalty coefficient (default: 1000.0).

    Returns:
        Tuple of (standard_loss, ewc_penalty) as tf.Tensors.
    """
    logging.info(f"compute_loss: Starting for context length={len(context)}, target length={len(target)}")
    current_step = model.optimizer.iterations.numpy() if hasattr(model.optimizer, 'iterations') else -1
    logging.info(f"compute_loss: current_step={current_step}")

    # Validate inputs
    if not context or not target:
        logging.warning("compute_loss: Empty context or target, returning zeros")
        return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)
    logging.debug(f"compute_loss: Context sample: {context[:50]}...")
    logging.debug(f"compute_loss: Target sample: {target[:50]}...")

    if not hasattr(model, 'vocab_size'):
        logging.error("Model lacks vocab_size attribute")
        raise AttributeError("HSMN model must define vocab_size")
    logging.info(f"compute_loss: Model vocab_size={model.vocab_size}")

    # Tokenize context
    try:
        context_str = context.numpy().decode('utf-8') if isinstance(context, tf.Tensor) else str(context)
        context_tokens = tokenizer(context_str, add_special_tokens=False, truncation=False)['input_ids']
        context_tokens_tf = tf.constant([context_tokens], dtype=tf.int32)
        logging.info(f"compute_loss: Context tokens shape={context_tokens_tf.shape}")
    except Exception as e:
        logging.error(f"Error tokenizing context: {str(e)}")
        return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

    # Tokenize target
    try:
        target_str = target.numpy().decode('utf-8') if isinstance(target, tf.Tensor) else str(target)
        target_tokens = tokenizer(target_str, add_special_tokens=True, truncation=False)['input_ids']
        target_tokens_tf = tf.constant([target_tokens], dtype=tf.int32)
        num_non_pad = tf.reduce_sum(tf.cast(target_tokens_tf != tokenizer.pad_token_id, tf.int32))
        logging.info(f"compute_loss: Target tokens shape={target_tokens_tf.shape}, non-pad tokens={num_non_pad.numpy()}")
        logging.debug(f"compute_loss: Target tokens first 10: {target_tokens_tf[0, :10].numpy()}")
    except Exception as e:
        logging.error(f"Error tokenizing target: {str(e)}")
        return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

    # Compute logits with tokenized inputs
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model((context_tokens_tf, target_tokens_tf), training=True)
        logging.info(f"compute_loss: Logits shape={logits.shape}")

    # Verify model connectivity with sum of logits
    sum_logits = tf.reduce_sum(logits)
    grads_check = tape.gradient(sum_logits, model.trainable_variables)
    none_grads = [var.name for var, grad in zip(model.trainable_variables, grads_check) if grad is None]
    if none_grads:
        logging.warning(f"compute_loss: {len(none_grads)} variables have None gradients: {none_grads[:5]}...")
    else:
        logging.info("compute_loss: All variables have gradients for sum_logits")
        for var, grad in zip(model.trainable_variables, grads_check):
            if grad is not None:
                logging.debug(f"compute_loss: Gradient norm for {var.name}: {tf.norm(grad).numpy():.4f}")

    # Verify logits
    if not tf.reduce_all(tf.math.is_finite(logits)):
        logging.warning("compute_loss: Logits contain non-finite values")
    logging.debug(f"compute_loss: Logits sample (first 5): {logits[0, 0, :5].numpy()}")

    # Prepare labels and compute loss with CategoricalCrossentropy
    labels = target_tokens_tf[:, 1:]  # Shifted target tokens
    logits_for_loss = logits[:, :-1, :]  # Align with labels
    logging.info(f"compute_loss: Labels shape={labels.shape}, Logits for loss shape={logits_for_loss.shape}")
    tf.debugging.assert_shapes(
        [(labels, (None, None)), (logits_for_loss, (None, None, model.vocab_size))],
        message="Labels or logits shape mismatch"
    )

    # Assume all tokens are valid (no padding)
    mask = tf.ones_like(labels, dtype=tf.float32)
    labels_one_hot = tf.one_hot(labels, depth=model.vocab_size)
    loss_fn_cat = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    loss_per_token = loss_fn_cat(labels_one_hot, logits_for_loss)
    sum_mask = tf.reduce_sum(mask)

    if tf.equal(sum_mask, 0.0):
        standard_loss = tf.constant(0.0, dtype=tf.float32)
        logging.warning("compute_loss: Mask sum is zero, setting standard_loss to 0.0")
    else:
        standard_loss = tf.reduce_sum(loss_per_token * mask) / sum_mask
    logging.info(f"compute_loss: Standard loss: {standard_loss.numpy():.4f}")
    if not tf.math.is_finite(standard_loss):
        logging.warning(f"compute_loss: Standard loss is non-finite: {standard_loss.numpy()}")

    # Compute EWC penalty
    ewc_penalty = tf.constant(0.0, dtype=tf.float32)
    if previous_fishers and previous_optimal_weights:
        try:
            logging.info(f"compute_loss: Computing EWC penalty for {len(previous_fishers)} previous tasks")
            for k in range(len(previous_fishers)):
                fisher_k = previous_fishers[k]
                opt_w_k = previous_optimal_weights[k]
                for var in model.trainable_variables:
                    if var.name in fisher_k:
                        F_k = fisher_k[var.name]
                        w_k = opt_w_k[var.name]
                        penalty_term = tf.reduce_sum(F_k * tf.square(var - w_k))
                        ewc_penalty += penalty_term
                        logging.debug(f"compute_loss: EWC penalty term for {var.name}: {penalty_term.numpy():.4f}")
            ewc_penalty *= lambda_ewc / 2
            logging.info(f"compute_loss: Total EWC penalty: {ewc_penalty.numpy():.4f}")
        except Exception as e:
            logging.error(f"Error computing EWC penalty: {str(e)}")
            ewc_penalty = tf.constant(0.0, dtype=tf.float32)

    return standard_loss, ewc_penalty

def test_compute_loss(model, tokenizer):
    """
    Test the compute_loss function with sample data to verify logits shape.
    """
    # Sample test data
    test_context = "What is the capital of France?"
    test_target = "The capital of France is Paris."
    
    # Define a dummy loss function (not used in compute_loss)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Dummy EWC parameters (empty for testing)
    previous_fishers = []
    previous_optimal_weights = []
    
    # Build the model with dummy data
    logging.info("Building model with dummy data for test_compute_loss...")
    dummy_context_tokens = tf.constant([[tokenizer.cls_token_id]], dtype=tf.int32)
    dummy_target_tokens = tf.constant(
        [[tokenizer.cls_token_id] + [tokenizer.pad_token_id] * (model.max_seq_len - 1)], 
        dtype=tf.int32
    )
    _ = model((dummy_context_tokens, dummy_target_tokens), training=False)
    
    # Get pruned layers and initialize pruning steps
    logging.info("Initializing pruning steps for test...")
    pruned_layers = get_pruned_layers(model)
    logging.info(f"Found {len(pruned_layers)} pruned layers for test")
    sync_pruning_step(model, pruned_layers)  # Use sync_pruning_step for consistency
    
    # Run compute_loss
    logging.info("Running compute_loss test...")
    try:
        standard_loss, ewc_penalty = compute_loss(
            model, test_context, test_target, loss_fn, 
            previous_fishers, previous_optimal_weights
        )
        logging.info(f"Test passed: Standard Loss={standard_loss.numpy():.4f}, EWC Penalty={ewc_penalty.numpy():.4f}")
        
        # Verify logits shape indirectly by checking the forward pass
        context_tokens = tokenizer(test_context, add_special_tokens=False,
                                   truncation=True, max_length=model.max_seq_len)['input_ids']
        target_tokens = tokenizer(test_target, add_special_tokens=True,
                                  truncation=True, max_length=model.max_seq_len,
                                  padding='max_length')['input_ids']
        context_tokens_tf = tf.constant([context_tokens], dtype=tf.int32)
        target_tokens_tf = tf.constant([target_tokens], dtype=tf.int32)
        logits = model((context_tokens_tf, target_tokens_tf), training=False)
        logging.info(f"Logits shape from model: {logits.shape}")
        assert len(logits.shape) == 3, f"Expected 3D logits, got {logits.shape}"
        assert logits.shape[0] == 1, "Batch size should be 1"
        assert logits.shape[1] == model.max_seq_len, "Sequence length mismatch"
        assert logits.shape[2] == model.vocab_size, "Vocab size mismatch"
        logging.info("Logits shape verification passed")
    except Exception as e:
        logging.error(f"Test failed: {str(e)}\n{traceback.format_exc()}")
        raise

def save_pruned_weights(model, path):
    """
    Save the model's weights directly, preserving pruning masks.
    
    Args:
        model: HSMN model instance.
        path: File path to save weights.
    """
    model.save_weights(path)
    logging.info(f"Saved weights to {path}")

def set_pruning_step(model, step):
    """
    Manually set the pruning_step for all pruned layers in the model.

    Args:
        model: The HSMN model instance.
        step: The step value to set for pruning (e.g., 1).
    """
    for layer in model.submodules:
        if hasattr(layer, 'pruning_step') and isinstance(layer.pruning_step, tf.Variable):
            layer.pruning_step.assign(step)
            logging.info(f"Set pruning_step to {step} for layer: {layer.name}")

def sync_pruning_step(model, pruned_layers):
    model.pruning_step_tracker.assign_add(1)
    current_step = model.pruning_step_tracker
    for layer in pruned_layers:
        tf.keras.backend.set_value(layer.pruning_step, current_step)
        logging.info(f"Layer {layer.name}: pruning_step set to {layer.pruning_step.numpy()}")
    logging.debug(f"Incremented pruning_step to {current_step.numpy()} for all pruned layers")

def create_split_generators(dataset, dataset_name: str, preprocess_fn) -> tuple[Generator[Dict[str, str], None, None], ...]:
    """
    Create generators for train, validation, and test splits using indices.

    Args:
        dataset: Hugging Face dataset object.
        dataset_name: Name of the dataset (e.g., 'dolly').
        preprocess_fn: Preprocessing function that processes one item at a time.

    Returns:
        Tuple of (train_generator, val_generator, test_generator).
    """
    num_items = len(dataset)
    indices = list(range(num_items))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)
    logging.info(f"Data split for {dataset_name}: Training: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_indices)}")

    def make_generator(indices):
        for idx in indices:
            item = dataset[idx]
            # Process the item directly without wrapping in a list
            try:
                for sample in preprocess_fn([item]):  # preprocess_fn expects an iterable
                    if sample and all(k in sample for k in ['context', 'target', 'task']):
                        yield sample
                    else:
                        logging.debug(f"Skipping invalid sample at index {idx}: {sample}")
            except Exception as e:
                logging.error(f"Error processing item at index {idx}: {str(e)}")
                continue

    train_gen = make_generator(train_indices)
    val_gen = make_generator(val_indices)
    test_gen = make_generator(test_indices)

    return train_gen, val_gen, test_gen

# Training Function for One Dataset
def train_on_dataset(model, dataset_name, dataset, epochs=200, batch_size=1, accum_steps=16,
                     model_path="hsmn_model", previous_fishers=None, previous_optimal_weights=None):
    """
    Trains the HSMN model on a single dataset with lazy loading, gradient accumulation, and EWC.
    Uses tf.data for efficient data processing to minimize VRAM usage (~6.3GB).
    Includes gradient debugging to diagnose None gradients and saves best weights per epoch.

    Args:
        model: The HSMN model instance.
        dataset_name: The name of the dataset being trained on.
        dataset: The loaded dataset from Hugging Face datasets.
        epochs: Number of training epochs.
        batch_size: Batch size for training data.
        accum_steps: Number of batches to accumulate gradients before applying.
        model_path: Base path for saving model weights.
        previous_fishers: List of Fisher Information Matrices from previous tasks (for EWC).
        previous_optimal_weights: List of optimal weights from previous tasks (for EWC).

    Returns:
        Tuple containing the trained model, updated list of previous_fishers, and updated list of previous_optimal_weights.
    """
    previous_fishers = previous_fishers or []
    previous_optimal_weights = previous_optimal_weights or []

    logging.info(f"Starting training on dataset: {dataset_name}")
    logging.info(f"Model trainable variables: {len(model.trainable_variables)}")

    # Check for duplicate layer names
    layer_names = [layer.name for layer in model.submodules]
    name_counts = Counter(layer_names)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        logging.warning(f"Duplicate layer names detected: {duplicates}")
    else:
        logging.info("No duplicate layer names found")

    def compute_fisher(model, data, batch_size=32):
        """
        Compute Fisher Information Matrix diagonal for validation data.
        """
        if not data or not isinstance(data, list):
            logging.error("Validation data must be a non-empty list")
            return {}

        fisher = {var.name: tf.zeros_like(var) for var in model.trainable_variables}
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch_items = data[i:i + batch_size]
            batch_losses = []

            for item in batch_items:
                try:
                    context_tokens = item['context_tokens']
                    target_tokens = item['target_tokens']
                    context_tokens_tf = tf.constant([context_tokens], dtype=tf.int32)
                    target_tokens_tf = tf.constant([target_tokens], dtype=tf.int32)

                    with tf.GradientTape() as tape:
                        logits = model((context_tokens_tf, target_tokens_tf), training=False)
                        log_probs = tf.nn.log_softmax(logits, axis=-1)
                        loss = tf.reduce_mean(log_probs)

                    grads = tape.gradient(loss, model.trainable_variables)
                    for var, grad in zip(model.trainable_variables, grads):
                        if grad is not None:
                            fisher[var.name] += tf.square(grad)
                    batch_losses.append(loss)
                except Exception as e:
                    logging.error(f"Error processing validation item: {e}")
                    continue

            if batch_losses:
                num_batches += 1

        if num_batches > 0:
            for name in fisher:
                fisher[name] /= num_batches
            logging.info(f"Computed Fisher over {num_batches} batches")
        else:
            logging.warning("No batches processed successfully")
        return fisher

    # Get preprocessor
    if dataset_name not in PREPROCESSORS:
        logging.warning(f"No preprocessor defined for dataset: {dataset_name}. Skipping.")
        return model, previous_fishers, previous_optimal_weights
    preprocess_fn = PREPROCESSORS[dataset_name]

    # Define generator functions for fresh data each epoch
    if dataset_name == "open_assistant":
        logging.info(f"Collecting all samples from OpenAssistant dataset with {len(dataset)} items")
        all_samples = list(preprocess_openassistant(dataset))
        logging.info(f"Collected {len(all_samples)} valid samples from OpenAssistant")

        if not all_samples:
            logging.warning("No valid samples found for OpenAssistant. Skipping training on this dataset.")
            return model, previous_fishers, previous_optimal_weights

        train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
        train_samples, val_samples = train_test_split(train_samples, test_size=0.25, random_state=42)
        logging.info(f"Split sizes: Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

        def train_gen_func():
            for sample in train_samples:
                if all(k in sample for k in ['context', 'target', 'task']):
                    context_str = sample['context']
                    target_str = sample['target']
                    context_tokens = tokenizer(context_str, add_special_tokens=False, truncation=True, max_length=model.max_seq_len)['input_ids']
                    target_tokens = tokenizer(target_str, add_special_tokens=True, truncation=True, max_length=model.max_seq_len, padding='max_length')['input_ids']
                    logging.debug(f"Tokenized context: {len(context_tokens)} tokens, target: {len(target_tokens)} tokens")
                    yield {
                        'context_tokens': context_tokens,
                        'target_tokens': target_tokens,
                        'task': sample['task']
                    }
                else:
                    logging.debug("Skipping invalid sample in train_gen_func")

        def val_gen_func():
            for sample in val_samples:
                if all(k in sample for k in ['context', 'target', 'task']):
                    context_str = sample['context']
                    target_str = sample['target']
                    context_tokens = tokenizer(context_str, add_special_tokens=False, truncation=True, max_length=model.max_seq_len)['input_ids']
                    target_tokens = tokenizer(target_str, add_special_tokens=True, truncation=True, max_length=model.max_seq_len, padding='max_length')['input_ids']
                    logging.debug(f"Tokenized context: {len(context_tokens)} tokens, target: {len(target_tokens)} tokens")
                    yield {
                        'context_tokens': context_tokens,
                        'target_tokens': target_tokens,
                        'task': sample['task']
                    }
                else:
                    logging.debug("Skipping invalid sample in val_gen_func")

        def test_gen_func():
            for sample in test_samples:
                if all(k in sample for k in ['context', 'target', 'task']):
                    context_str = sample['context']
                    target_str = sample['target']
                    context_tokens = tokenizer(context_str, add_special_tokens=False, truncation=True, max_length=model.max_seq_len)['input_ids']
                    target_tokens = tokenizer(target_str, add_special_tokens=True, truncation=True, max_length=model.max_seq_len, padding='max_length')['input_ids']
                    logging.debug(f"Tokenized context: {len(context_tokens)} tokens, target: {len(target_tokens)} tokens")
                    yield {
                        'context_tokens': context_tokens,
                        'target_tokens': target_tokens,
                        'task': sample['task']
                    }
                else:
                    logging.debug("Skipping invalid sample in test_gen_func")
    else:
        num_items = len(dataset)
        indices = list(range(num_items))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)
        logging.info(f"Data split: Training: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_indices)}")

        def train_gen_func():
            for idx in train_indices:
                item = dataset[idx]
                preprocessed = preprocess_fn([item])
                for sample in preprocessed:
                    if sample and all(k in sample for k in ['context', 'target', 'task']):
                        context_str = sample['context']
                        target_str = sample['target']
                        context_tokens = tokenizer(context_str, add_special_tokens=False, truncation=True, max_length=model.max_seq_len)['input_ids']
                        target_tokens = tokenizer(target_str, add_special_tokens=True, truncation=True, max_length=model.max_seq_len, padding='max_length')['input_ids']
                        logging.debug(f"Tokenized context: {len(context_tokens)} tokens, target: {len(target_tokens)} tokens")
                        yield {
                            'context_tokens': context_tokens,
                            'target_tokens': target_tokens,
                            'task': sample['task']
                        }
                    else:
                        logging.debug(f"Skipping invalid sample at index {idx}")

        def val_gen_func():
            for idx in val_indices:
                item = dataset[idx]
                preprocessed = preprocess_fn([item])
                for sample in preprocessed:
                    if sample and all(k in sample for k in ['context', 'target', 'task']):
                        context_str = sample['context']
                        target_str = sample['target']
                        context_tokens = tokenizer(context_str, add_special_tokens=False, truncation=True, max_length=model.max_seq_len)['input_ids']
                        target_tokens = tokenizer(target_str, add_special_tokens=True, truncation=True, max_length=model.max_seq_len, padding='max_length')['input_ids']
                        logging.debug(f"Tokenized context: {len(context_tokens)} tokens, target: {len(target_tokens)} tokens")
                        yield {
                            'context_tokens': context_tokens,
                            'target_tokens': target_tokens,
                            'task': sample['task']
                        }
                    else:
                        logging.debug(f"Skipping invalid sample at index {idx}")

        def test_gen_func():
            for idx in test_indices:
                item = dataset[idx]
                preprocessed = preprocess_fn([item])
                for sample in preprocessed:
                    if sample and all(k in sample for k in ['context', 'target', 'task']):
                        context_str = sample['context']
                        target_str = sample['target']
                        context_tokens = tokenizer(context_str, add_special_tokens=False, truncation=True, max_length=model.max_seq_len)['input_ids']
                        target_tokens = tokenizer(target_str, add_special_tokens=True, truncation=True, max_length=model.max_seq_len, padding='max_length')['input_ids']
                        logging.debug(f"Tokenized context: {len(context_tokens)} tokens, target: {len(target_tokens)} tokens")
                        yield {
                            'context_tokens': context_tokens,
                            'target_tokens': target_tokens,
                            'task': sample['task']
                        }
                    else:
                        logging.debug(f"Skipping invalid sample at index {idx}")

    # Function to create tf.data.Dataset with TensorFlow 2.10 optimizations
    def create_tf_dataset(generator, batch_size):
        return tf.data.Dataset.from_generator(
            lambda: generator,
            output_types={'context_tokens': tf.int32, 'target_tokens': tf.int32, 'task': tf.string},
            output_shapes={'context_tokens': [None], 'target_tokens': [model.max_seq_len], 'task': ()}
        ).padded_batch(
            batch_size,
            padded_shapes={'context_tokens': [None], 'target_tokens': [model.max_seq_len], 'task': []},
            padding_values={'context_tokens': tokenizer.pad_token_id, 'target_tokens': tokenizer.pad_token_id, 'task': ''}
        ).prefetch(tf.data.AUTOTUNE)

    # Initialize optimizer and loss with TensorFlow 2.10
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, clipvalue=1.0)
    model.optimizer = optimizer
    loss_fn_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Build model with dummy data
    try:
        logging.info("Building model with dummy data...")
        dummy_context_tokens = tf.constant([[tokenizer.cls_token_id] + tokenizer.encode("test input data")[:157]], dtype=tf.int32)
        dummy_target_tokens = tf.constant([tokenizer.encode("dummy answer", add_special_tokens=True,
                                                           max_length=model.max_seq_len, padding='max_length', truncation=True)], dtype=tf.int32)
        if hasattr(model, 'aggregator'):
            _ = model.aggregator(tf.zeros([2, model.compressed_dim]), tf.zeros([2, model.compressed_dim]))
            logging.info("Aggregator forward pass with dummy data completed")

        logging.info("Performing initial forward pass with training=False...")
        logits = model((dummy_context_tokens, dummy_target_tokens), training=False)
        logging.info(f"Initial forward pass: Logits shape={logits.shape}")

        # Get pruned layers after model is built
        pruned_layers = get_pruned_layers(model)
        logging.info(f"Found {len(pruned_layers)} pruned layers after build")
        for layer in pruned_layers:
            logging.info(f"Pruned layer detected: {layer.name}")
        if not pruned_layers:
            logging.warning("No pruned layers found after build. Pruning may not be applied correctly.")

        logging.info("Performing dummy training step to initialize pruning and gradients...")
        try:
            initial_weights = model.get_weights()
            sync_pruning_step(model, pruned_layers)

            with tf.GradientTape(persistent=True) as tape:
                logits = model((dummy_context_tokens, dummy_target_tokens), training=True)
                labels = dummy_target_tokens[:, 1:]
                logits_for_loss = logits[:, :-1, :]
                loss_per_token = loss_fn_sparse(labels, logits_for_loss)
                mask = tf.cast(labels != tokenizer.pad_token_id, tf.float32)
                dummy_loss = tf.reduce_sum(loss_per_token * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
                tf.debugging.check_numerics(dummy_loss, "Dummy loss contains NaN/Inf")
                logging.info(f"Dummy step: Loss={dummy_loss.numpy():.4f}, Logits shape={logits.shape}")

                # Debug connectivity with a simple loss
                connectivity_loss = tf.reduce_sum(logits)
                logging.info(f"Dummy step: Connectivity loss (sum of logits)={connectivity_loss.numpy():.4f}")

            # Compute gradients for main loss
            dummy_grads = tape.gradient(dummy_loss, model.trainable_variables)
            # Compute gradients for connectivity loss
            connectivity_grads = tape.gradient(connectivity_loss, model.trainable_variables)

            # Group variables by component for organized logging
            component_grads = {
                'ChunkEncoder': [],
                'Aggregator': [],
                'ReasoningModule': [],
                'Compressor': [],
                'Decompressor': [],
                'Other': []
            }
            for var, grad, conn_grad in zip(model.trainable_variables, dummy_grads, connectivity_grads):
                if 'chunk_encoder' in var.name.lower():
                    component = 'ChunkEncoder'
                elif 'aggregator' in var.name.lower():
                    component = 'Aggregator'
                elif 'reasoning_module' in var.name.lower():
                    component = 'ReasoningModule'
                elif 'compressor' in var.name.lower():
                    component = 'Compressor'
                elif 'decompressor' in var.name.lower():
                    component = 'Decompressor'
                else:
                    component = 'Other'
                component_grads[component].append((var.name, grad, conn_grad))

            # Log gradients by component
            logging.info("--- Dummy Step Gradient Report ---")
            none_count = 0
            zero_count = 0
            for component, grads_list in component_grads.items():
                logging.info(f"Component: {component}")
                for var_name, grad, conn_grad in grads_list:
                    if grad is None:
                        none_count += 1
                        grad_status = "None"
                    else:
                        grad_norm = tf.norm(grad).numpy()
                        grad_status = f"Norm={grad_norm:.6f}"
                        if grad_norm == 0.0:
                            zero_count += 1
                    conn_grad_status = "None" if conn_grad is None else f"Norm={tf.norm(conn_grad).numpy():.6f}"
                    logging.info(f"  Variable: {var_name}, Loss Gradient: {grad_status}, Connectivity Gradient: {conn_grad_status}")
            logging.info(f"Dummy Step: {none_count}/{len(model.trainable_variables)} variables have None gradients for loss")
            logging.info(f"Dummy Step: {zero_count}/{len(model.trainable_variables)} variables have zero gradients for loss")
            logging.info("--- End Dummy Step Gradient Report ---")

            optimizer.apply_gradients(zip([g for g in dummy_grads if g is not None],
                                         [v for g, v in zip(dummy_grads, model.trainable_variables) if g is not None]))

            model.set_weights(initial_weights)
            logging.info(f"Dummy step: Optimizer iterations={optimizer.iterations.numpy()}")

            if all(g is None for g in dummy_grads):
                logging.error("Dummy step failed: All gradients are None")
                return model, previous_fishers, previous_optimal_weights

            del tape
        except Exception as e:
            logging.error(f"Error in dummy training step: {str(e)}\n{traceback.format_exc()}")
            return model, previous_fishers, previous_optimal_weights

    except Exception as e:
        logging.error(f"Failed to build model: {str(e)}\n{traceback.format_exc()}")
        return model, previous_fishers, previous_optimal_weights

    # Training loop with dataset recreation per epoch
    best_val_loss = float('inf')
    patience = 10
    wait = 0
    accumulated_grads = [tf.zeros_like(var) for var in model.trainable_variables]

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        process = psutil.Process()
        mem_info = process.memory_info()
        logging.info(f"Memory usage before epoch: RSS={mem_info.rss / 1024**2:.2f} MB")

        # Recreate datasets for fresh data each epoch
        try:
            train_dataset = create_tf_dataset(train_gen_func(), batch_size=batch_size)
            val_dataset = create_tf_dataset(val_gen_func(), batch_size=1)
            logging.info(f"Created datasets for epoch {epoch + 1}: train_dataset, val_dataset")
        except Exception as e:
            logging.error(f"Error creating datasets for epoch {epoch + 1}: {str(e)}\n{traceback.format_exc()}")
            return model, previous_fishers, previous_optimal_weights

        # Training loop
        epoch_train_loss = 0
        num_batches = 0
        for batch_idx, batch in enumerate(train_dataset):
            try:
                context_tokens_tf = batch['context_tokens']
                target_tokens_tf = batch['target_tokens']
                logging.info(f"Batch {batch_idx + 1}: Context tokens shape={context_tokens_tf.shape}, Target tokens shape={target_tokens_tf.shape}")

                sync_pruning_step(model, pruned_layers)

                with tf.GradientTape(persistent=True) as tape:
                    logits = model((context_tokens_tf, target_tokens_tf), training=True)
                    logging.info(f"Batch {batch_idx + 1}: Logits shape={logits.shape}")
                    labels = target_tokens_tf[:, 1:]
                    logits_for_loss = logits[:, :-1, :]
                    loss_per_token = loss_fn_sparse(labels, logits_for_loss)
                    mask = tf.cast(labels != tokenizer.pad_token_id, tf.float32)
                    sum_mask = tf.reduce_sum(mask)
                    standard_loss = tf.constant(0.0) if tf.equal(sum_mask, 0.0) else tf.reduce_sum(loss_per_token * mask) / sum_mask
                    tf.debugging.check_numerics(standard_loss, f"Standard_loss batch {batch_idx + 1} NaN/Inf")

                    reg_loss = sum(tf.reduce_sum(tf.square(var)) * 1e-6 for var in model.trainable_variables)
                    ewc_penalty = tf.constant(0.0)
                    if previous_fishers and previous_optimal_weights:
                        for k in range(len(previous_fishers)):
                            fisher_k = previous_fishers[k]
                            opt_w_k = previous_optimal_weights[k]
                            for var in model.trainable_variables:
                                if var.name in fisher_k:
                                    F_k = fisher_k[var.name]
                                    w_k = opt_w_k[var.name]
                                    ewc_penalty += tf.reduce_sum(F_k * tf.square(var - w_k))

                    total_loss = standard_loss + (1000.0 / 2) * ewc_penalty + reg_loss
                    tf.debugging.check_numerics(total_loss, f"Total loss batch {batch_idx + 1} NaN/Inf")

                    logging.info(f"Batch {batch_idx + 1}: Total Loss={total_loss.numpy():.4f}, "
                                f"Standard Loss={standard_loss.numpy():.4f}, "
                                f"EWC Penalty={(1000.0 / 2) * ewc_penalty.numpy():.4f}, "
                                f"Reg Loss={reg_loss.numpy():.4f}")

                if not tf.math.is_finite(total_loss):
                    logging.warning(f"Non-finite loss ({total_loss.numpy()}) batch {batch_idx + 1}. Skipping.")
                    continue

                grads = tape.gradient(total_loss, model.trainable_variables)
                none_grads = [var.name for var, grad in zip(model.trainable_variables, grads) if grad is None]
                if none_grads:
                    logging.warning(f"Batch {batch_idx + 1}: {len(none_grads)} variables have None gradients: {none_grads[:5]}...")

                # Log gradient statistics
                component_grads = {
                    'ChunkEncoder': [],
                    'Aggregator': [],
                    'ReasoningModule': [],
                    'Compressor': [],
                    'Decompressor': [],
                    'Other': []
                }
                for var, grad in zip(model.trainable_variables, grads):
                    if 'chunk_encoder' in var.name.lower():
                        component = 'ChunkEncoder'
                    elif 'aggregator' in var.name.lower():
                        component = 'Aggregator'
                    elif 'reasoning_module' in var.name.lower():
                        component = 'ReasoningModule'
                    elif 'compressor' in var.name.lower():
                        component = 'Compressor'
                    elif 'decompressor' in var.name.lower():
                        component = 'Decompressor'
                    else:
                        component = 'Other'
                    component_grads[component].append((var.name, grad))

                logging.info(f"Batch {batch_idx + 1} Gradient Summary")
                for component, grads_list in component_grads.items():
                    none_count = sum(1 for _, g in grads_list if g is None)
                    zero_count = sum(1 for _, g in grads_list if g is not None and tf.norm(g).numpy() == 0.0)
                    total_vars = len(grads_list)
                    logging.info(f"  {component}: {none_count}/{total_vars} None, {zero_count}/{total_vars} Zero")
                    for var_name, grad in grads_list[:3]:
                        grad_status = "None" if grad is None else f"Norm={tf.norm(grad).numpy():.6f}"
                        logging.debug(f"    Variable: {var_name}, Gradient: {grad_status}")

                for i, grad in enumerate(grads):
                    if grad is not None:
                        accumulated_grads[i] += grad / accum_steps

                epoch_train_loss += total_loss.numpy()
                num_batches += 1

                if (batch_idx + 1) % accum_steps == 0:
                    grad_var_pairs = [(g, v) for g, v in zip(accumulated_grads, model.trainable_variables) if g is not None]
                    if grad_var_pairs:
                        optimizer.apply_gradients(grad_var_pairs)
                        logging.info(f"Applied gradients batch {batch_idx + 1}, iterations={optimizer.iterations.numpy()}")
                    accumulated_grads = [tf.zeros_like(var) for var in model.trainable_variables]

                if num_batches > 0:
                    logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Avg Loss={epoch_train_loss / num_batches:.4f}")

                del tape
            except Exception as e:
                logging.error(f"Error in batch {batch_idx + 1}: {str(e)}\n{traceback.format_exc()}")
                continue
            finally:
                gc.collect()

        # Validation loop
        val_loss_sum = 0
        val_batches = 0
        for batch in val_dataset:
            try:
                context_tokens_tf = batch['context_tokens']
                target_tokens_tf = batch['target_tokens']
                logits = model((context_tokens_tf, target_tokens_tf), training=False)
                labels = target_tokens_tf[:, 1:]
                logits_for_loss = logits[:, :-1, :]
                loss_per_token = loss_fn_sparse(labels, logits_for_loss)
                mask = tf.cast(labels != tokenizer.pad_token_id, tf.float32)
                sum_mask = tf.reduce_sum(mask)
                loss = tf.constant(0.0) if tf.equal(sum_mask, 0.0) else tf.reduce_sum(loss_per_token * mask) / sum_mask
                if tf.math.is_finite(loss):
                    val_loss_sum += loss.numpy()
                    val_batches += 1
            except Exception as e:
                logging.error(f"Error in validation batch: {str(e)}\n{traceback.format_exc()}")
                continue
            finally:
                gc.collect()

        if val_batches > 0:
            avg_val_loss = val_loss_sum / val_batches
            logging.info(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = f"{model_path}_{dataset_name}_best_epoch_{epoch + 1:02d}.h5"
                try:
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                        logging.info(f"Deleted existing file {checkpoint_path}")
                    model.save_weights(checkpoint_path)
                    logging.info(f"Saved best weights to {checkpoint_path} with loss={avg_val_loss:.4f}")
                    # Verify checkpoint integrity
                    model.load_weights(checkpoint_path)
                    logging.info(f"Successfully verified {checkpoint_path} by loading")
                except Exception as e:
                    logging.error(f"Error with {checkpoint_path}: {str(e)}\n{traceback.format_exc()}")
                    logging.warning("Continuing training despite weight saving error")
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logging.info(f"Early stopping epoch {epoch + 1}")
                    break
        else:
            logging.warning(f"No validation batches processed in epoch {epoch + 1}")
            wait += 1
            if wait >= patience:
                logging.info(f"Early stopping epoch {epoch + 1}")
                break

    # Compute Fisher
    val_data = []
    try:
        val_gen = val_gen_func()  # Recreate generator for Fisher computation
        for item in val_gen:
            val_data.append(item)
        if val_data:
            logging.info(f"Computing Fisher for {dataset_name}...")
            fisher = compute_fisher(model, val_data)
            if fisher:
                optimal_weights = {var.name: var.read_value() for var in model.trainable_variables}
                previous_fishers.append(fisher)
                previous_optimal_weights.append(optimal_weights)
                logging.info(f"Computed Fisher for {dataset_name}")
    except Exception as e:
        logging.error(f"Error computing Fisher: {str(e)}\n{traceback.format_exc()}")

    # Cleanup
    del optimizer
    tf.keras.backend.clear_session()
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
            logging.info("GPU memory stats reset")
        except Exception as e:
            logging.warning(f"Failed to reset GPU memory: {e}")

    return model, previous_fishers, previous_optimal_weights

# Define benchmark datasets and their evaluation splits
benchmark_datasets = [
    'alpaca', 'dolly', 'open_assistant', 'code_alpaca',
    'daily_dialog', 'personachat',
    'mmlu',
    'code_search_net', 'human_eval', 'mbpp', 'apps', 'codecontests',
    'open_r1_math', 'mathqa', 'gsm8k', 'aqua_rat', 'strategyqa',
    'sciq', 'stem',
    'arc_easy', 'arc_challenge', 'nq_open',
    'hellaswag','truthfulqa', 'ultrafeedback'
]

evaluation_splits = {
    'sciq': 'train',  # Downloaded train, use train with custom split
    'dolly': 'train',  # Downloaded train, use train with 20% validation
    'math': 'train',  # Downloaded train, use train with custom split
    'stem': 'train',  # Downloaded train, use train with custom split
    'code_search_net': 'train',  # Downloaded train, use train with custom split
    'daily_dialog': 'train',  # Downloaded train, use train with custom split
    'open_assistant': 'train',  # Downloaded train, use train with 20% validation
    'code_alpaca': 'train',  # Downloaded train, use train with 20% validation
    'alpaca': 'train',  # Downloaded train, use train with 20% validation
    'personachat': 'train',  # Downloaded train, use train with custom split
    'mmlu': 'test',  # Evaluation fetches test split online
    'arc_easy': 'test',  # Evaluation fetches test split online
    'arc_challenge': 'test',  # Evaluation fetches test split online
    'gsm8k': 'test',  # Evaluation fetches test split online
    'human_eval': 'test',  # Matches downloaded split
    'mbpp': 'test',  # Evaluation fetches test split online
    'truthfulqa': 'validation',  # Matches downloaded split
    'kjv_bible': 'train',  # Downloaded train, use train with custom split
}

top_models_performance = {
    'mmlu': {
        'GPT-3.5 Turbo': 70.0,
        'GPT-4': 86.4,
        'GPT-4 Turbo': 86.5,
        'GPT-4o': 88.7,
        'GPT-4.1': 90.2,
        'Claude 3 Opus': 86.8,
        'Claude 3.5 Sonnet': 88.7,
        'Gemini 1.5 Pro': 81.9,
        'Gemini 2.5 Pro': 89.8,
        'Grok 3': 92.7
    },
    'arc_easy': {
        'GPT-3.5 Turbo': 85.2,
        'GPT-4': 96.3,
        'GPT-4o': 95.0,
        'Claude 3.5 Sonnet': 94.0,
        'Gemini 1.5 Pro': 93.5
    },
    'arc_challenge': {
        'GPT-3.5 Turbo': 85.2,
        'GPT-4': 96.3,
        'GPT-4o': 94.8,
        'Claude 3.5 Sonnet': 93.0,
        'Gemini 1.5 Pro': 92.0
    },
    'gsm8k': {
        'GPT-3.5 Turbo': 57.1,
        'GPT-4': 92.0,
        'GPT-4o': 94.0,
        'Claude 3 Opus': 95.0,
        'Claude 3.5 Sonnet': 96.4,
        'Gemini 1.5 Pro': 91.7,
        'Grok 3': 89.3
    },
    'human_eval': {
        'GPT-3.5 Turbo': 48.1,
        'GPT-4': 67.0,
        'GPT-4o': 90.2,
        'Claude 3 Opus': 84.9,
        'Claude 3.5 Sonnet': 92.0,
        'Gemini 1.5 Pro': 71.9,
        'Grok 3': 86.5
    },
    'mbpp': {
        'GPT-4': 75.0,
        'GPT-4o': 80.0,
        'Claude 3.5 Sonnet': 78.0,
        'Gemini 1.5 Pro': 76.0
    },
    'truthfulqa': {
        'GPT-3.5 Turbo': 47.0,
        'GPT-4': 59.0,
        'GPT-4o': 60.0,
        'Claude 3.5 Sonnet': 62.0,
        'Gemini 1.5 Pro': 58.0
    }
}

def evaluate_dataset(model, eval_data, dataset_name):
    """
    Evaluate the model on a given dataset and return the performance metric.
    
    Args:
        model: HSMN model instance.
        eval_data: Preprocessed evaluation data.
        dataset_name: Name of the dataset (e.g., 'mmlu').
    
    Returns:
        Float representing the performance metric (e.g., accuracy).
    """
    logging.info(f"Evaluating on {dataset_name}")
    if dataset_name in ['mmlu', 'arc_easy', 'arc_challenge', 'gsm8k']:
        correct = 0
        total = 0
        for item in eval_data:
            context = item['context']
            target = item['target']
            try:
                prediction = model.generate(context, task='chat')
                if prediction.strip().lower() == target.strip().lower():
                    correct += 1
                total += 1
            except Exception as e:
                logging.error(f"Error evaluating item in {dataset_name}: {e}")
                continue
        accuracy = correct / total if total > 0 else 0
        return accuracy
    else:
        # Placeholder for datasets like human_eval, mbpp, truthfulqa
        logging.warning(f"Evaluation for {dataset_name} not implemented")
        return 0.0

def train_cumulative(dataset_configs, vocab_size, max_seq_len=512, epochs=200, batch_size=1, accum_steps=16, model_path="hsmn_model"):
    """
    Train a single HSMN model cumulatively across multiple datasets with EWC.
    
    Args:
        dataset_configs: List of dictionaries containing dataset metadata.
        vocab_size: Size of the vocabulary for the model.
        max_seq_len: Maximum sequence length (default: 512).
        epochs: Number of epochs per dataset (default: 10).
        batch_size: Batch size (default: 1).
        accum_steps: Number of steps for gradient accumulation (default: 8).
        model_path: Base path for saving model weights (default: "hsmn_model").
    
    Returns:
        Trained HSMN model.
    """
    # Initialize the model once
    model = HSMN(vocab_size=vocab_size, max_seq_len=max_seq_len, tokenizer=tokenizer)
    logging.info("Initialized HSMN model for cumulative training")

    # Filter and order dataset_configs based on benchmark_datasets
    ordered_dataset_configs = [config for config in dataset_configs if config['name'] in benchmark_datasets]
    ordered_dataset_configs = sorted(ordered_dataset_configs, key=lambda x: benchmark_datasets.index(x['name']))
    logging.info(f"Training on datasets: {[config['name'] for config in ordered_dataset_configs]}")

    # Initialize EWC storage
    previous_fishers = []
    previous_optimal_weights = []

    # Train sequentially on each dataset
    for config in ordered_dataset_configs:
        dataset_name = config["name"]
        dataset_path = f"datasets/{dataset_name}"
        
        # Clear session to manage memory between datasets
        tf.keras.backend.clear_session()
        gc.collect()
        
        try:
            dataset = load_from_disk(dataset_path)
            print(f"Loaded {dataset_name} with {len(dataset)} items")
            logging.info(f"Training on {dataset_name} with {len(dataset)} items")
            model, previous_fishers, previous_optimal_weights = train_on_dataset(
                model, dataset_name, dataset, epochs=epochs, batch_size=batch_size, 
                accum_steps=accum_steps, model_path=model_path, 
                previous_fishers=previous_fishers, previous_optimal_weights=previous_optimal_weights
            )
        except Exception as e:
            logging.error(f"Failed to load/train {dataset_name}: {e}")
            print(f"Failed to process {dataset_name}: {e}")
            continue
    
    # Evaluation
    scorecard = {}
    for dataset_name in benchmark_datasets:
        config = next((c for c in dataset_configs if c['name'] == dataset_name), None)
        if config is None:
            logging.warning(f"No config found for {dataset_name}")
            continue
        eval_split = evaluation_splits.get(dataset_name)
        if eval_split is None:
            logging.warning(f"No evaluation split defined for {dataset_name}")
            continue
        try:
            dataset = load_dataset(config['repo'], config.get('config'), split=eval_split)
            eval_data = PREPROCESSORS[dataset_name](dataset)
        except Exception as e:
            logging.error(f"Failed to load or preprocess evaluation data for {dataset_name}: {e}")
            continue
        hsmn_score = evaluate_dataset(model, eval_data, dataset_name)
        top_scores = top_models_performance.get(dataset_name, {})
        scorecard[dataset_name] = {'HSMN': hsmn_score, **top_scores}

    # Print scorecard
    print("Scorecard:")
    for dataset, scores in scorecard.items():
        print(f"{dataset}: HSMN: {scores['HSMN']}, Top models: {scores}")

    # Save scorecard to file
    with open('scorecard.txt', 'w') as f:
        for dataset, scores in scorecard.items():
            f.write(f"{dataset}: HSMN: {scores['HSMN']}, Top models: {scores}\n")

    # Save the final cumulative model
    final_weights_path = "hsmn_model_final.h5"
    if model is not None:
        try:
            save_pruned_weights(model, final_weights_path)
            print(f"Final cumulative model saved to '{final_weights_path}'")
            logging.info(f"Saved final cumulative model to {final_weights_path}")
        except Exception as e:
            logging.error(f"Failed to save final model: {e}")
            print(f"Failed to save final model: {e}")
    
    return model

def test_validation_loop(model, tokenizer, dataset_name, val_gen, max_samples=10):
    """
    Test the validation loop with a small number of samples to verify model behavior.
    """
    logging.info(f"Testing validation loop for {dataset_name} with up to {max_samples} samples")
    
    # Build the model with dummy data
    logging.info("Building model with dummy data for test_validation_loop...")
    dummy_context_tokens = tf.constant([[tokenizer.cls_token_id]], dtype=tf.int32)
    dummy_target_tokens = tf.constant(
        [[tokenizer.cls_token_id] + [tokenizer.pad_token_id] * (model.max_seq_len - 1)], 
        dtype=tf.int32
    )
    _ = model((dummy_context_tokens, dummy_target_tokens), training=False)
    
    # Initialize pruning steps
    pruned_layers = get_pruned_layers(model)
    logging.info(f"Found {len(pruned_layers)} pruned layers for validation test")
    sync_pruning_step(model, pruned_layers)
    
    # Initialize loss function
    loss_fn_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    # Process validation samples
    val_loss_sum = 0.0
    val_batches = 0
    for i, item in enumerate(val_gen):
        if i >= max_samples:
            break
        try:
            # Tokenize context and target
            context = item['context']
            target = item['target']
            context_tokens = tokenizer(
                context,
                add_special_tokens=False,
                truncation=True,
                max_length=model.max_seq_len
            )['input_ids']
            target_tokens = tokenizer(
                target,
                add_special_tokens=True,
                truncation=True,
                max_length=model.max_seq_len,
                padding='max_length'
            )['input_ids']
            context_tokens_tf = tf.constant([context_tokens], dtype=tf.int32)
            target_tokens_tf = tf.constant([target_tokens], dtype=tf.int32)
            
            logging.info(f"Validation test sample {i+1}: context_tokens shape={context_tokens_tf.shape}, "
                         f"target_tokens shape={target_tokens_tf.shape}")
            
            # Compute logits
            logits = model((context_tokens_tf, target_tokens_tf), training=False)
            logging.info(f"Validation test sample {i+1}: logits shape={logits.shape}")
            
            # Verify logits shape
            assert len(logits.shape) == 3, f"Expected 3D logits, got {logits.shape}"
            assert logits.shape[0] == 1, "Batch size should be 1"
            assert logits.shape[1] == model.max_seq_len, "Sequence length mismatch"
            assert logits.shape[2] == model.vocab_size, "Vocab size mismatch"
            
            # Compute loss
            labels = target_tokens_tf[:, 1:]
            logits_for_loss = logits[:, :-1, :]
            loss_per_token = loss_fn_sparse(labels, logits_for_loss)
            mask = tf.cast(labels != tokenizer.pad_token_id, tf.float32)
            sum_mask = tf.reduce_sum(mask)
            loss = tf.constant(0.0) if tf.equal(sum_mask, 0.0) else tf.reduce_sum(loss_per_token * mask) / sum_mask
            
            if tf.math.is_finite(loss):
                val_loss_sum += loss.numpy()
                val_batches += 1
                logging.info(f"Validation test sample {i+1}: Loss={loss.numpy():.4f}")
            else:
                logging.warning(f"Validation test sample {i+1}: Non-finite loss, skipping")
        except Exception as e:
            logging.error(f"Error in validation test sample {i+1}: {str(e)}\n{traceback.format_exc()}")
            continue
        finally:
            gc.collect()
    
    if val_batches > 0:
        avg_val_loss = val_loss_sum / val_batches
        logging.info(f"Validation test completed: Average Loss={avg_val_loss:.4f} over {val_batches} samples")
    else:
        logging.warning("No valid samples processed in validation test")
        raise ValueError("Validation test failed: No valid samples processed")
    
    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    dataset_configs = [
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
        {"name": "hellaswag", "repo": "Rowan/hellaswag", "split": "train"},
        {"name": "apps", "repo": "codeparrot/apps", "split": "train", "trust_remote_code": True},
        {"name": "mathqa", "repo": "allenai/math_qa", "split": "train", "trust_remote_code": True},
        {"name": "strategyqa", "repo": "tau/strategyqa", "split": "train"},
        {"name": "codecontests", "repo": "deepmind/code_contests", "split": "train"},
        {"name": "nq_open", "repo": "google-research-datasets/nq_open", "split": "train", "trust_remote_code": True},
        {"name": "aqua_rat", "repo": "deepmind/aqua_rat", "split": "train"},
        {"name": "ultrafeedback", "repo": "HuggingFaceH4/ultrafeedback_binarized", "split": "train_sft"},
    ]

    # Train the model cumulatively
    trained_model = train_cumulative(dataset_configs, vocab_size=VOCAB_SIZE)