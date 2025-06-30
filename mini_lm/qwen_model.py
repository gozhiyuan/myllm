import math
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig,
)
import gc
import os
import sys  # For checking Python version


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Convert back to original dtype
        hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states


def get_rotary_cos_sin(seq_len, dim, device, base=10000.0):
    """Generate rotary embeddings as used in Qwen2."""
    # Generate frequency tensor
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Create positions tensor
    t = torch.arange(seq_len, device=device).float()

    # Compute frequencies for positions
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    # Compute cos and sin embeddings
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # Add batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]

    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys."""
    # q, k have shape (batch_size, num_heads, seq_len, head_dim)
    # cos, sin have shape (1, 1, seq_len, head_dim)

    # Make a copy of q and k
    q_embed = q.clone()
    k_embed = k.clone()

    # Get dimensions
    dim = q.shape[-1]

    # Split embeddings into real and imaginary parts (first and second half)
    q_1, q_2 = q[..., : dim // 2], q[..., dim // 2 :]
    k_1, k_2 = k[..., : dim // 2], k[..., dim // 2 :]

    # Truncate cos/sin to match sequence length
    cos_q = cos[:, :, : q.shape[2], :]
    sin_q = sin[:, :, : q.shape[2], :]
    cos_k = cos[:, :, : k.shape[2], :]
    sin_k = sin[:, :, : k.shape[2], :]

    # Apply rotary embeddings
    q_embed[..., : dim // 2] = q_1 * cos_q - q_2 * sin_q
    q_embed[..., dim // 2 :] = q_2 * cos_q + q_1 * sin_q
    k_embed[..., : dim // 2] = k_1 * cos_k - k_2 * sin_k
    k_embed[..., dim // 2 :] = k_2 * cos_k + k_1 * sin_k

    return q_embed, k_embed


class QwenAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding.
    Exactly matching Qwen2's implementation to load weights directly.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # Exact weight dimensions from Qwen2
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(getattr(config, "attention_dropout", 0.0))

        # Register causal mask
        mask = torch.full(
            (1, 1, self.max_position_embeddings, self.max_position_embeddings),
            float("-inf"),
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self, hidden_states, past_key_value=None, use_cache=False, attention_mask=None
    ):
        batch_size, seq_length, hidden_size = hidden_states.shape

        # Compute QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = seq_length

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            # Concatenate past key-values
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            # Save key and value for future use
            past_key_value = (key_states, value_states)

        # Apply rotary embeddings
        cos, sin = get_rotary_cos_sin(kv_seq_len, self.head_dim, hidden_states.device)
        cos = cos.to(dtype=hidden_states.dtype)
        sin = sin.to(dtype=hidden_states.dtype)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Compute attention
        # Dot product attention (q @ k.T)
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Apply causal mask
        attention_mask_causal = self.mask[:, :, :seq_length, :kv_seq_len]
        attention_scores = attention_scores + attention_mask_causal

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax and dropout
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # Compute context (attn @ v)
        context = torch.matmul(attention_probs, value_states)

        # Reshape output
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_length, self.hidden_size)

        # Apply output projection
        output = self.o_proj(context)

        return (output, past_key_value) if use_cache else output


class QwenMLP(nn.Module):
    """MLP layer for Qwen2."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Standard MLP projections as in Qwen2
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU activation pattern
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QwenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

        self.use_parallel_residual = getattr(config, "use_parallel_residual", True)

    def forward(
        self, hidden_states, past_key_value=None, use_cache=False, attention_mask=None
    ):
        residual = hidden_states

        # Layer Norm before self-attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with layer norm as per Qwen2
        if use_cache:
            attn_output, present_key_value = self.self_attn(
                hidden_states,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
        else:
            attn_output = self.self_attn(hidden_states, attention_mask=attention_mask)
            present_key_value = None

        # Parallel or sequential residual
        if self.use_parallel_residual:
            # Layer Norm for MLP branch
            mlp_hidden_states = self.post_attention_layernorm(hidden_states)
            # MLP
            mlp_output = self.mlp(mlp_hidden_states)
            # Combine both branches
            hidden_states = residual + attn_output + mlp_output
        else:
            # Sequential residual
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)

        if use_cache:
            return hidden_states, present_key_value
        else:
            return hidden_states


class MiniLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QwenBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)

        # Initialize buffers for KV cache
        self.kv_cache_initialized = False

    def _init_kv_cache(self, batch_size, max_seq_len, device):
        """Initialize the kv cache for incremental decoding."""
        if self.kv_cache_initialized:
            return

        head_dim = self.config.hidden_size // self.config.num_heads
        self.past_key_values = tuple(
            (
                torch.zeros(
                    (
                        batch_size,
                        self.config.num_key_value_heads,
                        max_seq_len,
                        head_dim,
                    ),
                    device=device,
                    dtype=self.embed_tokens.weight.dtype,
                ),
                torch.zeros(
                    (
                        batch_size,
                        self.config.num_key_value_heads,
                        max_seq_len,
                        head_dim,
                    ),
                    device=device,
                    dtype=self.embed_tokens.weight.dtype,
                ),
            )
            for _ in range(self.config.num_hidden_layers)
        )
        self.kv_cache_initialized = True

    def forward(
        self, input_ids, past_key_values=None, use_cache=False, attention_mask=None
    ):
        batch_size, seq_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Initialize past_key_values if needed
        if past_key_values is None and use_cache:
            past_key_values = [None] * len(self.layers)

        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if use_cache:
                hidden_states, present_key_value = layer(
                    hidden_states,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
                present_key_values.append(present_key_value)
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)

        outputs = {
            "last_hidden_state": hidden_states,
        }

        if use_cache:
            outputs["past_key_values"] = present_key_values

        return outputs


class MiniLLMForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MiniLLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        """Tie the weights between the input and output embeddings."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids,
        past_key_values=None,
        labels=None,
        use_cache=False,
        attention_mask=None,
        **kwargs,
    ):
        # Get model outputs
        model_outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )

        hidden_states = model_outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        outputs = {
            "loss": loss,
            "logits": logits,
        }

        if use_cache:
            outputs["past_key_values"] = model_outputs["past_key_values"]

        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """Prepare inputs for generation."""
        # Only the last token for inputs_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # If attention mask is provided, extend it
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        max_length=20,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        eos_token_id=None,
        pad_token_id=None,
        attention_mask=None,
    ):
        """Generation with more efficient KV caching and sampling."""
        bsz, input_len = input_ids.size()
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Track generated ids
        output_ids = input_ids.clone()
        past_key_values = None
        cur_len = input_len

        # Keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            bsz, dtype=torch.long, device=input_ids.device
        )

        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                output_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

            # Forward pass
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :].float()
            past_key_values = outputs["past_key_values"]

            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-p (nucleus) sampling
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                # Get sorted logits and indices
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1),
                    dim=-1,
                )
                # Remove tokens with probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Create a sparse mask from sorted indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, -float("inf")
                )

                # Sample from filtered distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Finished sequences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

            # Update output_ids
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((bsz, 1), device=attention_mask.device),
                    ],
                    dim=-1,
                )

            # Update which sequences are finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences & (
                    next_tokens != eos_token_id
                )

            # Stop when all sequences are finished
            if unfinished_sequences.max() == 0:
                break

            cur_len += 1

        return output_ids

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class MiniLLMConfig:
    def __init__(self, **kwargs):
        # Qwen2-0.5B-Instruct configuration with defaults
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.hidden_size = kwargs.get("hidden_size", 896)
        self.intermediate_size = kwargs.get("intermediate_size", 4864)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.num_attention_heads = kwargs.get("num_attention_heads", 16)
        self.num_heads = kwargs.get("num_heads", 16)  # For compatibility
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 16)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-5)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # Dropout settings
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.hidden_dropout = kwargs.get("hidden_dropout", 0.0)

        # Token IDs
        self.pad_token_id = kwargs.get("pad_token_id", 151643)
        self.eos_token_id = kwargs.get("eos_token_id", 151643)
        self.bos_token_id = kwargs.get("bos_token_id", 151643)

        # Rotary embeddings
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rotary_pct = kwargs.get("rotary_pct", 1.0)
        self.rotary_emb_base = kwargs.get("rotary_emb_base", 10000.0)

        # Architecture specifics
        self.use_parallel_residual = kwargs.get("use_parallel_residual", True)


def load_and_setup_models_for_comparison(model_name="Qwen/Qwen2-0.5B-Instruct"):
    """Loads the official model efficiently and attempts to load weights into the custom model."""

    # Load tokenizer first
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine the target device
    if torch.cuda.is_available():
        target_device = torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        target_device = torch.device("mps")  # For Apple Silicon
    else:
        target_device = torch.device("cpu")
    print(f"Target device: {target_device}")

    official_model = None  # Initialize to None
    custom_model = None  # Initialize to None

    # 1. Load the official model using accelerate for memory efficiency
    print("\nLoading official model with accelerate optimizations...")
    os.makedirs("accelerate_offload", exist_ok=True)
    try:
        official_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Load in half precision
            low_cpu_mem_usage=True,
            device_map="auto",
            offload_folder="accelerate_offload",
        )
        print("Official model loaded successfully.")
        official_model.eval()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load the official model: {e}")
        # No need to clean up official_model here as it failed to assign
        raise  # Re-raise the original error

    # 2. Attempt to initialize the custom model on CPU
    print("\nAttempting to initialize custom MiniLLM model on CPU...")
    try:
        config = official_model.config  # Get config from the loaded official model
        custom_config = MiniLLMConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            use_parallel_residual=getattr(config, "use_parallel_residual", True),
        )
        custom_model = MiniLLMForCausalLM(custom_config).to(
            dtype=torch.float16
        )  # Init on CPU, float16
        custom_model.eval()
        print("Custom model initialized successfully on CPU.")
    except Exception as e:
        print(f"ERROR: Failed to initialize custom model structure on CPU: {e}")
        print("Cannot proceed with weight copy and comparison.")
        # Clean up official model (which must have loaded successfully to reach here)
        if official_model is not None:
            del official_model
        gc.collect()
        torch.mps.empty_cache() if hasattr(torch, "mps") else None
        raise RuntimeError(
            "Failed to initialize custom model due to memory or other error."
        ) from e

    # 3. Copy weights (Layer-by-Layer) from official to custom (CPU)
    print("\nTransferring weights to custom model (on CPU)...")
    try:
        with torch.no_grad():
            # Embeddings
            custom_model.model.embed_tokens.weight.copy_(
                official_model.model.embed_tokens.weight.cpu()
            )
            # Transformer Blocks
            for i in range(custom_config.num_hidden_layers):
                print(
                    f"  Copying layer {i+1}/{custom_config.num_hidden_layers}...",
                    end="\r",
                )
                custom_model.model.layers[i].self_attn.q_proj.weight.copy_(
                    official_model.model.layers[i].self_attn.q_proj.weight.cpu()
                )
                custom_model.model.layers[i].self_attn.k_proj.weight.copy_(
                    official_model.model.layers[i].self_attn.k_proj.weight.cpu()
                )
                custom_model.model.layers[i].self_attn.v_proj.weight.copy_(
                    official_model.model.layers[i].self_attn.v_proj.weight.cpu()
                )
                custom_model.model.layers[i].self_attn.o_proj.weight.copy_(
                    official_model.model.layers[i].self_attn.o_proj.weight.cpu()
                )
                custom_model.model.layers[i].mlp.gate_proj.weight.copy_(
                    official_model.model.layers[i].mlp.gate_proj.weight.cpu()
                )
                custom_model.model.layers[i].mlp.up_proj.weight.copy_(
                    official_model.model.layers[i].mlp.up_proj.weight.cpu()
                )
                custom_model.model.layers[i].mlp.down_proj.weight.copy_(
                    official_model.model.layers[i].mlp.down_proj.weight.cpu()
                )
                custom_model.model.layers[i].input_layernorm.weight.copy_(
                    official_model.model.layers[i].input_layernorm.weight.cpu()
                )
                custom_model.model.layers[i].post_attention_layernorm.weight.copy_(
                    official_model.model.layers[i].post_attention_layernorm.weight.cpu()
                )
            print("\n  Layer copies finished.")
            # Final LayerNorm
            custom_model.model.norm.weight.copy_(official_model.model.norm.weight.cpu())
            # LM Head
            if not custom_config.tie_word_embeddings:
                custom_model.lm_head.weight.copy_(official_model.lm_head.weight.cpu())
            else:
                custom_model.tie_weights()
        print("Weight transfer complete (on CPU).")
    except Exception as e:
        print(f"ERROR: Failed during weight transfer: {e}")
        if official_model is not None:
            del official_model
        if custom_model is not None:
            del custom_model
        gc.collect()
        torch.mps.empty_cache() if hasattr(torch, "mps") else None
        raise RuntimeError("Failed during weight transfer.") from e

    # 4. Move custom model to target device
    print(f"\nMoving custom model from CPU to target device: {target_device}...")
    try:
        custom_model = custom_model.to(target_device)
        print("Custom model moved to target device successfully.")
    except Exception as e:
        print(f"ERROR: Failed to move custom model to target device: {e}")
        if official_model is not None:
            del official_model  # Official model still exists here
        if custom_model is not None:
            del custom_model  # Custom model exists but failed to move
        gc.collect()
        torch.mps.empty_cache() if hasattr(torch, "mps") else None
        raise RuntimeError("Failed moving custom model to target device.") from e

    print("\nSetup for comparison complete.")
    # Return both models, tokenizer, and the target device
    return official_model, custom_model, tokenizer, target_device


# Modified test generation function to match Qwen's generation style
def test_generation(
    model,
    tokenizer,
    device,
    prompt,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
):
    """Generate text with minimal memory usage."""

    # Format prompt using the chat template
    print(f"Prompt: {prompt}")

    # Tokenize input with minimal memory usage
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Free memory after tokenization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate with careful memory management
    with torch.no_grad():
        try:
            # Use shorter sequence length and conservative memory settings
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,  # Enable KV caching for efficiency
            )

            # Free GPU memory immediately after generation
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Remove input from output
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text) :]

            # Clean up tensors
            del input_ids, output_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return generated_text.strip()

        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback

            traceback.print_exc()

            # Clean up even on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return f"Error generating text: {str(e)}"


# Test with memory optimization
import gc
import os
import torch
import sys  # For checking Python version

# Check Python version for MPS compatibility
if sys.version_info >= (3, 13):
    print(
        "Warning: MPS backend might have issues with Python 3.13+. Consider using Python 3.11 or 3.12."
    )

# Set low memory environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# Optional: If MPS still causes issues, uncomment the line below
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Clean up before starting
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
if hasattr(torch, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Model configuration
model_name = "Qwen/Qwen2-0.5B-Instruct"


def run_test(model, tokenizer, device, prompt):
    """Runs the generation test."""
    print("\nTesting generation...")
    # Tokenize input
    # Note: Inputs might need to be explicitly moved to the model's device if device_map places parts on CPU
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to the primary device reported by the model
    # This is generally recommended when using device_map
    input_device = model.device
    print(f"Moving inputs to device: {input_device}")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,  # Pass inputs dictionary directly
            max_length=inputs["input_ids"].shape[1] + 20,  # Generate 20 new tokens
            do_sample=False,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print("\nTest completed successfully!")


# --- Main Execution ---
official_model = None
custom_model = None
tokenizer = None
device = None

try:
    # --- Load and Prepare Both Models ---
    print(f"--- Loading {model_name} and preparing custom model for comparison ---")
    # This function now attempts to return both models
    official_model, custom_model, tokenizer, device = (
        load_and_setup_models_for_comparison(model_name)
    )

    # --- Run Test on Official Model ---
    print("\n" + "=" * 20 + " TESTING OFFICIAL MODEL " + "=" * 20)
    # Display model info
    print("\nOfficial Model info:")
    print(f"  Model Class: {official_model.__class__.__name__}")
    print(f"  Device Map: {official_model.hf_device_map}")
    if hasattr(official_model, "config"):
        config = official_model.config
        params = sum(p.numel() for p in official_model.parameters() if p.requires_grad)
        print(f"  Approximate total parameters: {params / 1_000_000:.2f}M")
        print(f"  Dtype: {next(official_model.parameters()).dtype}")
    # Run test
    run_test(official_model, tokenizer, device, "Hello, how are you?")
    print("=" * 60)

    # --- Run Test on Custom Model ---
    print("\n" + "=" * 20 + " TESTING CUSTOM MODEL " + "=" * 20)
    # Display model info
    print("\nCustom Model info:")
    if hasattr(custom_model, "config"):
        config = custom_model.config
        params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)
        print(f"  Approximate total parameters: {params / 1_000_000:.2f}M")
        print(f"  Device: {next(custom_model.parameters()).device}")
        print(f"  Dtype: {next(custom_model.parameters()).dtype}")
    # Run test
    run_test(custom_model, tokenizer, device, "Hello, how are you?")
    print("=" * 60)

except torch.mps.OutOfMemoryError:
    print("\nError: MPS out of memory. ")
    print(
        "This likely happened while loading/moving the second (custom) model or during generation."
    )
    print("Try closing other applications using the GPU.")
    import traceback

    traceback.print_exc()
except RuntimeError as e:
    print(f"\nRuntime Error occurred: {e}")
    # Check for specific error messages related to custom model setup
    error_str = str(e).lower()
    if (
        "failed to initialize custom model" in error_str
        or "failed during weight transfer" in error_str
        or "failed moving custom model" in error_str
    ):
        print(
            "Failure occurred during the setup of the custom model, likely due to memory constraints."
        )
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    import traceback

    traceback.print_exc()

# Final Clean up
finally:
    print("\nPerforming final cleanup...")
    # Delete models and tokenizer if they exist
    print("  Deleting official model reference...")
    if "official_model" in locals() and official_model is not None:
        del official_model
    print("  Deleting custom model reference...")
    if "custom_model" in locals() and custom_model is not None:
        del custom_model
    print("  Deleting tokenizer reference...")
    if "tokenizer" in locals() and tokenizer is not None:
        del tokenizer
    print("  Running garbage collection and clearing cache...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Cleanup finished.")
