# smol_model.py: Custom implementation mirroring SmolLM-135M-Instruct

import torch
import torch.nn as nn
import math
import gc
import os
import sys
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, logging

# Suppress verbose logging if desired
# logging.set_verbosity_error()

print(f"PyTorch Version: {torch.__version__}")
print(f"Python Version: {sys.version}")


# --- Model Definitions ---


class SmolLMConfig:
    """Configuration class for the custom SmolLM model, using official values."""

    def __init__(self, official_config=None, **kwargs):
        # Use values directly from the provided table / official config
        self.vocab_size = kwargs.get("vocab_size", 151936)  # Default to Qwen's vocab size
        self.hidden_size = 576
        self.intermediate_size = 1536
        self.num_hidden_layers = 30
        self.num_attention_heads = 9
        self.num_key_value_heads = 3  # GQA confirmed
        self.hidden_act = "silu"
        self.max_position_embeddings = 8192
        self.rms_norm_eps = 1e-05
        self.rope_theta = 100000.0
        self.tie_word_embeddings = True
        self.use_cache = True
        self.use_flashattention = kwargs.get("use_flashattention", False)

        # MOE configuration
        self.use_moe = kwargs.get("use_moe", False)
        self.num_experts = kwargs.get("num_experts", 8)
        self.num_experts_per_token = kwargs.get("num_experts_per_token", 2)
        self.moe_capacity_factor = kwargs.get("moe_capacity_factor", 1.25)
        self.moe_router_aux_loss_weight = kwargs.get("moe_router_aux_loss_weight", 0.01)

        # Tokenizer configuration
        self.tokenizer_path = kwargs.get("tokenizer_path", "tokenizer")  # Path to tokenizer directory
        self.tokenizer_type = kwargs.get("tokenizer_type", "qwen")  # 'qwen' or 'custom'

        # Parameters often set by tokenizer or need defaults if not in config
        # These might be overwritten later in load_models_for_comparison
        self.pad_token_id = kwargs.get("pad_token_id", None)
        self.eos_token_id = kwargs.get("eos_token_id", None)
        self.bos_token_id = kwargs.get("bos_token_id", None)

        # Other common parameters (can be added if needed)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)

        # --- Derived values ---
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_key_value_heads <= 0:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) must be positive"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        print(f"Initialized SmolLMConfig: GQA with {self.num_key_value_groups} groups.")

        if self.use_moe:
            print(
                f"Using MOE with {self.num_experts} experts, {self.num_experts_per_token} experts per token."
            )

        if self.use_flashattention:
            try:
                import flash_attn
                print("Flash Attention is enabled and available.")
            except ImportError:
                print(
                    "Warning: Flash Attention requested but not available. Falling back to standard attention."
                )
                self.use_flashattention = False


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):  # Use eps from config later
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps  # Store config value
        self.variance_epsilon = eps  # Keep both for robustness

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# --- Rotary Embedding Functions --- (Assuming standard RoPE)
def get_rotary_cos_sin(seq_len, dim, device, dtype, base=10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)  # Shape: [seq_len, dim]
    sin = emb.sin().to(dtype)  # Shape: [seq_len, dim]
    return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_cached, sin_cached, position_ids):
    # q, k: [bsz, num_heads, seq_len, head_dim] or [bsz, num_kv_heads, ...]
    # cos_cached, sin_cached: [max_pos_emb, head_dim]
    # position_ids: [bsz, seq_len]

    # Ensure position_ids is on the same device as cached tensors
    position_ids = position_ids.to(cos_cached.device)

    # Gather the sin/cos embeddings based on the position_ids
    # cos_cached shape: [max_pos_emb, head_dim] -> select -> [bsz, seq_len, head_dim]
    cos = cos_cached[position_ids].unsqueeze(1)  # [bsz, 1, seq_len, head_dim]
    sin = sin_cached[position_ids].unsqueeze(1)  # [bsz, 1, seq_len, head_dim]

    # Apply RoPE to query
    # q: [bsz, num_heads, seq_len, head_dim]
    # cos/sin: [bsz, 1, seq_len, head_dim] -> broadcasts across head dim
    q_embed = (q * cos) + (rotate_half(q) * sin)

    # Apply RoPE to key
    # k: [bsz, num_kv_heads, seq_len, head_dim]
    # cos/sin: [bsz, 1, seq_len, head_dim] -> broadcasts across head dim
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# --- Attention Mechanism --- (Assuming MHA + RoPE)
class SmolAttention(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True  # Standard decoder attention
        self.use_flashattention = config.use_flashattention

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Standard K, Q, V, O projections - Assuming no bias based on common trends
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self._init_rope()

    def _init_rope(self):
        # Precompute rotary embedding cos/sin
        # Note: We compute it up to max_position_embeddings but slice during forward
        # This state isn't saved in the state_dict, recalculated on load
        dim = self.head_dim
        cos, sin = get_rotary_cos_sin(
            self.max_position_embeddings,
            dim,
            device="cpu",
            dtype=torch.float32,
            base=self.rope_theta,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # [bsz, 1, q_len, kv_len]
        position_ids: torch.Tensor | None = None,  # [bsz, q_len]
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len, _ = hidden_states.size()

        # Verify position_ids are passed correctly
        if position_ids is None:
            raise ValueError(
                "position_ids cannot be None in SmolAttention forward pass"
            )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Move rope cache to correct device and dtype inside forward
        # cos = self.cos_cached.to(dtype=hidden_states.dtype, device=hidden_states.device)
        # sin = self.sin_cached.to(dtype=hidden_states.dtype, device=hidden_states.device)
        # Use cached tensors directly, apply_rotary_pos_emb handles device/dtype via position_ids
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, self.cos_cached, self.sin_cached, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Handle Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(
                key_states, dim=1, repeats=self.num_key_value_groups
            )
            value_states = torch.repeat_interleave(
                value_states, dim=1, repeats=self.num_key_value_groups
            )

        # Use Flash Attention if configured and available
        if self.use_flashattention:
            try:
                import flash_attn

                # Prepare causal mask for flash attention
                if attention_mask is not None:
                    # Flash Attention requires a specific mask format
                    attention_mask = attention_mask.squeeze(1).squeeze(
                        1
                    )  # [bsz, seq_len]

                    # Create causal mask as expected by flash_attn
                    if (
                        not hasattr(self, "causal_mask")
                        or self.causal_mask.size(0) < q_len
                    ):
                        self.register_buffer(
                            "causal_mask",
                            torch.triu(
                                torch.ones(q_len, kv_seq_len, dtype=torch.bool),
                                diagonal=1,
                            ),
                            persistent=False,
                        )

                    # Combine with padding mask if provided
                    attention_mask = (
                        attention_mask.bool() | self.causal_mask[:q_len, :kv_seq_len]
                    )
                else:
                    # Use only causal mask
                    if (
                        not hasattr(self, "causal_mask")
                        or self.causal_mask.size(0) < q_len
                    ):
                        self.register_buffer(
                            "causal_mask",
                            torch.triu(
                                torch.ones(q_len, kv_seq_len, dtype=torch.bool),
                                diagonal=1,
                            ),
                            persistent=False,
                        )
                    attention_mask = self.causal_mask[:q_len, :kv_seq_len]

                # Prepare for flash attention
                q = query_states.transpose(1, 2)  # [bsz, seq_len, n_heads, head_dim]
                k = key_states.transpose(1, 2)  # [bsz, seq_len, n_heads, head_dim]
                v = value_states.transpose(1, 2)  # [bsz, seq_len, n_heads, head_dim]

                # Apply flash attention
                attn_output = flash_attn.flash_attn_varlen_qkvpacked(
                    torch.stack([q, k, v], dim=2),
                    ~attention_mask if attention_mask is not None else None,
                    causal=self.is_causal,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                )

                # Reshape output
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            except (ImportError, Exception) as e:
                # Fall back to standard attention if flash_attn is not available or fails
                print(
                    f"Flash Attention failed, falling back to standard attention: {e}"
                )
                self.use_flashattention = False

                # Call standard attention implementation (next block)
                attn_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) / math.sqrt(self.head_dim)

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_weights = self.attn_dropout(attn_weights)

                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        else:
            # Standard attention implementation
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


# --- MLP --- (Explicitly using SwiGLU structure)
class SmolMLP(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Use SiLU activation as specified in the 135M config
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            # Fallback or error if config specifies something else unexpected
            print(
                f"Warning/Error: Unexpected hidden_act '{config.hidden_act}' found, using SiLU as default for SwiGLU-like structure."
            )
            self.act_fn = nn.SiLU()

    def forward(self, x):
        # Explicitly implement SwiGLU activation pattern:
        # gate(x) * up(x), then down_proj
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --- Expert Class (for MOE) ---
class Expert(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.SiLU()

    def forward(self, x):
        # Same SwiGLU pattern as in SmolMLP
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --- MOE Feed Forward (Mixture of Experts based on DeepSeek V2) ---
class MOEFeedForward(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.capacity_factor = config.moe_capacity_factor
        self.aux_loss_weight = config.moe_router_aux_loss_weight

        # Create router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Create experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])

        # Initialize router with near-zero weights for stable training start
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)

        self.aux_loss = torch.tensor(0.0, device="cpu")

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]

        # Router logits and probabilities
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)

        # Get top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )  # Shape: [batch_size * seq_len, num_experts_per_token]

        # Normalize the expert weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Calculate router z-loss (auxiliary loss) for load balancing
        router_z_loss = torch.mean(torch.square(router_logits))
        self.aux_loss = self.aux_loss_weight * router_z_loss.to(self.aux_loss.device)

        # Calculate load balancing loss - ensure all experts get used equally
        # Compute the fraction of tokens routed to each expert
        routing_fraction = router_probs.mean(0)
        # Penalize when the routing distribution is different from uniform
        aux_loss_balance = torch.sum(
            routing_fraction * torch.log(routing_fraction) * self.num_experts
        )
        self.aux_loss += self.aux_loss_weight * aux_loss_balance.to(
            self.aux_loss.device
        )

        # Calculate number of tokens dispatched to each expert
        token_count = x_flat.size(0)
        capacity = int(self.capacity_factor * token_count / self.num_experts)

        # Create output tensor
        final_output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            # Get indices of tokens assigned to this expert
            token_indices = torch.nonzero(expert_mask).squeeze(-1)
            if token_indices.shape[0] == 0:
                continue  # Skip if no tokens assigned to this expert

            # Apply capacity constraint (select top tokens by weight if over capacity)
            if token_indices.shape[0] > capacity:
                # Find the weight for each token to this expert
                token_importance = torch.max(
                    torch.where(
                        expert_indices == expert_idx,
                        expert_weights,
                        torch.zeros_like(expert_weights),
                    ),
                    dim=-1,
                )[0]
                # Sort by importance (weight) and select top tokens within capacity
                top_idx = torch.sort(token_importance[token_indices], descending=True)[
                    1
                ][:capacity]
                token_indices = token_indices[top_idx]

            # Collect the weights and indices
            mask = expert_indices == expert_idx
            # Get expert weights for selected tokens
            selected_weights = torch.where(
                mask, expert_weights, torch.zeros_like(expert_weights)
            )
            flat_weights = selected_weights[token_indices]

            # Get the correct weight for each token (may be in different positions in the topk)
            weight_indices = torch.argmax(mask[token_indices], dim=1)
            token_weights = torch.gather(
                flat_weights, 1, weight_indices.unsqueeze(-1)
            ).squeeze(-1)

            # Process these tokens with the expert
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)

            # Scale outputs by corresponding weights
            scaled_output = expert_output * token_weights.unsqueeze(-1)

            # Add to final output (using index_add for parallel processing)
            final_output.index_add_(0, token_indices, scaled_output)

        # Reshape back to original input shape
        return final_output.reshape(batch_size, seq_len, hidden_dim)


# --- Transformer Block --- (Assuming Pre-LN)
class SmolBlock(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SmolAttention(config=config)

        # Use MOE feed forward if configured, otherwise use standard MLP
        if config.use_moe:
            self.mlp = MOEFeedForward(config)
        else:
            self.mlp = SmolMLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Check state_dict keys for second norm name, assuming post_attention_layernorm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states

        normalized_hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, present_key_value = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_outputs

        # Fully Connected
        residual = hidden_states
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized_hidden_states)

        return hidden_states, present_key_value


# --- Main Model ---
class SmolLMModel(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [SmolBlock(config=config) for _ in range(config.num_hidden_layers)]
        )
        # Check final norm name in state_dict keys, assuming 'norm'
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False  # Can be enabled for training

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # Create causal mask for decoder + handle padding mask
        # Adapted from Hugging Face Llama implementation
        batch_size, seq_length = input_shape
        combined_attention_mask = None
        # Target length takes KV cache length into account
        target_length = seq_length + past_key_values_length

        if seq_length > 0:  # Only need causal mask if new tokens are processed
            # Create causal mask [1, 1, seq_length, target_length]
            # Fill with -inf where q_pos < k_pos
            causal_mask = torch.full(
                (seq_length, target_length),
                fill_value=torch.finfo(inputs_embeds.dtype).min,
                device=inputs_embeds.device,
            )
            # Indices for upper triangle (make sure mask is applied correctly)
            # We want mask[q, k] = -inf if k > q + past_kv_len
            # Simplified: create upper triangle for the *current* query length against *total* key length
            indices = torch.arange(target_length, device=inputs_embeds.device)
            causal_mask = torch.where(
                indices > indices.view(-1, 1)[:seq_length] + past_key_values_length,
                causal_mask,
                0,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                0
            )  # [1, 1, seq_length, target_length]
            combined_attention_mask = causal_mask

        # Incorporate padding mask if provided [bsz, full_seq_len]
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"Attention mask should be 2D [batch_size, seq_length], but got {attention_mask.dim()}D"
                )

            # Expand padding mask to [bsz, 1, q_len, kv_len]
            # Select relevant part of mask if using KV cache (usually full mask is fine)
            expanded_attn_mask = attention_mask[:, None, None, :target_length].expand(
                batch_size, 1, seq_length, target_length
            )
            # Where mask is False (0), add large negative number
            # Convert expanded_attn_mask to boolean for torch.where
            padding_mask = torch.where(
                expanded_attn_mask.bool(), 0.0, torch.finfo(inputs_embeds.dtype).min
            )

            if combined_attention_mask is not None:
                # Merge causal mask and padding mask
                combined_attention_mask = torch.minimum(
                    combined_attention_mask, padding_mask
                )
            else:
                combined_attention_mask = padding_mask

        return combined_attention_mask  # Shape: [bsz, 1, q_len, target_length]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,  # Expected [bsz, seq_len]
        position_ids: torch.Tensor | None = None,  # Expected [bsz, seq_len]
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool | None = None,
    ) -> dict:
        batch_size, seq_length = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Determine past_key_values length
        past_key_values_length = 0
        if (
            past_key_values is not None
            and len(past_key_values) > 0
            and past_key_values[0] is not None
        ):
            past_key_values_length = past_key_values[0][0].shape[
                2
            ]  # Get seq len from K cache

        # Create position_ids if not provided
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).expand(
                batch_size, seq_length
            )  # Expand to batch size
        else:
            position_ids = position_ids.view(batch_size, seq_length).long()

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Prepare the 4D attention mask
        _attention_mask_4d = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=_attention_mask_4d,  # Pass the 4D mask
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache if use_cache else None,
        }


# --- Causal LM Head Model ---
class SmolLMForCausalLM(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.model = SmolLMModel(config)
        self.vocab_size = config.vocab_size
        # Check name in state dict, assuming 'lm_head'
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init() # If using PreTrainedModel style
        if config.tie_word_embeddings:
            self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            output_embeddings.weight = input_embeddings.weight
            print("Tied input and output embedding weights.")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[torch.Tensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,  # Catch extra args
    ) -> dict:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # Cast logits to float32 for stability

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # Add auxiliary loss from MOE if used
            if self.config.use_moe:
                # Collect auxiliary losses from MOE layers
                moe_aux_loss = torch.tensor(0.0, device=loss.device)
                for layer in self.model.layers:
                    if hasattr(layer.mlp, "aux_loss"):
                        moe_aux_loss = moe_aux_loss + layer.mlp.aux_loss.to(loss.device)

                # Add to the main loss
                loss = loss + moe_aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            # Not returning hidden_states/attentions
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # Function used by generate loops (like HF's)

        # Determine sequence length and past length
        input_seq_len = input_ids.shape[-1]
        past_length = 0
        if (
            past_key_values is not None
            and len(past_key_values) > 0
            and past_key_values[0] is not None
        ):
            past_length = past_key_values[0][0].shape[-2]  # Seq len from K cache

        # If KV cache is used, only need the last input token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # Select the last token ID
            input_seq_len = 1  # Now only processing one token

        # --- Position IDs Handling ---
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            # Create position IDs: range starts from past_length
            device = input_ids.device
            position_ids = torch.arange(
                past_length,
                past_length + input_seq_len,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(
                0
            )  # Shape [1, seq_len (or 1 if cache)]
            # No need to expand to batch size here, broadcasting handles it later if needed
        else:
            # If position_ids are provided, ensure they correspond to the last token when using cache
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]  # Select the last position ID

        # --- Attention Mask Handling ---
        # If attention mask is passed, prune it like input_ids for KV cache case
        if attention_mask is not None and past_key_values is not None:
            # This assumes attention_mask is [bsz, full_seq_len]
            # It doesn't *strictly* need pruning for most implementations, but helps clarify
            # attention_mask = attention_mask[:, -1:] # Optional: Prune mask (often not needed)
            pass  # Usually the full mask is handled correctly by the attention mechanism itself

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,  # Pass the potentially full mask
            "position_ids": position_ids,  # Pass the correctly calculated position IDs
        }

    # Basic greedy generate method (can be enhanced)
    @torch.inference_mode()
    def generate(self, input_ids, max_length=50, attention_mask=None, **kwargs):
        # Inherit eos_token_id etc. from config if possible
        eos_token_id = kwargs.get("eos_token_id", self.config.eos_token_id)
        pad_token_id = kwargs.get("pad_token_id", self.config.pad_token_id)
        use_cache = kwargs.get("use_cache", True)
        if pad_token_id is None and eos_token_id is not None:
            print(
                f"Warning: pad_token_id is None, using eos_token_id ({eos_token_id}) for padding during generation"
            )
            pad_token_id = eos_token_id  # Use EOS if PAD is not set
        if pad_token_id is None:
            raise ValueError(
                "pad_token_id must be set for generation if unfinished sequences need padding."
            )

        batch_size, cur_len = input_ids.shape
        # Ensure attention mask covers prompt if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )

        output_ids = input_ids
        past_key_values = None
        position_ids = None  # Let prepare_inputs handle initial creation

        while cur_len < max_length:
            # Prepare inputs using helper method
            model_inputs = self.prepare_inputs_for_generation(
                output_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,  # Pass current position_ids
                use_cache=use_cache,
            )
            # Get updated position_ids for the next iteration
            position_ids = model_inputs["position_ids"]

            # Forward pass
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Greedy decoding
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Handle padding for finished sequences
            # Use `~` to invert the unfinished_sequences mask (0 becomes 1, 1 becomes 0)
            finished_sequences_mask = (
                ~unfinished_sequences.bool()
            )  # Convert to bool then invert
            next_tokens = (
                next_tokens * unfinished_sequences
                + pad_token_id * finished_sequences_mask.long()
            )

            # Append token
            output_ids = torch.cat([output_ids, next_tokens[:, None]], dim=-1)
            cur_len += 1

            # Update attention mask and position_ids for next iteration
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
            )
            # Position for the *next* token will be cur_len (or past_len + 1)
            # `prepare_inputs_for_generation` handles this based on past_key_values
            position_ids = None  # Let prepare_inputs recalculate based on new length

            # Check EOS condition
            if eos_token_id is not None:
                # Use .ne() for clarity
                unfinished_sequences = unfinished_sequences.ne(0) & next_tokens.ne(
                    eos_token_id
                )
                if unfinished_sequences.max() == 0:  # All sequences finished
                    break

        return output_ids


# --- Model Initialization Function ---
def initialize_model(
    model_name="HuggingFaceTB/SmolLM-135M-Instruct",
    use_moe=False,
    num_experts=8,
    tokenizer_path="tokenizer",
    tokenizer_type="qwen",
):
    """Initialize a SmolLM model with or without MOE architecture."""
    print(f"--- Initializing SmolLM model ---")

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    if tokenizer_type == "qwen":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Handle pad_token
    if tokenizer.pad_token is None:
        print(
            "Warning: Tokenizer does not have a pad token. Setting pad_token=eos_token."
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize custom model config
    custom_config = SmolLMConfig(
        use_moe=use_moe,
        num_experts=num_experts,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        vocab_size=len(tokenizer),  # Use tokenizer's vocabulary size
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
    )

    # Create model
    model = SmolLMForCausalLM(custom_config)

    # Move to device
    model = model.to(device)
    print(f"Model initialized on {device}")

    return model, tokenizer, device


# --- Test Function ---
def run_test(model, model_name, tokenizer, device, prompt):
    """Runs generation test and prints output."""
    print(f"\n--- Testing {model_name} ---")
    print(f"Using device: {device}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Input shape: {inputs['input_ids'].shape}")

    # Generate
    print("Generating response...")
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response from {model_name}:")
        print(response)
        print("-" * 40)
        return response
    except Exception as e:
        print(f"ERROR during generation for {model_name}: {e}")
        import traceback

        traceback.print_exc()
        print("-" * 40)
        return f"ERROR: {e}"


# --- Main Execution ---
if __name__ == "__main__":
    # Example of initializing and testing a model with MOE
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    use_moe = True  # Set to True to test MOE architecture
    num_experts = 8

    try:
        # Initialize model
        model, tokenizer, device = initialize_model(model_name, use_moe, num_experts)

        # Run a test
        prompt_text = "Question: What is gravity?\nAnswer:"
        run_test(
            model, f"SmolLM{'_MOE' if use_moe else ''}", tokenizer, device, prompt_text
        )

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        print("\nPerforming cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                if torch.mps.is_initialized():
                    torch.mps.empty_cache()
            except AttributeError:
                pass
        print("Cleanup finished.")
