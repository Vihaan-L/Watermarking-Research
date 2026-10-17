import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import OrderedDict
from dataclasses import dataclass

from torch.nn.modules.module import _IncompatibleKeys
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, LogitsProcessorList, AutoModelForCausalLM, GenerationMixin
from src.rl.configs import GRPOConfig, SFTConfig

logger = logging.getLogger(__name__)

def get_watermarked_model(
    model_args: ModelConfig,
    training_args: SFTConfig | GRPOConfig,
    vocab_size: int
) -> nn.Module:
    """Load frozen LM + trainable watermark head, return wrapped model."""
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    quant_cfg = get_quantization_config(model_args)
    base_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=True,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=not training_args.gradient_checkpointing,
        device_map=get_kbit_device_map() if quant_cfg is not None else None,
        quantization_config=quant_cfg,
    )

    # 1️⃣  frozen language model
    base_lm = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **base_kwargs,
    )
    
    # Check if they match
    if vocab_size != base_lm.vocab_size:
        logger.info("Warning: Vocabulary size mismatch!")
        # Either resize model embeddings to match tokenizer
        base_lm.resize_token_embeddings(vocab_size)

    code_model_cfg = {
        "vocab_size": vocab_size,
        "max_length": 4096
    }
    # 2️⃣  watermark head (trainable)
    watermark_head = CodeModel(code_model_cfg)
    
    if training_args.code_ckpt_path:
        logger.info(f"Loading watermark head from {training_args.code_ckpt_path}")
        final_path = os.path.join(training_args.code_ckpt_path, "pytorch_model.bin")
        watermark_head.load_state_dict(torch.load(final_path))
    
    watermark_cfg = {
        "context_width": training_args.watermark_context_width,
        "gamma": training_args.watermark_gamma,
        "z_threshold": training_args.watermark_z_threshold,
        "switch_threshold": training_args.watermark_switch_threshold,
        "delta": 2.0,
        "watermark_temperature": 1.0,
    }

    # 3️⃣  combined wrapper
    return WatermarkedModel(base_lm, watermark_head, watermark_cfg)



class WatermarkedModel(nn.Module, GenerationMixin):
    """
    Wrapper that keeps the base LM frozen and trains only the watermark head.

    Applies a learned watermark bias to the logits output by the base LM
    across the entire sequence length.
    """
    def __init__(self, base_lm: AutoModelForCausalLM, watermark_model: nn.Module, watermark_cfg: Dict[str, Any]):
        super().__init__()
        self.base_lm = base_lm
        self.watermark_model = watermark_model # This is an instance of CodeModel

        self.context_width = watermark_cfg["context_width"]
        self.gamma = watermark_cfg["gamma"]
        self.z_threshold = watermark_cfg["z_threshold"] # Retained, but not used in current logic
        self.switch_threshold = watermark_cfg["switch_threshold"] # Retained, but not used in current logic
        self.delta = watermark_cfg.get("delta", 2.0) # Bias strength
        self.watermark_temperature = watermark_cfg.get("watermark_temperature", 1.0) # Temperature for softmax in STE

        # ----- freeze base parameters -----
        for p in self.base_lm.parameters():
            p.requires_grad = False
        # ----- ensure watermark head parameters are trainable -----
        for p in self.watermark_model.parameters():
            p.requires_grad = True

        # Track misc flags expected by 🤗 Trainer subclasses
        self.warnings_issued: dict[str, bool] = {}
        # Ensure CodeModel knows its context width if needed internally (it does now)
        self.watermark_model.context_width = self.context_width

    # --------------------------------------------------------------------- #
    @property
    def config(self):
        """
        Expose the underlying LM’s config so external utilities that expect
        ``model.config`` keep working (e.g. Trainer, generation helpers).
        """
        return self.base_lm.config

    @property
    def device(self):
        """ Expose the underlying LM's device """
        return self.base_lm.device

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 2️⃣ Delegate to the frozen base LM.
            base = getattr(self, "base_lm", None)
            try:
                return getattr(base, name)
            except AttributeError:
                raise AttributeError(
                    f"{type(self).__name__} object has no attribute {name!r}"
                )
                
    # ──────────────────────────────────────────────────────────────────────
    def state_dict(self,
                   destination: dict[str, torch.Tensor] | None = None,
                   prefix: str = "",
                   keep_vars: bool = False):
        """
        Return a state‑dict that **only** contains the watermark head.

        The keys are left **un‑prefixed**, e.g. ``ln_f.weight`` instead of
        ``watermark_model.ln_f.weight``. If you prefer the long form,
        change the line that builds *wm_sd*.
        """
        # 1️⃣ grab the head’s own state‑dict
        # Make sure watermark_model is initialized before calling state_dict
        if not hasattr(self, 'watermark_model'):
             raise RuntimeError("Watermark model not initialized yet.")
        wm_sd = self.watermark_model.state_dict(prefix="", keep_vars=keep_vars)

        # 2️⃣ optionally clone into the destination ordered‑dict Hugging Face expects
        if destination is None:
            destination = OrderedDict()
            # Prevent modifying the original state dict if keep_vars is False
            if not keep_vars:
                 wm_sd = OrderedDict((k, v.detach().clone()) for k, v in wm_sd.items())

        destination.update(wm_sd)

        return destination

    def watermark_batch_forward(self, input_ids: torch.Tensor
                                ) -> dict[str, torch.Tensor]:
        """
        Run the learned watermark head on every position *i* of every
        sequence, where the head only sees the rolling context window
        `input_ids[:, max(0, i-C+1):i+1]` (C = context_width).

        Returns
        -------
        {
            "full_logits": Tensor  (B, L, V),
            "full_switch": Tensor  (B, L)            -- raw (un-sigmoid’ed)
        }
        """
        B, L = input_ids.shape
        C     = self.context_width
        dev   = input_ids.device
        pad_id = 0

        # 1️⃣ left-pad so we can take C-sized windows everywhere
        if C > 1:
            left_pad = torch.full((B, C - 1), pad_id,
                                  dtype=input_ids.dtype, device=dev)
            padded   = torch.cat([left_pad, input_ids], dim=1)   # (B, L+C-1)
        else:                                    # C == 1
            padded = input_ids

        # 2️⃣ sliding windows: (B, L, C)
        # unfold creates a *view* -- no data copy
        windows = padded.unfold(dimension=1, size=C, step=1)     # (B, L, C)
        windows = windows.contiguous()                           # keep it safe

        # 3️⃣ flatten batch & time so CodeModel sees (B·L, C)
        flat_windows = windows.view(-1, C)

        # 4️⃣ run watermark head; it returns logits/switch for **last** pos
        wm_out       = self.watermark_model(flat_windows)
        wm_logits    = wm_out["logits"]      # (B·L, V)
        wm_switch    = wm_out["switch"]      # (B·L,)

        # 5️⃣ reshape back to (B, L, ·)
        V = wm_logits.size(-1)
        wm_logits = wm_logits.view(B, L, V)
        wm_switch = wm_switch.view(B, L)

        return {"full_logits": wm_logits,
                "full_switch": wm_switch}


    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids must be provided to WatermarkedModel")
        
        base_out = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        llm_logits = base_out.logits

        # watermark_output = self.watermark_model(input_ids)
        watermark_output = self.watermark_batch_forward(input_ids)
        watermark_logits = watermark_output["full_logits"]
        watermark_switch = watermark_output["full_switch"]

        wm_batch_size, wm_seq_length, wm_vocab_size = watermark_logits.shape
        llm_batch_size, llm_seq_length, llm_vocab_size = llm_logits.shape

        if wm_batch_size != llm_batch_size:
            raise RuntimeError(f"Batch size mismatch: llm_logits ({llm_batch_size}) vs watermark_logits ({wm_batch_size})")
        if wm_vocab_size != llm_vocab_size:
            raise RuntimeError(f"Vocab size mismatch: llm_logits ({llm_vocab_size}) vs watermark_logits ({wm_vocab_size})")

        k = max(1, int(wm_vocab_size * self.gamma))

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(watermark_logits) + 1e-10) + 1e-10)
        noisy_logits = watermark_logits + gumbel_noise

        _, topk_indices = torch.topk(noisy_logits, k, dim=-1)

        hard_mask = torch.zeros_like(noisy_logits)
        hard_mask.scatter_(-1, topk_indices, 1.0)

        soft_probs = torch.softmax(noisy_logits / self.watermark_temperature, dim=-1)

        bias_mask_full = hard_mask - soft_probs.detach() + soft_probs

        raw_switch_logits = watermark_switch.clamp(min=-3.0, max=3.0).unsqueeze(-1)
        switch_prob = torch.sigmoid(raw_switch_logits)
        hard_switch_decision = (switch_prob > self.switch_threshold).float()
        ste_applied_switch = hard_switch_decision - switch_prob.detach() + switch_prob
        bias_mask_full = bias_mask_full * ste_applied_switch

        bias_mask = bias_mask_full

        if llm_seq_length != wm_seq_length:
            if wm_seq_length > llm_seq_length:
                bias_mask = bias_mask[:, -llm_seq_length:, :]
                switch_prob = switch_prob[:, -llm_seq_length:]
            else:
                raise RuntimeError(
                    f"Cannot align sequences: llm_logits sequence length ({llm_seq_length}) "
                    f"is longer than bias_mask sequence length ({wm_seq_length}). Unexpected scenario."
                )

        modified_logits = llm_logits + bias_mask * self.delta

        
        output = WatermarkedCausalLMOutput(
            loss=base_out.loss,
            logits=modified_logits,  # Your modified logits
            past_key_values=base_out.past_key_values,
            hidden_states=base_out.hidden_states,
            attentions=base_out.attentions,
            switch=switch_prob  # Your switch probability
        )
    
        return output

@dataclass
class WatermarkedCausalLMOutput(CausalLMOutputWithPast):
    """Same as CausalLMOutputWithPast but with an additional switch field for watermarking."""
    switch: Optional[torch.FloatTensor] = None


class CodeModel(nn.Module):
    """
    CodeModel: A transformer-based model for code analysis and generation.

    Processes input sequences using self-attention restricted to a rolling
    context window defined by `context_width`.

    The model consists of:
    - Token and position embeddings
    - Transformer encoder layers with windowed attention
    - Output layer for token logits and switch predictions
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Set default configuration values
        default_config = {
            "max_length": 20, # Default, will be overridden by context_width later
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.2,
        }

        # Ensure vocab_size is provided
        if "vocab_size" not in config:
            raise ValueError("vocab_size must be provided in config")

        # Update defaults with provided config
        self.config = default_config.copy()
        self.config.update(config)

        # --- Store context_width ---
        # max_length now effectively means context_width for position embeddings
        self.context_width = self.config["max_length"]
        if self.context_width <= 0:
             raise ValueError("context_width (derived from max_length in config) must be positive.")

        # Create model components
        self.wte = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        # Position embeddings only need to handle the context_width size
        self.wpe = nn.Embedding(self.config["max_length"], self.config["d_model"])
        self.drop = nn.Dropout(self.config["dropout"])

        # Create transformer encoder layer
        # Ensure batch_first=True for easier mask handling and input shapes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
            batch_first=True, # IMPORTANT: batch_first=True
            norm_first=True
        )

        # Create transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config["num_layers"]
        )

        # Output layers
        self.ln_f = nn.LayerNorm(self.config["d_model"])
        # Output dim is vocab_size (for token logits) + 1 (for switch logit)
        self.output = nn.Linear(self.config["d_model"], self.config["vocab_size"] + 1)

        # Initialize weights
        self.apply(self._init_weights)
        logger.info(
            f"Initialized CodeModel with vocab_size: {self.config['vocab_size']}, "
            f"context_width: {self.context_width}"
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use Kaiming initialization for better gradient flow
            # Consider smaller std dev like Transformer default if Kaiming causes issues
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            torch.nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token ids of shape (batch_size, sequence_length)
            
        Returns:
            Dictionary containing:
                - logits: Token logits of shape (batch_size, sequence_length, vocab_size)
                - switch: Switch logits of shape (batch_size, sequence_length)
        """
        device = input_ids.device
        
        # Handle single sequence input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        batch_size, seq_length = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Transform through layers
        hidden_states = self.transformer(src=hidden_states)
        hidden_states = self.ln_f(hidden_states)
        
        # Get outputs
        outputs = self.output(hidden_states)
        
        logits = outputs[..., -1, :-1]
        switch = outputs[..., -1, -1]
        full_logits = outputs[..., :-1]
        full_switch = outputs[..., -1]
            
        return {
            'logits': logits,
            'switch': switch,
            'full_logits': full_logits,
            'full_switch': full_switch
        }
