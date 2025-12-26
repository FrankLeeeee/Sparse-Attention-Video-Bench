try:
    from spas_sage_attn import spas_sage_attn_meansim_cuda
except Exception:
    spas_sage_attn_meansim_cuda = None
    
import os
import warnings
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers.models.transformers.transformer_wan import _get_qkv_projections, _get_added_kv_projections, dispatch_attention_fn

_ALLOW_SPARGE_FALLBACK = os.getenv("SPARSE_BENCH_ALLOW_FALLBACK", "0").lower() in {"1", "true", "yes", "y"}
_WARNED_SPARGE_FALLBACK = False

class WanSpargeAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if spas_sage_attn_meansim_cuda is None:
            msg = (
                "Sparge(Wan) requires the optional CUDA extension `spas_sage_attn`, but it could not be imported (spas_sage_attn_meansim_cuda=None). "
                "Please install/compile the extension first. To run the process anyway, you can set the environment variable "
                "`SPARSE_BENCH_ALLOW_FALLBACK=1` to fall back to standard attention (this does not represent Sparge's performance/results)."
            )
            if not _ALLOW_SPARGE_FALLBACK:
                raise RuntimeError(msg)
            global _WARNED_SPARGE_FALLBACK
            if not _WARNED_SPARGE_FALLBACK:
                warnings.warn(msg, RuntimeWarning)
                _WARNED_SPARGE_FALLBACK = True
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
        else:
            hidden_states = spas_sage_attn_meansim_cuda(
                query,
                key,
                value,
                simthreshd1=0.6,
                cdfthreshd=0.98,
                is_causal=False,
                tensor_layout="NHD",
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states




