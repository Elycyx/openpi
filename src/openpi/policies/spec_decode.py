"""Speculative decoding for Pi0 / Pi0.5 flow-matching policies.

Ported from the RLinf reference implementation (``spec.py``) and adapted to
openpi's ``Policy`` / ``BaseModel`` abstractions. Supports both JAX (``Pi0``)
and PyTorch (``PI0Pytorch``) models.

Speculative decoding at eval time works as follows:

  1. Sample ``spec_batch_size`` parallel action drafts from distinct noises in a
     single batched forward pass.
  2. Compute a cross-sample agreement metric (confidence) over the cumulative
     action deltas; pick the "best" draft.
  3. Run 1-2 verification passes that clamp accepted positions (via noise
     injection along the flow-matching forward process) and check whether the
     remaining positions re-produce values close to the draft (within a delta
     threshold in the **executed / un-normalized** action space).
  4. Return the accepted prefix, padded / extended with the best draft up to
     the full ``action_horizon``.

This mirrors ``spec.py`` one-for-one so parameters can be shared across the two
codebases.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import logging
import os
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SpecConfig (mirrors spec.py's OpenPi0Config spec_* fields exactly)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SpecConfig:
    """Speculative decoding configuration. Field names match ``spec.py``."""

    # master switch
    enable_speculative: bool = False
    # number of parallel drafts sampled from distinct noises
    spec_batch_size: int = 8
    # draft action horizon (None -> model.action_horizon)
    spec_action_horizon: int | None = None
    # diffusion / flow-matching steps used for draft + verify (None -> 10)
    spec_diffusion_num_steps: int | None = None
    # confidence selector hyperparameters
    spec_conf_alpha: float = 0.8
    spec_conf_eps: float = 1e-6
    # per-dim delta threshold in exec space (scalar fallback)
    spec_delta_threshold: float = 0.1
    # optional per-dim thresholds in exec space, takes precedence over scalar
    spec_delta_thresholds: tuple[float, ...] | None = None
    # extra switches
    spec_debug: bool = False
    spec_log_conf_stats: bool = True
    spec_verify_conf: bool = True
    spec_verify_seq: bool = True
    # mirror spec.py's ``action_env_dim``: controls compare_dim heuristic
    action_env_dim: int = 7
    # optional debug log path
    spec_log_path: str | None = None


# ---------------------------------------------------------------------------
# Sampler protocol / framework-specific implementations
# ---------------------------------------------------------------------------
class _SpecSampler(Protocol):
    """A framework-specific sampler usable by :class:`SpeculativeDecoder`.

    The sampler owns the model and knows how to:
      * run the flow-matching sampler with optional ``fixed_actions`` injection;
      * repeat / slice observations along the batch dimension;
      * (optionally) compute and reuse a prefix KV cache across draft / verify.

    All returned action tensors are numpy float32 ``(B, H, D)`` (model / raw
    action space, i.e. before output-transform / unnormalize).
    """

    action_dim: int
    action_horizon: int

    def compute_prefix_cache(self, observation) -> Any: ...  # noqa: D401,E701

    def repeat_observation(self, observation, batch_size: int): ...  # noqa: D401,E701

    def sample(
        self,
        observation,
        *,
        num_steps: int,
        fixed_actions: np.ndarray | None = None,
        fixed_action_mask: np.ndarray | None = None,
        prefix_cache: Any = None,
    ) -> np.ndarray: ...


# ---------------- PyTorch sampler ----------------
class _PytorchSpecSampler:
    def __init__(self, model):
        import torch  # local import; torch is only required for pytorch samplers

        from openpi.models_pytorch.pi0_pytorch import (  # noqa: WPS433
            PI0Pytorch,
            make_att_2d_masks,
        )

        if not isinstance(model, PI0Pytorch):
            raise TypeError(f"Pytorch spec sampler requires PI0Pytorch, got {type(model).__name__}")
        self._torch = torch
        self._make_att_2d_masks = make_att_2d_masks
        self.model = model
        self.action_dim = int(model.config.action_dim)
        self.action_horizon = int(model.config.action_horizon)

    # ---- helpers for caches / tree manipulation ----
    @staticmethod
    def _legacy_cache_from(cache):
        if isinstance(cache, tuple):
            return cache
        if hasattr(cache, "to_legacy_cache"):
            try:
                return cache.to_legacy_cache()
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _legacy_cache_take_first(legacy_cache):
        return tuple((k[:1], v[:1]) for (k, v) in legacy_cache)

    @staticmethod
    def _legacy_cache_repeat(legacy_cache, batch_size):
        return tuple(
            (
                k.repeat_interleave(int(batch_size), dim=0),
                v.repeat_interleave(int(batch_size), dim=0),
            )
            for (k, v) in legacy_cache
        )

    @classmethod
    def _normalize_pkv(cls, pkv):
        if pkv is None:
            return None
        if isinstance(pkv, tuple):
            try:
                from transformers.cache_utils import DynamicCache  # noqa: WPS433

                return DynamicCache.from_legacy_cache(pkv)
            except Exception:  # noqa: BLE001
                return pkv
        return pkv

    @classmethod
    def _take_first(cls, value):
        import torch

        if value is None:
            return None
        legacy = cls._legacy_cache_from(value)
        if legacy is not None:
            first = cls._legacy_cache_take_first(legacy)
            try:
                from transformers.cache_utils import DynamicCache  # noqa: WPS433

                return DynamicCache.from_legacy_cache(first)
            except Exception:  # noqa: BLE001
                return first
        if torch.is_tensor(value):
            return value[:1]
        if isinstance(value, (list, tuple)):
            items = [cls._take_first(v) for v in value]
            return type(value)(items)
        if isinstance(value, dict):
            return {k: cls._take_first(v) for k, v in value.items()}
        return value

    @classmethod
    def _expand(cls, value, batch_size: int):
        import torch

        if value is None:
            return None
        legacy = cls._legacy_cache_from(value)
        if legacy is not None:
            current = int(legacy[0][0].shape[0]) if legacy else None
            if current is None or current == int(batch_size):
                return cls._normalize_pkv(legacy)
            if current == 1:
                repeated = cls._legacy_cache_repeat(legacy, batch_size)
                try:
                    from transformers.cache_utils import DynamicCache  # noqa: WPS433

                    return DynamicCache.from_legacy_cache(repeated)
                except Exception:  # noqa: BLE001
                    return repeated
            raise ValueError(f"Expected cache batch 1 or {batch_size}, got {current}")
        if torch.is_tensor(value):
            if int(value.shape[0]) == int(batch_size):
                return value
            if int(value.shape[0]) == 1:
                return value.repeat(batch_size, *([1] * (value.ndim - 1)))
            raise ValueError(f"Expected batch 1 or {batch_size}, got {int(value.shape[0])}")
        if isinstance(value, (list, tuple)):
            return type(value)(cls._expand(v, batch_size) for v in value)
        if isinstance(value, dict):
            return {k: cls._expand(v, batch_size) for k, v in value.items()}
        return value

    @staticmethod
    def _repeat_tensor(value, batch_size: int):
        import torch

        if value is None:
            return None
        if torch.is_tensor(value):
            if int(value.shape[0]) == int(batch_size):
                return value
            if int(value.shape[0]) == 1:
                return value.repeat(batch_size, *([1] * (value.ndim - 1)))
            raise ValueError(f"Expected batch 1 or {batch_size}, got {int(value.shape[0])}")
        arr = np.asarray(value)
        if int(arr.shape[0]) == int(batch_size):
            return arr
        if int(arr.shape[0]) == 1:
            return np.repeat(arr, batch_size, axis=0)
        raise ValueError(f"Expected batch 1 or {batch_size}, got {int(arr.shape[0])}")

    def repeat_observation(self, observation: _model.Observation, batch_size: int) -> _model.Observation:
        images = {k: self._repeat_tensor(v, batch_size) for k, v in observation.images.items()}
        image_masks = {k: self._repeat_tensor(v, batch_size) for k, v in observation.image_masks.items()}
        return _model.Observation(
            images=images,
            image_masks=image_masks,
            state=self._repeat_tensor(observation.state, batch_size),
            tokenized_prompt=self._repeat_tensor(observation.tokenized_prompt, batch_size),
            tokenized_prompt_mask=self._repeat_tensor(observation.tokenized_prompt_mask, batch_size),
            token_ar_mask=self._repeat_tensor(observation.token_ar_mask, batch_size),
            token_loss_mask=self._repeat_tensor(observation.token_loss_mask, batch_size),
        )

    def _observation_first(self, observation: _model.Observation) -> _model.Observation:
        return _model.Observation(
            images={k: v[:1] for k, v in observation.images.items()},
            image_masks={k: v[:1] for k, v in observation.image_masks.items()},
            state=observation.state[:1],
            tokenized_prompt=self._take_first(observation.tokenized_prompt),
            tokenized_prompt_mask=self._take_first(observation.tokenized_prompt_mask),
            token_ar_mask=self._take_first(observation.token_ar_mask),
            token_loss_mask=self._take_first(observation.token_loss_mask),
        )

    # ---- prefix cache ----
    def compute_prefix_cache(self, observation: _model.Observation):
        obs_single = self._observation_first(observation)
        images, img_masks, lang_tokens, lang_masks, _state = self.model._preprocess_observation(
            obs_single, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = self._make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self._torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        if prefix_att_2d_masks_4d.is_floating_point():
            prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=prefix_embs.dtype)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        past_key_values = self._normalize_pkv(past_key_values)
        return (self._take_first(prefix_pad_masks), self._take_first(past_key_values))

    # ---- sample (with optional fixed action injection) ----
    @property
    def _torch_nograd(self):
        return self._torch.no_grad()

    def sample(
        self,
        observation: _model.Observation,
        *,
        num_steps: int,
        fixed_actions: np.ndarray | None = None,
        fixed_action_mask: np.ndarray | None = None,
        prefix_cache: Any = None,
    ) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            bsize = int(observation.state.shape[0])
            horizon = int(self.model.config.action_horizon)
            action_dim = int(self.model.config.action_dim)
            device = observation.state.device

            noise = self.model.sample_noise((bsize, horizon, action_dim), device)

            if fixed_actions is None:
                fixed_actions_t = torch.zeros_like(noise)
            else:
                fixed_actions_t = torch.as_tensor(fixed_actions, device=device, dtype=noise.dtype)
                if fixed_actions_t.ndim == 2:
                    fixed_actions_t = fixed_actions_t[None].expand(bsize, -1, -1)
                if int(fixed_actions_t.shape[1]) > horizon:
                    fixed_actions_t = fixed_actions_t[:, :horizon]

            if fixed_action_mask is None:
                fixed_action_mask_t = torch.zeros((bsize, horizon), dtype=torch.bool, device=device)
            else:
                fixed_action_mask_t = torch.as_tensor(fixed_action_mask, device=device, dtype=torch.bool)
                if fixed_action_mask_t.ndim == 1:
                    fixed_action_mask_t = fixed_action_mask_t[None].expand(bsize, -1)
                if int(fixed_action_mask_t.shape[1]) > horizon:
                    fixed_action_mask_t = fixed_action_mask_t[:, :horizon]
            if fixed_action_mask_t.ndim == 2:
                fixed_action_mask_t = fixed_action_mask_t[:, :, None]

            images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
                observation, train=False
            )

            prefix_pad_masks = None
            past_key_values = None
            use_prefix_cache = False
            if prefix_cache is not None:
                cached_pad_masks, cached_pkv = prefix_cache
                try:
                    prefix_pad_masks = self._expand(cached_pad_masks, bsize)
                    past_key_values = self._expand(cached_pkv, bsize)
                    past_key_values = self._normalize_pkv(past_key_values)
                    use_prefix_cache = True
                except ValueError:
                    use_prefix_cache = False

            if not use_prefix_cache:
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )
                prefix_att_2d_masks = self._make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
                if prefix_att_2d_masks_4d.is_floating_point():
                    prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=prefix_embs.dtype)
                self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
                self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

                _, past_key_values = self.model.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

            x_t = noise
            timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device)
            timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)

            for idx in range(int(num_steps)):
                t_input = timesteps[idx]
                t_next = timesteps[idx + 1]

                t_exp = t_input.expand(bsize)[:, None, None]
                fixed_x_t = t_exp * noise + (1.0 - t_exp) * fixed_actions_t
                x_t = torch.where(fixed_action_mask_t, fixed_x_t, x_t)

                v_t = self.model.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    t_input.expand(bsize),
                )
                x_t = x_t + dt * v_t

                t_exp_next = t_next.expand(bsize)[:, None, None]
                fixed_x_t_next = t_exp_next * noise + (1.0 - t_exp_next) * fixed_actions_t
                x_t = torch.where(fixed_action_mask_t, fixed_x_t_next, x_t)

            return x_t.detach().float().cpu().numpy()


# ---------------- JAX sampler ----------------
def _jax_prefix_forward(graphdef, state, observation):
    """Pure-JAX prefix forward; jit-compiled and shared across all sample calls."""
    from flax import nnx  # noqa: WPS433

    from openpi.models.pi0 import make_attn_mask  # noqa: WPS433

    model = nnx.merge(graphdef, state)
    obs = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    return prefix_mask, kv_cache


def _jax_sample_inner(
    graphdef,
    state,
    observation,
    noise,
    fixed_actions_j,
    mask_j,
    prefix_mask,
    kv_cache,
    num_steps,
):
    """Pure-JAX Euler loop with optional fixed-action injection.

    ``num_steps`` is treated as a static argument by ``jax.jit`` (closed over the
    surrounding partial), so the whole loop is unrolled into a single GPU kernel.
    """
    import einops  # noqa: WPS433
    from flax import nnx  # noqa: WPS433

    from openpi.models.pi0 import make_attn_mask  # noqa: WPS433

    model = nnx.merge(graphdef, state)
    obs = _model.preprocess_observation(None, observation, train=False)
    bsize = noise.shape[0]
    horizon = noise.shape[1]

    dt = -1.0 / float(num_steps)
    timesteps = jnp.linspace(1.0, 1.0 / num_steps, num_steps)
    timesteps = jnp.concatenate([timesteps, jnp.zeros((1,))], axis=0)

    def body(idx, x_t):
        t_input = timesteps[idx]
        t_next = timesteps[idx + 1]

        # Re-inject fixed positions at the current noise level
        fixed_x_t = t_input * noise + (1.0 - t_input) * fixed_actions_j
        x_t = jnp.where(mask_j, fixed_x_t, x_t)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            obs, x_t, jnp.broadcast_to(t_input, (bsize,))
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_repeat = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_repeat, suffix_attn_mask], axis=-1)
        step_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=step_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -horizon:])
        x_t = x_t + dt * v_t

        # Post-step re-injection at the next noise level
        fixed_x_t_next = t_next * noise + (1.0 - t_next) * fixed_actions_j
        x_t = jnp.where(mask_j, fixed_x_t_next, x_t)
        return x_t

    return jax.lax.fori_loop(0, int(num_steps), body, noise)


class _JaxSpecSampler:
    def __init__(self, model):
        from flax import nnx  # noqa: WPS433

        from openpi.models.pi0 import Pi0  # noqa: WPS433

        if not isinstance(model, Pi0):
            raise TypeError(f"Jax spec sampler requires Pi0, got {type(model).__name__}")
        self.model = model
        self.action_dim = int(model.action_dim)
        self.action_horizon = int(model.action_horizon)
        self._rng = jax.random.key(0)

        # Freeze module state (mirrors nnx_utils.module_jit). The graphdef +
        # state are then closed over by jit-compiled functions so we get a
        # single fused GPU kernel for the whole Euler loop.
        graphdef, state = nnx.split(self.model)
        self._state = state
        self._jit_prefix = jax.jit(functools.partial(_jax_prefix_forward, graphdef))
        # ``num_steps`` becomes static, so different N values trigger separate
        # compilations (rarely changed in practice).
        self._jit_sample = jax.jit(
            functools.partial(_jax_sample_inner, graphdef),
            static_argnames=("num_steps",),
        )

    def _next_rng(self):
        self._rng, rng = jax.random.split(self._rng)
        return rng

    # ---- batch-axis helpers ----------------------------------------------
    # For observation / prefix_mask tensors the batch axis is 0.
    @staticmethod
    def _repeat_axis0(value, batch_size: int):
        if value is None:
            return None
        if hasattr(value, "shape"):
            if int(value.shape[0]) == int(batch_size):
                return value
            if int(value.shape[0]) == 1:
                return jnp.repeat(value, batch_size, axis=0)
            raise ValueError(f"Expected batch 1 or {batch_size}, got {int(value.shape[0])}")
        if isinstance(value, dict):
            return {k: _JaxSpecSampler._repeat_axis0(v, batch_size) for k, v in value.items()}
        return value

    # Backwards-compat alias for external users that may have called
    # ``_repeat_tree`` directly (e.g. test code).
    _repeat_tree = _repeat_axis0

    @staticmethod
    def _slice_axis0_first(value):
        if value is None:
            return None
        if hasattr(value, "shape"):
            return value[:1] if int(value.shape[0]) > 1 else value
        if isinstance(value, dict):
            return {k: _JaxSpecSampler._slice_axis0_first(v) for k, v in value.items()}
        return value

    # For the KV cache the layout is (layers, batch, tokens, kv_heads, head_dim);
    # batch is axis 1. See :class:`openpi.models.gemma.KVCache`.
    @staticmethod
    def _kv_take_first(kv_cache):
        if kv_cache is None:
            return None
        k, v = kv_cache
        if int(k.shape[1]) == 1:
            return (k, v)
        return (k[:, :1], v[:, :1])

    @staticmethod
    def _kv_repeat(kv_cache, batch_size: int):
        if kv_cache is None:
            return None
        k, v = kv_cache
        cur = int(k.shape[1])
        if cur == int(batch_size):
            return kv_cache
        if cur == 1:
            return (jnp.repeat(k, batch_size, axis=1), jnp.repeat(v, batch_size, axis=1))
        raise ValueError(f"Expected kv_cache batch 1 or {batch_size}, got {cur}")

    def repeat_observation(self, observation: _model.Observation, batch_size: int) -> _model.Observation:
        return _model.Observation(
            images={k: self._repeat_axis0(v, batch_size) for k, v in observation.images.items()},
            image_masks={k: self._repeat_axis0(v, batch_size) for k, v in observation.image_masks.items()},
            state=self._repeat_axis0(observation.state, batch_size),
            tokenized_prompt=self._repeat_axis0(observation.tokenized_prompt, batch_size),
            tokenized_prompt_mask=self._repeat_axis0(observation.tokenized_prompt_mask, batch_size),
            token_ar_mask=self._repeat_axis0(observation.token_ar_mask, batch_size),
            token_loss_mask=self._repeat_axis0(observation.token_loss_mask, batch_size),
        )

    def _observation_first(self, observation: _model.Observation) -> _model.Observation:
        return _model.Observation(
            images={k: self._slice_axis0_first(v) for k, v in observation.images.items()},
            image_masks={k: self._slice_axis0_first(v) for k, v in observation.image_masks.items()},
            state=self._slice_axis0_first(observation.state),
            tokenized_prompt=self._slice_axis0_first(observation.tokenized_prompt),
            tokenized_prompt_mask=self._slice_axis0_first(observation.tokenized_prompt_mask),
            token_ar_mask=self._slice_axis0_first(observation.token_ar_mask),
            token_loss_mask=self._slice_axis0_first(observation.token_loss_mask),
        )

    # ---- prefix cache (aligned with PyTorch sampler) ---------------------
    def compute_prefix_cache(self, observation: _model.Observation) -> Any:
        """Run the prefix (vision + language) forward pass once at ``bsize=1``.

        Returns ``(prefix_mask, kv_cache)`` where ``prefix_mask`` has batch axis 0
        and ``kv_cache = (cache_k, cache_v)`` with batch axis 1 (layout
        ``(L, B, T, K, H)``). Both are ``bsize=1`` and will be broadcast to the
        draft / verify batch size inside :meth:`sample`.
        """
        obs_single = self._observation_first(observation)
        prefix_mask, kv_cache = self._jit_prefix(self._state, obs_single)
        return (self._slice_axis0_first(prefix_mask), self._kv_take_first(kv_cache))

    def sample(
        self,
        observation: _model.Observation,
        *,
        num_steps: int,
        fixed_actions: np.ndarray | None = None,
        fixed_action_mask: np.ndarray | None = None,
        prefix_cache: Any = None,
    ) -> np.ndarray:
        bsize = int(observation.state.shape[0])
        horizon = self.action_horizon
        action_dim = self.action_dim

        rng = self._next_rng()
        noise = jax.random.normal(rng, (bsize, horizon, action_dim))

        if fixed_actions is None:
            fixed_actions_j = jnp.zeros_like(noise)
        else:
            arr = np.asarray(fixed_actions, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.broadcast_to(arr[None, ...], (bsize,) + arr.shape).copy()
            if int(arr.shape[1]) > horizon:
                arr = arr[:, :horizon]
            fixed_actions_j = jnp.asarray(arr)

        if fixed_action_mask is None:
            mask_j = jnp.zeros((bsize, horizon, 1), dtype=jnp.bool_)
        else:
            m = np.asarray(fixed_action_mask)
            if m.ndim == 1:
                m = np.broadcast_to(m[None, ...], (bsize,) + m.shape).copy()
            if int(m.shape[1]) > horizon:
                m = m[:, :horizon]
            if m.ndim == 2:
                m = m[..., None]
            mask_j = jnp.asarray(m.astype(np.bool_))

        # Resolve prefix cache: reuse the supplied (bsize=1) cache when valid;
        # otherwise compute it on the fly (mirrors PyTorch fallback path).
        use_prefix_cache = False
        prefix_mask = None
        kv_cache = None
        if prefix_cache is not None:
            cached_prefix_mask, cached_kv_cache = prefix_cache
            try:
                prefix_mask = self._repeat_axis0(cached_prefix_mask, bsize)
                kv_cache = self._kv_repeat(cached_kv_cache, bsize)
                use_prefix_cache = True
            except ValueError:
                use_prefix_cache = False

        if not use_prefix_cache:
            prefix_mask, kv_cache = self._jit_prefix(self._state, observation)

        x_t = self._jit_sample(
            self._state,
            observation,
            noise,
            fixed_actions_j,
            mask_j,
            prefix_mask,
            kv_cache,
            num_steps=int(num_steps),
        )
        return np.asarray(x_t).astype(np.float32)


def build_sampler(model, *, is_pytorch: bool) -> _SpecSampler:
    if is_pytorch:
        return _PytorchSpecSampler(model)
    return _JaxSpecSampler(model)


# ---------------------------------------------------------------------------
# Core speculative decoding algorithm (framework-agnostic, numpy)
# ---------------------------------------------------------------------------
class SpeculativeDecoder:
    """Core spec-decoding algorithm. Delegates sampling to a :class:`_SpecSampler`."""

    def __init__(self, sampler: _SpecSampler, spec_config: SpecConfig):
        self.sampler = sampler
        self.spec_config = spec_config

    # ---------------- public entry ----------------
    def decode(
        self,
        observation,
        *,
        output_transform_sample_fn: Callable[[dict], dict],
    ) -> dict:
        """Run speculative decoding on a single-sample observation (bsize=1).

        ``output_transform_sample_fn`` takes a single-sample dict
        ``{"actions": ndarray (H,D), "state": ndarray (D_state,)}`` and returns
        the corresponding exec-space (i.e. post-transform, un-normalized) dict.

        Returns the full post-transform output dict produced by a final pass of
        ``output_transform_sample_fn`` on the accepted (model-space) chunk. The
        caller **should not** re-apply the output transform afterwards.
        """
        if int(observation.state.shape[0]) != 1:
            raise ValueError(
                f"SpeculativeDecoder expects bsize=1 observation, got {int(observation.state.shape[0])}"
            )

        action_horizon = self._resolve_action_horizon()
        diffusion_num_steps = self._resolve_diffusion_num_steps()
        batch_size = self._resolve_batch_size()
        action_dim = int(self.sampler.action_dim)
        delta_thresholds = self._resolve_delta_thresholds(action_dim)

        chunk_raw, _chunk_exec, info = self._speculative_decode_chunk(
            observation,
            batch_size=batch_size,
            action_horizon=action_horizon,
            diffusion_num_steps=diffusion_num_steps,
            conf_alpha=float(self.spec_config.spec_conf_alpha),
            conf_eps=float(self.spec_config.spec_conf_eps),
            delta_thresholds=delta_thresholds,
            output_len=action_horizon,
            verify_conf_enabled=bool(self.spec_config.spec_verify_conf),
            verify_seq_enabled=bool(self.spec_config.spec_verify_seq),
            spec_debug=bool(self.spec_config.spec_debug),
            output_transform_sample_fn=output_transform_sample_fn,
        )

        logger.info(
            "spec_decode accepted=%d horizon=%d (conf_prefix=%d seq_prefix=%d)",
            int(info.get("accepted_prefix_len", 0)),
            action_horizon,
            int(info.get("accepted_prefix_len_conf", 0)),
            int(info.get("accepted_prefix_len_seq", 0)),
        )
        self._append_log(
            f"spec_decode accepted={int(info.get('accepted_prefix_len', 0))} "
            f"conf={int(info.get('accepted_prefix_len_conf', 0))} "
            f"seq={int(info.get('accepted_prefix_len_seq', 0))} horizon={action_horizon}"
        )

        # Produce the final output dict by running the single-sample output
        # transform on the accepted raw chunk. This preserves all keys (and
        # per-dim post-processing) the transform pipeline normally produces.
        state_np = np.asarray(_as_numpy(observation.state[0]), dtype=np.float32)
        final_out = dict(output_transform_sample_fn({"actions": chunk_raw, "state": state_np}))
        final_out["spec_info"] = info
        return final_out

    # ---------------- param resolution ----------------
    def _resolve_action_horizon(self) -> int:
        h = self.spec_config.spec_action_horizon
        if h is None:
            h = self.sampler.action_horizon
        h = int(h)
        model_h = int(self.sampler.action_horizon)
        if h > model_h:
            raise ValueError(f"spec_action_horizon={h} must be <= model action_horizon={model_h}")
        return h

    def _resolve_diffusion_num_steps(self) -> int:
        n = self.spec_config.spec_diffusion_num_steps
        if n is None:
            n = 10
        n = int(n)
        if n < 1:
            raise ValueError(f"spec_diffusion_num_steps must be >= 1, got {n}")
        return n

    def _resolve_batch_size(self) -> int:
        b = int(self.spec_config.spec_batch_size)
        if b < 1:
            raise ValueError(f"spec_batch_size must be >= 1, got {b}")
        return b

    def _resolve_delta_thresholds(self, action_dim: int) -> np.ndarray:
        value = self.spec_config.spec_delta_thresholds
        if value is None:
            return np.repeat(np.float32(self.spec_config.spec_delta_threshold), 6)
        if isinstance(value, str):
            parts = [p for p in value.replace(",", " ").split() if p]
            arr = np.asarray([float(p) for p in parts], dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)
        if arr.size == 1:
            arr = np.repeat(arr, 6)
        return arr

    def _spec_conf_dim(self, action_dim: int) -> int:
        """Mirrors spec.py: compare dim = min(6, action_env_dim-1) effectively."""
        if action_dim >= 7:
            return 6
        if action_dim > 1:
            return action_dim - 1
        return action_dim

    def _spec_compare_dim(self, pred_dim: int, draft_dim: int) -> int:
        env_dim = int(getattr(self.spec_config, "action_env_dim", min(pred_dim, draft_dim)))
        compare_dim = self._spec_conf_dim(env_dim)
        return max(0, min(int(compare_dim), int(pred_dim), int(draft_dim)))

    # ---------------- logging ----------------
    def _append_log(self, line: str) -> None:
        path = self.spec_config.spec_log_path
        if not path:
            return
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:  # noqa: BLE001
            return

    # ---------------- confidence / selection ----------------
    def _compute_confidence(
        self, actions: np.ndarray, *, alpha: float, eps: float, conf_dim: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        arr = np.asarray(actions)
        if arr.ndim == 2:
            arr = arr[None, ...]
        use_dim = min(conf_dim, arr.shape[-1])
        u = np.cumsum(arr[:, :, :use_dim], axis=1)
        mu = np.mean(u, axis=0)
        var = np.var(u, axis=0)
        d2 = np.sum((u - mu) ** 2 / (var + eps), axis=-1)
        conf_cont = np.exp(-0.5 * d2)
        conf = conf_cont
        tau = np.median(conf, axis=0)
        high = conf >= tau[None, :]
        mu_abs = np.abs(mu)
        conf_stats = {
            "u_abs_mean": float(np.mean(np.abs(u))) if u.size else float("nan"),
            "u_abs_max": float(np.max(np.abs(u))) if u.size else float("nan"),
            "mu_abs_mean": float(np.mean(mu_abs)) if mu_abs.size else float("nan"),
            "mu_abs_max": float(np.max(mu_abs)) if mu_abs.size else float("nan"),
            "var_mean": float(np.mean(var)) if var.size else float("nan"),
            "var_max": float(np.max(var)) if var.size else float("nan"),
            "var_min": float(np.min(var)) if var.size else float("nan"),
            "conf_mean": float(np.mean(conf)) if conf.size else float("nan"),
            "conf_std": float(np.std(conf)) if conf.size else float("nan"),
            "tau_mean": float(np.mean(tau)) if tau.size else float("nan"),
            "alpha": float(alpha),
            "eps": float(eps),
        }
        return conf.astype(np.float32), tau.astype(np.float32), high, conf_stats

    def _select_draft(
        self, actions: np.ndarray, *, alpha: float, eps: float, conf_dim: int
    ) -> dict[str, Any]:
        conf, tau, high, conf_stats = self._compute_confidence(
            actions, alpha=alpha, eps=eps, conf_dim=conf_dim
        )
        count = np.sum(high, axis=1).astype(np.int32)
        sum_conf = np.sum(conf * high, axis=1).astype(np.float32)
        best_count = int(np.max(count))
        candidates = np.flatnonzero(count == best_count)
        best = (
            int(candidates[0])
            if candidates.size == 1
            else int(candidates[np.argmax(sum_conf[candidates])])
        )
        return {
            "conf": conf,
            "tau": tau,
            "high": high,
            "count": count,
            "sum": sum_conf,
            "best": best,
            "conf_stats": conf_stats,
        }

    def _action_match(
        self, pred: np.ndarray, draft: np.ndarray, *, delta_thresholds: np.ndarray
    ) -> bool:
        pred_1d = np.asarray(pred).reshape(-1)
        draft_1d = np.asarray(draft).reshape(-1)
        compare_dim = self._spec_compare_dim(int(pred_1d.shape[-1]), int(draft_1d.shape[-1]))
        if compare_dim == 0:
            return True
        if delta_thresholds.size < compare_dim:
            raise ValueError(
                f"delta_thresholds must have at least {compare_dim} elements, got {delta_thresholds.size}"
            )
        thr = np.asarray(delta_thresholds, dtype=np.float32).reshape(-1)[:compare_dim]
        return bool(np.all(np.abs(pred_1d[:compare_dim] - draft_1d[:compare_dim]) < thr))

    # ---------------- exec-space conversion ----------------
    @staticmethod
    def _apply_transform_batch(
        output_transform_sample_fn: Callable[[dict], dict],
        actions_raw: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Apply ``output_transform_sample_fn`` sample-by-sample to a (B,H,D) array."""
        batch = int(actions_raw.shape[0])
        exec_samples = []
        state_np = np.asarray(state, dtype=np.float32)
        for i in range(batch):
            exec_samples.append(
                output_transform_sample_fn({"actions": actions_raw[i], "state": state_np})["actions"]
            )
        exec_arr = np.stack([np.asarray(s, dtype=np.float32) for s in exec_samples], axis=0)
        return exec_arr

    # ---------------- main spec algorithm ----------------
    def _speculative_decode_chunk(
        self,
        observation,
        *,
        batch_size: int,
        action_horizon: int,
        diffusion_num_steps: int,
        conf_alpha: float,
        conf_eps: float,
        delta_thresholds: np.ndarray,
        output_len: int,
        verify_conf_enabled: bool,
        verify_seq_enabled: bool,
        spec_debug: bool,
        output_transform_sample_fn: Callable[[dict], dict],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        # -------------------- 1) draft --------------------
        prefix_cache = self.sampler.compute_prefix_cache(observation)
        obs_b = self.sampler.repeat_observation(observation, batch_size)

        actions_raw_full = self.sampler.sample(
            obs_b,
            num_steps=int(diffusion_num_steps),
            prefix_cache=prefix_cache,
        )
        state_np = np.asarray(_as_numpy(observation.state[0]), dtype=np.float32)
        actions_full_exec = self._apply_transform_batch(
            output_transform_sample_fn, actions_raw_full, state_np
        )

        conf_enabled = bool(verify_conf_enabled)
        seq_enabled = bool(verify_seq_enabled)
        if not conf_enabled and not seq_enabled:
            raise ValueError("At least one of spec_verify_conf or spec_verify_seq must be True")

        conf_dim = self._spec_conf_dim(int(actions_full_exec.shape[-1]))
        selection = self._select_draft(
            actions_full_exec, alpha=conf_alpha, eps=conf_eps, conf_dim=conf_dim
        )
        conf = np.asarray(selection["conf"])
        conf_stats = selection.get("conf_stats", {})
        log_conf_stats = bool(getattr(self.spec_config, "spec_log_conf_stats", True))
        selected_path = int(selection["best"])
        best_first_idx = selected_path

        b, h, d_exec = actions_full_exec.shape
        d_model = int(actions_raw_full.shape[2])
        greedy_chain_exec = np.zeros((h, d_exec), dtype=actions_full_exec.dtype)
        greedy_chain_exec[0] = actions_full_exec[best_first_idx, 0]
        greedy_chain_raw_full = np.zeros((h, d_model), dtype=actions_raw_full.dtype)
        greedy_chain_raw_full[0] = actions_raw_full[best_first_idx, 0]
        for t in range(1, h):
            greedy_chain_exec[t] = actions_full_exec[selected_path, t]
            greedy_chain_raw_full[t] = actions_raw_full[selected_path, t]

        draft_conf_per_t = conf[selected_path].astype(np.float32)
        pos = np.arange(0, h, dtype=np.int64)
        order_conf = pos[np.lexsort((pos, -draft_conf_per_t[pos]))]
        order_seq = pos

        def _pending(order: np.ndarray, accepted: set[int]) -> np.ndarray:
            if order.size == 0:
                return order
            pending = [int(p) for p in order if int(p) not in accepted]
            if not pending:
                return np.zeros((0,), dtype=np.int64)
            return np.asarray(pending, dtype=np.int64)

        def _prefix_len(accepted: set[int]) -> int:
            length = 1
            for t in range(1, h):
                if t not in accepted:
                    break
                length += 1
            return length

        accepted_positions_conf: set[int] = {0} if conf_enabled else set(range(h))
        accepted_positions_seq: set[int] = {0} if seq_enabled else set(range(h))
        accepted_rank_conf = 0 if conf_enabled else max(0, h - 1)
        accepted_rank_seq = 0 if seq_enabled else max(0, h - 1)
        fail_pos_conf: int | None = None
        fail_action_exec_conf: np.ndarray | None = None
        fail_action_raw_conf: np.ndarray | None = None
        fail_pos_seq: int | None = None
        fail_action_exec_seq: np.ndarray | None = None
        fail_action_raw_seq: np.ndarray | None = None
        conf_active = conf_enabled
        seq_active = seq_enabled

        def _run_verify(
            *,
            kind: str,
            verify_order: np.ndarray,
            verify_actions_exec_slice: np.ndarray,
            verify_actions_raw_slice: np.ndarray,
            accepted: set[int],
            accepted_rank: int,
        ) -> tuple[
            set[int], int, bool, int | None, np.ndarray | None, np.ndarray | None, dict[str, Any] | None
        ]:
            for i in range(int(verify_order.shape[0])):
                p = int(verify_order[i])
                pred_vec = np.asarray(verify_actions_exec_slice[i, p])
                draft_vec = np.asarray(greedy_chain_exec[p])
                if self._action_match(pred_vec, draft_vec, delta_thresholds=delta_thresholds):
                    accepted.add(p)
                    accepted_rank += 1
                    continue

                pred_1d = pred_vec.reshape(-1)
                draft_1d = draft_vec.reshape(-1)
                cmp_dim = self._spec_compare_dim(int(pred_1d.shape[0]), int(draft_1d.shape[0]))
                abs_diff = np.abs(pred_1d[:cmp_dim] - draft_1d[:cmp_dim])
                over = abs_diff >= np.asarray(delta_thresholds[:cmp_dim], dtype=np.float32)
                conf_pos = (
                    float(draft_conf_per_t[p]) if 0 <= p < int(draft_conf_per_t.shape[0]) else float("nan")
                )
                reject_detail = {
                    "kind": kind,
                    "pos": int(p),
                    "rank": int(i),
                    "conf": conf_pos,
                    "abs_diff_max": float(np.max(abs_diff)) if abs_diff.size else float("nan"),
                    "over_dims": over.astype(np.int8).tolist(),
                }
                if spec_debug:
                    self._append_log(
                        f"spec_reject kind={kind} pos={p} rank={i} conf={conf_pos:.4f} "
                        f"abs_diff_max={reject_detail['abs_diff_max']:.4f} over_dims={reject_detail['over_dims']} "
                        f"pred={np.round(pred_1d[:cmp_dim], 3).tolist()} "
                        f"draft={np.round(draft_1d[:cmp_dim], 3).tolist()}"
                    )
                return (
                    accepted,
                    accepted_rank,
                    False,
                    p,
                    pred_vec,
                    np.asarray(verify_actions_raw_slice[i, p]),
                    reject_detail,
                )
            return accepted, accepted_rank, True, None, None, None, None

        # ----- single-chunk verification over the full horizon -----
        verify_h = int(h)
        order_conf_pending = (
            _pending(order_conf, accepted_positions_conf)
            if conf_active
            else np.zeros((0,), dtype=np.int64)
        )
        order_seq_pending = (
            _pending(order_seq, accepted_positions_seq)
            if seq_active
            else np.zeros((0,), dtype=np.int64)
        )
        k_conf = int(order_conf_pending.shape[0])
        k_seq = int(order_seq_pending.shape[0])
        k_total = k_conf + k_seq

        reject_detail_conf: dict[str, Any] | None = None
        reject_detail_seq: dict[str, Any] | None = None
        if k_total > 0:
            fixed_actions_batch = np.zeros(
                (k_total, verify_h, d_model), dtype=greedy_chain_raw_full.dtype
            )
            fixed_action_mask_batch = np.zeros((k_total, verify_h), dtype=np.bool_)

            if conf_active:
                pos_fixed = np.asarray(
                    sorted(p for p in accepted_positions_conf if p < verify_h), dtype=np.int64
                )
                for i in range(k_conf):
                    if pos_fixed.size:
                        fixed_actions_batch[i, pos_fixed] = greedy_chain_raw_full[pos_fixed]
                        fixed_action_mask_batch[i, pos_fixed] = True
                    if i > 0:
                        prev = order_conf_pending[:i]
                        fixed_actions_batch[i, prev] = greedy_chain_raw_full[prev]
                        fixed_action_mask_batch[i, prev] = True

            if seq_active:
                pos_fixed = np.asarray(
                    sorted(p for p in accepted_positions_seq if p < verify_h), dtype=np.int64
                )
                for i in range(k_seq):
                    j = k_conf + i
                    if pos_fixed.size:
                        fixed_actions_batch[j, pos_fixed] = greedy_chain_raw_full[pos_fixed]
                        fixed_action_mask_batch[j, pos_fixed] = True
                    if i > 0:
                        prev = order_seq_pending[:i]
                        fixed_actions_batch[j, prev] = greedy_chain_raw_full[prev]
                        fixed_action_mask_batch[j, prev] = True

            obs_verify = self.sampler.repeat_observation(observation, k_total)
            verify_actions_raw = self.sampler.sample(
                obs_verify,
                num_steps=int(diffusion_num_steps),
                fixed_actions=fixed_actions_batch,
                fixed_action_mask=fixed_action_mask_batch,
                prefix_cache=prefix_cache,
            )
            verify_actions_exec = self._apply_transform_batch(
                output_transform_sample_fn, verify_actions_raw, state_np
            )

            if conf_active:
                (
                    accepted_positions_conf,
                    accepted_rank_conf,
                    conf_active,
                    conf_fail_pos,
                    conf_fail_exec,
                    conf_fail_raw,
                    reject_detail_conf,
                ) = _run_verify(
                    kind="conf",
                    verify_order=order_conf_pending,
                    verify_actions_exec_slice=verify_actions_exec[:k_conf],
                    verify_actions_raw_slice=verify_actions_raw[:k_conf],
                    accepted=accepted_positions_conf,
                    accepted_rank=accepted_rank_conf,
                )
                if not conf_active and fail_pos_conf is None:
                    fail_pos_conf = conf_fail_pos
                    fail_action_exec_conf = conf_fail_exec
                    fail_action_raw_conf = conf_fail_raw

            if seq_active:
                (
                    accepted_positions_seq,
                    accepted_rank_seq,
                    seq_active,
                    seq_fail_pos,
                    seq_fail_exec,
                    seq_fail_raw,
                    reject_detail_seq,
                ) = _run_verify(
                    kind="seq",
                    verify_order=order_seq_pending,
                    verify_actions_exec_slice=verify_actions_exec[k_conf:],
                    verify_actions_raw_slice=verify_actions_raw[k_conf:],
                    accepted=accepted_positions_seq,
                    accepted_rank=accepted_rank_seq,
                )
                if not seq_active and fail_pos_seq is None:
                    fail_pos_seq = seq_fail_pos
                    fail_action_exec_seq = seq_fail_exec
                    fail_action_raw_seq = seq_fail_raw

        accepted_prefix_len_conf = _prefix_len(accepted_positions_conf)
        accepted_prefix_len_seq = _prefix_len(accepted_positions_seq)
        accepted_prefix_len = int(min(accepted_prefix_len_conf, accepted_prefix_len_seq))

        # build return chunks (both exec for display / final output, and raw)
        chunk_exec = greedy_chain_exec[:accepted_prefix_len].copy()
        chunk_raw = greedy_chain_raw_full[:accepted_prefix_len].copy()
        append_pos = None
        if accepted_prefix_len_conf < accepted_prefix_len_seq:
            if fail_pos_conf is not None and int(fail_pos_conf) == accepted_prefix_len:
                if fail_action_exec_conf is not None:
                    chunk_exec = np.concatenate([chunk_exec, fail_action_exec_conf[None]], axis=0)
                if fail_action_raw_conf is not None:
                    chunk_raw = np.concatenate([chunk_raw, fail_action_raw_conf[None]], axis=0)
                append_pos = int(fail_pos_conf)
        elif accepted_prefix_len_seq < accepted_prefix_len_conf:
            if fail_pos_seq is not None and int(fail_pos_seq) == accepted_prefix_len:
                if fail_action_exec_seq is not None:
                    chunk_exec = np.concatenate([chunk_exec, fail_action_exec_seq[None]], axis=0)
                if fail_action_raw_seq is not None:
                    chunk_raw = np.concatenate([chunk_raw, fail_action_raw_seq[None]], axis=0)
                append_pos = int(fail_pos_seq)
        else:
            if fail_pos_seq is not None and int(fail_pos_seq) == accepted_prefix_len:
                if fail_action_exec_seq is not None:
                    chunk_exec = np.concatenate([chunk_exec, fail_action_exec_seq[None]], axis=0)
                if fail_action_raw_seq is not None:
                    chunk_raw = np.concatenate([chunk_raw, fail_action_raw_seq[None]], axis=0)
                append_pos = int(fail_pos_seq)
            elif fail_pos_conf is not None and int(fail_pos_conf) == accepted_prefix_len:
                if fail_action_exec_conf is not None:
                    chunk_exec = np.concatenate([chunk_exec, fail_action_exec_conf[None]], axis=0)
                if fail_action_raw_conf is not None:
                    chunk_raw = np.concatenate([chunk_raw, fail_action_raw_conf[None]], axis=0)
                append_pos = int(fail_pos_conf)

        accepted_actions_exec = np.asarray(chunk_exec)
        accepted_exec_len = int(accepted_actions_exec.shape[0]) if accepted_actions_exec.ndim > 0 else 0

        # pad to output_len
        if output_len > 0:
            if chunk_exec.shape[0] < int(output_len):
                pad_end = min(int(output_len), int(greedy_chain_exec.shape[0]))
                if chunk_exec.shape[0] < pad_end:
                    chunk_exec = np.concatenate(
                        [chunk_exec, greedy_chain_exec[chunk_exec.shape[0] : pad_end]], axis=0
                    )
                    chunk_raw = np.concatenate(
                        [chunk_raw, greedy_chain_raw_full[chunk_raw.shape[0] : pad_end]], axis=0
                    )
            if chunk_exec.shape[0] < int(output_len):
                last_exec = chunk_exec[-1] if chunk_exec.size else greedy_chain_exec[0]
                last_raw = chunk_raw[-1] if chunk_raw.size else greedy_chain_raw_full[0]
                pad_count = int(output_len) - int(chunk_exec.shape[0])
                chunk_exec = np.concatenate(
                    [chunk_exec, np.repeat(last_exec[None, ...], pad_count, axis=0)], axis=0
                )
                chunk_raw = np.concatenate(
                    [chunk_raw, np.repeat(last_raw[None, ...], pad_count, axis=0)], axis=0
                )
            if chunk_exec.shape[0] > int(output_len):
                chunk_exec = chunk_exec[: int(output_len)]
                chunk_raw = chunk_raw[: int(output_len)]

        info: dict[str, Any] = {
            "accepted_prefix_len": int(accepted_prefix_len),
            "accepted_prefix_len_conf": int(accepted_prefix_len_conf),
            "accepted_prefix_len_seq": int(accepted_prefix_len_seq),
            "accepted_rank": int(min(accepted_rank_conf, accepted_rank_seq)),
            "accepted_rank_conf": int(accepted_rank_conf),
            "accepted_rank_seq": int(accepted_rank_seq),
            "best_first_idx": int(best_first_idx),
            "selected_path": selected_path,
            "accepted_actions": accepted_actions_exec,
            "accepted_exec_len": int(accepted_exec_len),
            "conf_stats": conf_stats if log_conf_stats and isinstance(conf_stats, dict) else None,
            "spec_verify_conf": bool(conf_enabled),
            "spec_verify_seq": bool(seq_enabled),
        }
        reject_detail = None
        if accepted_prefix_len_conf < accepted_prefix_len_seq:
            reject_detail = reject_detail_conf
        elif accepted_prefix_len_seq < accepted_prefix_len_conf:
            reject_detail = reject_detail_seq
        else:
            reject_detail = reject_detail_seq or reject_detail_conf
        if reject_detail is not None:
            info["reject"] = reject_detail
        if append_pos is not None:
            info["append_pos"] = int(append_pos)
        return chunk_raw, chunk_exec, info


def _as_numpy(x) -> np.ndarray:
    """Convert JAX / torch tensors to numpy (float32 for non-supported float types)."""
    if x is None:
        return x
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        try:
            import torch  # noqa: WPS433

            t = x.detach().cpu()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            return np.asarray(t)
        except ImportError:
            pass
    return np.asarray(x)
