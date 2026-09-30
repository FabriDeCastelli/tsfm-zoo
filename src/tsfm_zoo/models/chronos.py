"""Chronos family (amazon-science/chronos-forecasting 2.3.2): Chronos-2, Chronos-Bolt, Chronos (T5)."""
import math

import torch
from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutput

from ..base import DEFAULT_QUANTILES, TSFM, MissingInputError
from ..checkpoints import fetch
from ..forecast import Forecast, select_quantiles
from ._common import check_head_state, patch_layout, run_encoder, special_tokens


class Chronos2(TSFM):
    """Chronos-2: patch-16 encoder with group attention across the rows of a group.

    Rows of one group are the C channels of a batch item plus its covariates.
    The encoder is bidirectional and appends one token per future patch, so
    context tokens attend to future tokens: `z` depends on how many future
    patches are appended. Without `horizon` this is one, which is exactly
    upstream's `embed`; with `horizon` it is ceil(horizon / 16), and the
    native head reads those future tokens.
    """

    name = "chronos-2"
    uses = frozenset({"mask", "past_covariates", "future_covariates", "channel_groups", "layer"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = True
    summary = "reg"
    training_corpora = "autogluon/chronos_datasets + synthetic (no GIFT-Eval-Pretrain)"

    def __init__(self, pipeline, seed: int = 0):
        super().__init__(seed)
        self.pipeline = pipeline
        self.model = pipeline.model  # registered here so .to()/.eval() reach it
        cfg = self.model.chronos_config
        self.embed_dim = self.model.model_dim
        self.max_context = cfg.context_length
        self.patch_size = cfg.input_patch_size
        self.max_output_patches = cfg.max_output_patches
        self.native_levels = [float(q) for q in cfg.quantiles]
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", dtype=torch.float32):
        path = fetch("chronos", variant)
        pipeline = Chronos2Pipeline.from_pretrained(path, device_map=device, dtype=dtype)
        return cls(pipeline, seed=seed)

    def _encode(self, x, mask, timestamps, *, horizon=None, past_covariates=None, future_covariates=None,
                channel_groups=None, layer=None, **_):
        B, T, C = x.shape
        x, mask = self._truncate(x, mask, self.max_context)
        T_ctx = x.shape[1]
        n_out = min(math.ceil(horizon / self.patch_size), self.max_output_patches) if horizon else 1

        # Rows follow upstream's order: targets, past-only covariates, known-future covariates.
        rows = [x.transpose(1, 2)]
        if past_covariates is not None:
            rows.append(past_covariates[:, -T_ctx:].transpose(1, 2))
        n_future = 0
        if future_covariates is not None:
            if future_covariates.shape[1] <= T:
                raise ValueError(f"{self.name}: future_covariates must be [B, T+H, K], got T+H={future_covariates.shape[1]} <= T={T}")
            n_future = future_covariates.shape[2]
            rows.append(future_covariates[:, T - T_ctx:T].transpose(1, 2))
        context = torch.cat(rows, dim=1).float()                       # [B, R, T_ctx]
        R = context.shape[1]

        future = None
        if future_covariates is not None:
            known = future_covariates[:, T:].transpose(1, 2).float()   # [B, K_f, H_cov]
            future = torch.full((B, R, known.shape[-1]), torch.nan, device=x.device)
            future[:, R - n_future:] = known
            future = future.flatten(0, 1)

        group_ids = torch.arange(B, device=x.device).repeat_interleave(R)
        if channel_groups is not None:
            if R > C:
                raise ValueError(f"{self.name}: channel_groups cannot be combined with covariates")
            G = int(channel_groups.max()) + 1
            group_ids = (torch.arange(B, device=x.device)[:, None] * G + channel_groups.to(x.device)[None]).flatten()

        device = self.model.device
        (encoder_out, loc_scale, *_), at_layer = run_encoder(self.model.encoder.block, layer, lambda: self.model.encode(
            context=context.flatten(0, 1).to(device),
            group_ids=group_ids.to(device),
            future_covariates=None if future is None else future[:, :n_out * self.patch_size].to(device),
            num_output_patches=n_out,
        ))
        hidden = encoder_out[0] if layer is None else at_layer
        hidden = hidden.view(B, R, -1, self.embed_dim)

        n_patches = hidden.shape[2] - 1 - n_out
        loc, scale = (t.view(B, R) for t in loc_scale)
        self.last_scale = (loc[:, :C], scale[:, :C])
        self.token_kinds = ["patch"] * n_patches + ["reg"] + ["future"] * n_out
        spans, observed = patch_layout(mask, T, self.patch_size, n_patches)
        future_spans = [[T + self.patch_size * j, T + self.patch_size * (j + 1)] for j in range(n_out)]
        self.token_span, self.token_observed = special_tokens(spans, observed, [[-1, -1]] + future_spans)
        self._state = dict(context=context.flatten(0, 1), group_ids=group_ids, future=future, hidden=hidden,
                          loc_scale=loc_scale, n_out=n_out, C=C, R=R, layer=layer)
        return hidden[:, :C]

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        rows = check_head_state(self, z)
        if horizon is None:
            raise MissingInputError(f"{self.name}: task='forecast' needs horizon")

        n_out, B, R, C = rows["n_out"], z.shape[0], rows["R"], rows["C"]
        hidden = rows["hidden"].flatten(0, 1)
        preds = self.model.output_patch_embedding(hidden[:, -n_out:])
        preds = rearrange(preds, "b n (q p) -> b (q n p)", q=len(self.native_levels), p=self.patch_size)
        preds = self.model.instance_norm.inverse(preds, rows["loc_scale"])
        preds = rearrange(preds, "b (q h) -> b q h", q=len(self.native_levels))    # [B*R, Qn, h]

        # Past the native horizon, upstream unrolls quantile paths autoregressively.
        steps, parts, remaining = 0, [preds], horizon - preds.shape[-1]
        if remaining > 0:
            p = self.pipeline
            device = self.model.device
            context, group_ids, future, weights = p._prepare_inputs_for_long_horizon_unrolling(
                context=rows["context"].to(device), group_ids=rows["group_ids"].to(device),
                future_covariates=self._future_for_unroll(rows, horizon),
                unrolled_quantiles=torch.tensor(DEFAULT_QUANTILES))
            while remaining > 0:
                preds, context, future = p._autoregressive_unroll_for_long_horizon(
                    context=context, group_ids=group_ids, future_covariates=future, prediction=preds,
                    unrolled_quantiles=torch.tensor(DEFAULT_QUANTILES), unrolled_sample_weights=weights,
                    num_output_patches=min(math.ceil(remaining / self.patch_size), self.max_output_patches))
                parts.append(preds)
                remaining -= preds.shape[-1]
                steps += 1
        preds = torch.cat(parts, dim=-1)[..., :horizon].float()
        preds = preds.view(B, R, len(self.native_levels), horizon)[:, :C].permute(0, 3, 1, 2)  # [B, H, C, Qn]

        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
            rollout_steps=steps,
        )

    def _future_for_unroll(self, rows, horizon):
        """Future covariates over the whole horizon, NaN-padded; upstream slides them after each step."""
        future = rows["future"]
        if future is None:
            return None
        future = future[:, :horizon]
        pad = horizon - future.shape[-1]
        if pad > 0:
            future = torch.nn.functional.pad(future, (0, pad), value=torch.nan)
        return future.to(self.model.device)


class ChronosBolt(TSFM):
    """Chronos-Bolt: patch-16 T5 encoder plus a one-token decoder that emits 64 steps of 9 quantiles.

    Univariate: the C channels are folded into the batch.
    """

    name = "chronos-bolt"
    uses = frozenset({"mask", "layer"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "reg"
    training_corpora = "autogluon/chronos_datasets + synthetic; leaderboard flags test-data leakage for GIFT-Eval"

    def __init__(self, pipeline, seed: int = 0):
        super().__init__(seed)
        self.pipeline = pipeline
        self.model = pipeline.model
        cfg = self.model.chronos_config
        self.embed_dim = self.model.model_dim
        self.max_context = cfg.context_length
        self.patch_size = cfg.input_patch_size
        self.native_horizon = cfg.prediction_length
        self.native_levels = [float(q) for q in cfg.quantiles]
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", dtype=torch.float32):
        pipeline = ChronosBoltPipeline.from_pretrained(fetch("chronos", variant), device_map=device, dtype=dtype)
        return cls(pipeline, seed=seed)

    def _encode(self, x, mask, timestamps, *, layer=None, **_):
        B, T, C = x.shape
        x, mask = self._truncate(x, mask, self.max_context)
        context = x.transpose(1, 2).flatten(0, 1).float().to(self.model.device)     # [B*C, T_ctx]
        (hidden, loc_scale, input_embeds, attention_mask), at_layer = run_encoder(
            self.model.encoder.block, layer, lambda: self.model.encode(context=context))
        hidden = (hidden if layer is None else at_layer).view(B, C, -1, self.embed_dim)

        n_patches = hidden.shape[2] - 1
        self.last_scale = tuple(t.view(B, C) for t in loc_scale)
        self.token_kinds = ["patch"] * n_patches + ["reg"]
        spans, observed = patch_layout(mask, T, self.patch_size, n_patches)
        self.token_span, self.token_observed = special_tokens(spans, observed, [[-1, -1]])
        self._state = dict(hidden=hidden, context=context, loc_scale=loc_scale, input_embeds=input_embeds,
                           attention_mask=attention_mask, layer=layer)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        B, C = z.shape[:2]
        Qn = len(self.native_levels)
        m = self.model
        out = m.decode(state["input_embeds"], state["attention_mask"], state["hidden"].flatten(0, 1))
        pred = m.output_patch_embedding(out).view(B * C, Qn, self.native_horizon)
        pred = m.instance_norm.inverse(pred.view(B * C, -1), state["loc_scale"]).view(B * C, Qn, -1)

        # Past the native horizon: upstream's loop (ChronosBoltPipeline.predict), unchanged.
        predictions, remaining, steps = [pred], horizon - pred.shape[-1], 0
        context = state["context"]
        if remaining > 0:
            context = context.unsqueeze(1).repeat(1, Qn, 1)
        quantile_tensor = torch.tensor(self.native_levels, device=context.device)
        while remaining > 0:
            context = torch.cat([context, pred], dim=-1)[..., -self.max_context:]
            b, n, t = context.shape
            pred = m(context=context.reshape(b * n, t)).quantile_preds.to(context)
            pred = torch.quantile(pred.reshape(b, n * n, -1), q=quantile_tensor, dim=1).transpose(0, 1)
            predictions.append(pred)
            remaining -= pred.shape[-1]
            steps += 1
        preds = torch.cat(predictions, dim=-1)[..., :horizon].float()
        preds = preds.view(B, C, Qn, horizon).permute(0, 3, 1, 2)                    # [B, H, C, Qn]

        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
            rollout_steps=steps,
        )


class ChronosT5(TSFM):
    """Chronos (T5): mean-scaled values quantized into tokens, one token per time step, plus EOS.

    The native head samples token paths from the T5 decoder, cross-attending
    to `z`. Sampling uses the model's seed, not the global RNG. Univariate:
    the C channels are folded into the batch.
    """

    name = "chronos-t5"
    uses = frozenset({"mask", "layer", "num_samples"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "mean"
    normalization = "scale"
    training_corpora = "autogluon/chronos_datasets (+ KernelSynth); leaderboard flags test-data leakage for GIFT-Eval"

    def __init__(self, pipeline, seed: int = 0):
        super().__init__(seed)
        self.pipeline = pipeline
        self.model = pipeline.model  # ChronosModel; .model is the HF T5
        cfg = self.model.config
        self.embed_dim = self.model.model.config.d_model
        self.max_context = cfg.context_length
        self.patch_size = 1
        self.native_horizon = cfg.prediction_length
        self.default_samples = cfg.num_samples
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", dtype=torch.float32):
        pipeline = ChronosPipeline.from_pretrained(fetch("chronos", variant), device_map=device, dtype=dtype)
        return cls(pipeline, seed=seed)

    def _tokens(self, context):
        token_ids, attention_mask, scale = self.pipeline.tokenizer.context_input_transform(context.cpu())
        return token_ids.to(self.model.device), attention_mask.to(self.model.device), scale

    def _encode(self, x, mask, timestamps, *, layer=None, **_):
        B, T, C = x.shape
        x, mask = self._truncate(x, mask, self.max_context)
        context = x.transpose(1, 2).flatten(0, 1).float()                            # [B*C, T_ctx]
        token_ids, attention_mask, scale = self._tokens(context)
        hidden, at_layer = run_encoder(self.model.model.encoder.block, layer,
                                       lambda: self.model.encode(input_ids=token_ids, attention_mask=attention_mask))
        hidden = (hidden if layer is None else at_layer).view(B, C, -1, self.embed_dim)

        n_steps = context.shape[-1]
        self.last_scale = (torch.zeros(B, C), scale.view(B, C))  # mean scaling: no shift
        self.token_kinds = ["patch"] * n_steps + ["eos"]
        spans, observed = patch_layout(mask, T, 1, n_steps)
        self.token_span, self.token_observed = special_tokens(spans, observed, [[-1, -1]])
        self._state = dict(hidden=hidden, context=context, attention_mask=attention_mask, scale=scale, layer=layer)
        return hidden

    def _sample(self, hidden, attention_mask, scale, length, num_samples):
        """Upstream's ChronosModel.forward, with encoder outputs given instead of recomputed."""
        m = self.model
        cfg = m.config
        with torch.random.fork_rng(devices=[m.device] if m.device.type == "cuda" else []):
            torch.manual_seed(self._call_seed)
            self._call_seed += 1
            preds = m.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                attention_mask=attention_mask,
                generation_config=self._generation_config(length, num_samples),
            )
        samples = preds[..., 1:].reshape(hidden.shape[0], num_samples, -1)
        return self.pipeline.tokenizer.output_transform(samples.cpu(), scale)

    def _generation_config(self, length, num_samples):
        from transformers import GenerationConfig
        cfg = self.model.config
        return GenerationConfig(min_new_tokens=length, max_new_tokens=length, do_sample=True,
                                num_return_sequences=num_samples, eos_token_id=cfg.eos_token_id,
                                pad_token_id=cfg.pad_token_id, temperature=cfg.temperature,
                                top_k=cfg.top_k, top_p=cfg.top_p)

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, num_samples=None, **_):
        state = check_head_state(self, z)
        B, C = z.shape[:2]
        num_samples = num_samples or self.default_samples
        self._call_seed = self.seed

        # Past the native horizon: upstream's loop (ChronosPipeline.predict) appends the sample median.
        context = state["context"]
        pred = self._sample(state["hidden"].flatten(0, 1), state["attention_mask"], state["scale"],
                            min(horizon, self.native_horizon), num_samples)
        predictions, remaining, steps = [pred], horizon - pred.shape[-1], 0
        while remaining > 0:
            context = torch.cat([context, pred.median(dim=1).values], dim=-1)
            token_ids, attention_mask, scale = self._tokens(context)
            hidden = self.model.encode(input_ids=token_ids, attention_mask=attention_mask)
            pred = self._sample(hidden, attention_mask, scale, min(remaining, self.native_horizon), num_samples)
            predictions.append(pred)
            remaining -= pred.shape[-1]
            steps += 1
        samples = torch.cat(predictions, dim=-1).float()                             # [B*C, S, H]
        samples = samples.view(B, C, num_samples, horizon).permute(0, 2, 3, 1)       # [B, S, H, C]

        levels = list(quantile_levels or DEFAULT_QUANTILES)
        quantiles = torch.quantile(samples, torch.tensor(levels), dim=1).permute(1, 2, 3, 0)
        return Forecast(
            median=samples.quantile(0.5, dim=1),
            quantiles=quantiles,
            quantile_levels=torch.tensor(levels),
            samples=samples,
            rollout_steps=steps,
        )


class Chronos:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "chronos-2"
        if variant.startswith("chronos-2"):
            return Chronos2.from_pretrained(variant, **kwargs)
        if variant.startswith("bolt-"):
            return ChronosBolt.from_pretrained(variant, **kwargs)
        if variant.startswith("t5-"):
            return ChronosT5.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown chronos variant {variant!r}")


MODEL = Chronos
