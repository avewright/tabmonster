from __future__ import annotations

import torch
from torch.nn import functional as F
from torch import nn

from tabula.config import ExperimentConfig
from tabula.data.datasets import TabularBatch
from tabula.data.episodes import EpisodeBatch


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale * self.weight


def _build_norm(kind: str, d_model: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(d_model)
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"Unsupported normalization kind '{kind}'.")


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.ff = nn.ModuleDict(
                {
                    "value": nn.Linear(d_model, d_ff),
                    "gate": nn.Linear(d_model, d_ff),
                    "out": nn.Linear(d_ff, d_model),
                }
            )
        elif activation == "gelu":
            self.ff = nn.ModuleDict(
                {
                    "value": nn.Linear(d_model, d_ff),
                    "out": nn.Linear(d_ff, d_model),
                }
            )
        else:
            raise ValueError(f"Unsupported feed-forward activation '{activation}'.")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            hidden = self.ff["value"](x) * F.silu(self.ff["gate"](x))
        else:
            hidden = F.gelu(self.ff["value"](x))
        hidden = self.dropout(hidden)
        return self.ff["out"](hidden)


class NumericFeatureTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_features: int,
        embedding_type: str,
        periodic_features: int,
    ) -> None:
        super().__init__()
        if embedding_type == "periodic" and periodic_features < 1:
            raise ValueError("numeric_periodic_features must be at least 1 when numeric_embedding='periodic'.")
        self.d_model = d_model
        self.num_features = num_features
        self.embedding_type = embedding_type
        input_dim = 1 if embedding_type == "linear" else 1 + 2 * periodic_features
        self.projections = nn.ModuleList([nn.Linear(input_dim, d_model) for _ in range(num_features)])
        if embedding_type == "periodic":
            self.periodic_weight = nn.Parameter(torch.randn(num_features, periodic_features))
            self.periodic_bias = nn.Parameter(torch.zeros(num_features, periodic_features))
        else:
            self.register_parameter("periodic_weight", None)
            self.register_parameter("periodic_bias", None)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        if self.num_features == 0:
            return x_num.new_zeros((x_num.shape[0], 0, self.d_model))
        tokens: list[torch.Tensor] = []
        for idx, projection in enumerate(self.projections):
            value = x_num[:, idx : idx + 1]
            if self.embedding_type == "periodic":
                if self.periodic_weight is None or self.periodic_bias is None:
                    raise RuntimeError("Periodic numeric parameters are not initialized.")
                phase = value * self.periodic_weight[idx].unsqueeze(0) + self.periodic_bias[idx].unsqueeze(0)
                value = torch.cat([value, torch.sin(phase), torch.cos(phase)], dim=-1)
            tokens.append(projection(value))
        return torch.stack(tokens, dim=1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        norm_kind: str,
        ffn_activation: str,
    ) -> None:
        super().__init__()
        self.norm1 = _build_norm(norm_kind, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = _build_norm(norm_kind, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, ffn_activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TextCellEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_tokens: int,
        norm_kind: str,
        ffn_activation: str,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max(max_tokens, 1), d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_model * 2,
                    dropout=dropout,
                    norm_kind=norm_kind,
                    ffn_activation=ffn_activation,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = _build_norm(norm_kind, d_model)

    def forward(self, token_embeddings: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if token_embeddings.shape[1] == 0:
            return token_embeddings.new_zeros((token_embeddings.shape[0], token_embeddings.shape[-1]))
        valid_rows = token_mask.any(dim=1)
        if not valid_rows.any():
            return token_embeddings.new_zeros((token_embeddings.shape[0], token_embeddings.shape[-1]))
        positions = torch.arange(token_embeddings.shape[1], device=token_embeddings.device)
        hidden = token_embeddings[valid_rows] + self.position_embedding(positions)[None, :, :]
        key_padding_mask = ~token_mask[valid_rows]
        for block in self.blocks:
            hidden = block(hidden, key_padding_mask=key_padding_mask)
        pooled = (hidden * token_mask[valid_rows].unsqueeze(-1)).sum(dim=1) / token_mask[valid_rows].sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        encoded = token_embeddings.new_zeros((token_embeddings.shape[0], token_embeddings.shape[-1]))
        encoded[valid_rows] = self.norm(pooled)
        return encoded


class PretrainedTextCellEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        max_length: int,
        trainable: bool,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Pretrained text encoding requires `transformers`. Install it with `pip install transformers` "
                "or switch model.text_encoder back to `custom`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.trainable = trainable
        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        self.projection = nn.Linear(hidden_size, output_dim)
        if not trainable:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, text_values: list[str], device: torch.device) -> torch.Tensor:
        if not text_values:
            return self.projection.weight.new_zeros((0, self.projection.out_features))
        encoded = self.tokenizer(
            text_values,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.set_grad_enabled(self.trainable):
            outputs = self.encoder(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1)
        return self.projection(pooled)


class SchemaTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        max_length: int,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.encoder = PretrainedTextCellEncoder(
            model_name=model_name,
            output_dim=output_dim,
            max_length=max_length,
            trainable=trainable,
        )
        self.trainable = trainable
        self._cache: dict[tuple[torch.device, tuple[str, ...]], torch.Tensor] = {}

    def forward(self, schema_texts: list[str], device: torch.device) -> torch.Tensor:
        if not schema_texts:
            return self.encoder.projection.weight.new_zeros((0, self.encoder.projection.out_features), device=device)
        cache_key = (device, tuple(schema_texts))
        if not self.trainable and cache_key in self._cache:
            return self._cache[cache_key]
        encoded = self.encoder(schema_texts, device)
        if not self.trainable:
            self._cache[cache_key] = encoded.detach()
        return encoded


class AttentionPooling(nn.Module):
    """Learned attention pooling over sequence tokens."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> (B, D) via single attention query
        query = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(query, x, x, need_weights=False)
        return out.squeeze(1)


class TabularTransformer(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        num_numeric: int,
        num_categorical: int,
        num_text: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        model_cfg = config.model
        if model_cfg.text_encoder not in {"custom", "pretrained"}:
            raise ValueError(f"Unsupported text encoder '{model_cfg.text_encoder}'.")
        if model_cfg.schema_encoder not in {"hash", "pretrained"}:
            raise ValueError(f"Unsupported schema encoder '{model_cfg.schema_encoder}'.")
        d_model = model_cfg.d_model
        self.text_encoder_kind = model_cfg.text_encoder
        self.schema_encoder_kind = model_cfg.schema_encoder
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.num_text = num_text
        self.feature_token_dropout = model_cfg.feature_token_dropout
        self.max_categories = model_cfg.max_categories

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_embedding = nn.Embedding(4, d_model)
        self.feature_slot_embedding = nn.Embedding(max(num_numeric + num_categorical + num_text + 1, 1), d_model)
        self.num_embeddings = nn.Parameter(torch.randn(1, max(num_numeric, 1), d_model) * 0.02)
        self.num_tokenizer = NumericFeatureTokenizer(
            d_model=d_model,
            num_features=num_numeric,
            embedding_type=model_cfg.numeric_embedding,
            periodic_features=model_cfg.numeric_periodic_features,
        )
        self.text_embeddings = nn.Parameter(torch.randn(1, max(num_text, 1), d_model) * 0.02)
        self.register_buffer("cat_offset", torch.arange(num_categorical), persistent=False)
        self.cat_embedding = nn.Embedding(model_cfg.max_categories * max(num_categorical, 1), d_model)
        self.text_token_embedding = (
            nn.Embedding(8192, d_model, padding_idx=0) if self.text_encoder_kind == "custom" else None
        )
        self.text_encoder = (
            TextCellEncoder(
                d_model=d_model,
                n_heads=model_cfg.text_encoder_heads,
                n_layers=model_cfg.text_encoder_layers,
                dropout=model_cfg.dropout,
                max_tokens=model_cfg.text_max_tokens,
                norm_kind=model_cfg.norm,
                ffn_activation=model_cfg.ffn_activation,
            )
            if self.text_encoder_kind == "custom"
            else PretrainedTextCellEncoder(
                model_name=model_cfg.text_pretrained_model_name,
                output_dim=d_model,
                max_length=model_cfg.text_pretrained_max_length,
                trainable=model_cfg.text_pretrained_trainable,
            )
        )
        self.name_embedding = nn.Embedding(8192, d_model, padding_idx=0)
        self.schema_text_encoder = (
            SchemaTextEncoder(
                model_name=model_cfg.schema_pretrained_model_name,
                output_dim=d_model,
                max_length=model_cfg.schema_pretrained_max_length,
                trainable=model_cfg.schema_pretrained_trainable,
            )
            if self.schema_encoder_kind == "pretrained"
            else None
        )
        self.profile_projection = nn.Linear(11, d_model)
        self.present_embedding = nn.Embedding(2, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=model_cfg.n_heads,
                    d_ff=model_cfg.d_ff,
                    dropout=model_cfg.dropout,
                    norm_kind=model_cfg.norm,
                    ffn_activation=model_cfg.ffn_activation,
                )
                for _ in range(model_cfg.n_layers)
            ]
        )
        self.norm = _build_norm(model_cfg.norm, d_model)
        self.pooling_kind = getattr(model_cfg, "pooling", "cls")
        self.attention_pool = AttentionPooling(d_model) if self.pooling_kind == "attention" else None
        self.head = nn.Linear(d_model, output_dim)

    def _feature_metadata_embedding(
        self,
        schema_texts: list[str],
        name_token_ids: torch.Tensor,
        profile_vectors: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if not schema_texts and name_token_ids.numel() == 0 and profile_vectors.numel() == 0:
            token_count = profile_vectors.shape[0] if profile_vectors.ndim > 0 else 0
            return self.cls_token.new_zeros((token_count, self.head.in_features))
        if self.schema_encoder_kind == "pretrained":
            if self.schema_text_encoder is None:
                raise RuntimeError("Pretrained schema encoder is not initialized correctly.")
            name_emb = self.schema_text_encoder(schema_texts, device)
        else:
            name_emb = (
                (
                    self.name_embedding(name_token_ids) * name_token_ids.ne(0).unsqueeze(-1)
                ).sum(dim=1)
                / name_token_ids.ne(0).sum(dim=1, keepdim=True).clamp_min(1)
                if name_token_ids.numel() > 0
                else self.cls_token.new_zeros((profile_vectors.shape[0], self.head.in_features), device=device)
            )
        profile_emb = (
            self.profile_projection(profile_vectors)
            if profile_vectors.numel() > 0
            else self.cls_token.new_zeros((len(schema_texts) or name_token_ids.shape[0], self.head.in_features), device=device)
        )
        return name_emb + profile_emb

    def _encode_numeric(
        self,
        x_num: torch.Tensor,
        x_num_mask: torch.Tensor | None = None,
        num_schema_texts: list[str] | None = None,
        num_name_token_ids: torch.Tensor | None = None,
        num_profile_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_numeric == 0:
            return x_num.new_zeros((x_num.shape[0], 0, self.head.in_features))
        tokens = self.num_tokenizer(x_num) + self.num_embeddings[:, : self.num_numeric]
        if num_name_token_ids is not None and num_profile_vectors is not None:
            metadata = self._feature_metadata_embedding(
                num_schema_texts or [],
                num_name_token_ids,
                num_profile_vectors,
                x_num.device,
            )
            tokens = tokens + metadata.unsqueeze(0)
        if x_num_mask is not None:
            tokens = tokens + self.present_embedding(x_num_mask.long())
        tokens = tokens + self.modality_embedding.weight[1].view(1, 1, -1)
        return tokens

    def _encode_categorical(
        self,
        x_cat: torch.Tensor,
        x_cat_mask: torch.Tensor | None = None,
        cat_schema_texts: list[str] | None = None,
        cat_name_token_ids: torch.Tensor | None = None,
        cat_profile_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_categorical == 0:
            return x_cat.new_zeros((x_cat.shape[0], 0, self.head.in_features), dtype=torch.float32)
        offsets = self.cat_offset.to(x_cat.device) * self.max_categories
        embedded = self.cat_embedding(x_cat + offsets)
        if cat_name_token_ids is not None and cat_profile_vectors is not None:
            metadata = self._feature_metadata_embedding(
                cat_schema_texts or [],
                cat_name_token_ids,
                cat_profile_vectors,
                x_cat.device,
            )
            embedded = embedded + metadata.unsqueeze(0)
        if x_cat_mask is not None:
            embedded = embedded + self.present_embedding(x_cat_mask.long())
        embedded = embedded + self.modality_embedding.weight[2].view(1, 1, -1)
        return embedded

    def _encode_text(
        self,
        x_text_token_ids: torch.Tensor,
        x_text_values: list[list[str]] | None = None,
        x_text_mask: torch.Tensor | None = None,
        text_schema_texts: list[str] | None = None,
        text_name_token_ids: torch.Tensor | None = None,
        text_profile_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_text == 0:
            batch_size = x_text_token_ids.shape[0]
            return x_text_token_ids.new_zeros((batch_size, 0, self.head.in_features), dtype=torch.float32)
        batch_size, num_text, token_count = x_text_token_ids.shape
        if self.text_encoder_kind == "custom":
            if self.text_token_embedding is None or not isinstance(self.text_encoder, TextCellEncoder):
                raise RuntimeError("Custom text encoder is not initialized correctly.")
            flat_ids = x_text_token_ids.reshape(batch_size * num_text, token_count)
            flat_mask = flat_ids.ne(0)
            token_emb = self.text_token_embedding(flat_ids)
            pooled = self.text_encoder(token_emb, flat_mask).reshape(batch_size, num_text, self.head.in_features)
        else:
            if x_text_values is None:
                raise ValueError("x_text_values are required for the pretrained text encoder.")
            if not isinstance(self.text_encoder, PretrainedTextCellEncoder):
                raise RuntimeError("Pretrained text encoder is not initialized correctly.")
            flat_values = [value for row in x_text_values for value in row]
            pooled = self.text_encoder(flat_values, x_text_token_ids.device).reshape(
                batch_size, num_text, self.head.in_features
            )
        tokens = pooled + self.text_embeddings[:, : self.num_text]
        if text_name_token_ids is not None and text_profile_vectors is not None:
            metadata = self._feature_metadata_embedding(
                text_schema_texts or [],
                text_name_token_ids,
                text_profile_vectors,
                x_text_token_ids.device,
            )
            tokens = tokens + metadata.unsqueeze(0)
        if x_text_mask is not None:
            tokens = tokens + self.present_embedding(x_text_mask.long())
        tokens = tokens + self.modality_embedding.weight[3].view(1, 1, -1)
        return tokens

    def _apply_feature_token_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.feature_token_dropout <= 0 or tokens.shape[1] == 0:
            return tokens
        keep_prob = 1.0 - self.feature_token_dropout
        if keep_prob <= 0:
            return torch.zeros_like(tokens)
        keep_mask = (torch.rand(tokens.shape[:2], device=tokens.device) < keep_prob).unsqueeze(-1)
        return tokens * keep_mask.to(tokens.dtype) / keep_prob

    def _encode_to_cls(self, batch: TabularBatch) -> torch.Tensor:
        """Encode *batch* through the full transformer stack and return the normed
        CLS token representation (one vector per row, before the head).
        This is the canonical internal encoder used by both :meth:`forward` and
        :class:`EpisodicTabularTransformer`.
        """
        num_tokens = self._encode_numeric(
            batch.x_num,
            batch.x_num_mask,
            batch.num_schema_texts,
            batch.num_name_token_ids,
            batch.num_profile_vectors,
        )
        cat_tokens = self._encode_categorical(
            batch.x_cat,
            batch.x_cat_mask,
            batch.cat_schema_texts,
            batch.cat_name_token_ids,
            batch.cat_profile_vectors,
        )
        text_tokens = self._encode_text(
            batch.x_text_token_ids,
            batch.x_text_values,
            batch.x_text_mask,
            batch.text_schema_texts,
            batch.text_name_token_ids,
            batch.text_profile_vectors,
        )
        tokens = torch.cat([num_tokens, cat_tokens, text_tokens], dim=1)
        tokens = self._apply_feature_token_dropout(tokens)
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1) + self.modality_embedding.weight[0].view(1, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        feature_ids = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.feature_slot_embedding(feature_ids)[None, :, :]
        for block in self.blocks:
            tokens = block(tokens)
        
        if self.pooling_kind == "attention" and self.attention_pool is not None:
            return self.norm(self.attention_pool(tokens))
        elif self.pooling_kind == "mean":
            return self.norm(tokens.mean(dim=1))
        else:
            return self.norm(tokens[:, 0])

    def forward(self, x_num: torch.Tensor | TabularBatch, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(x_num, TabularBatch):
            return self.head(self._encode_to_cls(x_num))
        # Raw-tensor back-compat path (no TabularBatch wrapping).
        if x_cat is None:
            raise ValueError("x_cat must be provided when calling TabularTransformer with raw tensors.")
        if self.text_encoder_kind == "pretrained":
            raise ValueError("Raw tensor calls are not supported when model.text_encoder='pretrained'.")
        num_tokens = self._encode_numeric(x_num)
        cat_tokens = self._encode_categorical(x_cat)
        text_tokens = x_num.new_zeros((x_num.shape[0], 0, self.head.in_features))
        tokens = torch.cat([num_tokens, cat_tokens, text_tokens], dim=1)
        tokens = self._apply_feature_token_dropout(tokens)
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1) + self.modality_embedding.weight[0].view(1, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        feature_ids = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.feature_slot_embedding(feature_ids)[None, :, :]
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.norm(tokens[:, 0])
        return self.head(pooled)


class EpisodicTabularTransformer(nn.Module):
    """Episode-aware model built on top of :class:`TabularTransformer`.

    For each :class:`~tabula.data.episodes.EpisodeBatch` the support rows and
    query rows are independently encoded through the same backbone.  The
    support CLS vectors are mean-pooled into a single *support-context* vector
    that is linearly projected and added to every query CLS vector before the
    shared classification head fires.

    This is an induction-network style conditioning: the task prototype (support
    context) shifts the query representations without touching the backbone
    weights during inference.  The baseline :class:`TabularTransformer` is kept
    entirely intact so both models can be trained and compared side-by-side.

    Args:
        config: Full experiment configuration.
        num_numeric: Number of numeric input features.
        num_categorical: Number of categorical input features.
        num_text: Number of text input features.
        output_dim: Number of output logits.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        num_numeric: int,
        num_categorical: int,
        num_text: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = TabularTransformer(config, num_numeric, num_categorical, num_text, output_dim)
        d_model = config.model.d_model
        # Projects the mean-pooled support context before adding it to query CLS vectors.
        self.support_proj = nn.Linear(d_model, d_model, bias=False)

    @property
    def head(self) -> nn.Linear:
        """Expose the backbone classification head for external access."""
        return self.backbone.head

    def forward(self, episode: EpisodeBatch) -> torch.Tensor:  # type: ignore[override]
        """Compute logits for query rows conditioned on support rows.

        Args:
            episode: An :class:`~tabula.data.episodes.EpisodeBatch` containing
                ``support`` and ``query`` :class:`~tabula.data.datasets.TabularBatch`
                instances.  Both sets must already be on the same device.

        Returns:
            Float tensor of shape ``(query_size, output_dim)``.
        """
        if not isinstance(episode, EpisodeBatch):
            raise TypeError(
                "EpisodicTabularTransformer.forward() expects an EpisodeBatch; "
                "use TabularTransformer for plain TabularBatch inputs."
            )
        support_cls = self.backbone._encode_to_cls(episode.support)  # (S, d)
        query_cls = self.backbone._encode_to_cls(episode.query)      # (Q, d)
        support_ctx = self.support_proj(support_cls.mean(dim=0, keepdim=True))  # (1, d)
        augmented = query_cls + support_ctx                           # (Q, d)
        return self.backbone.head(augmented)
