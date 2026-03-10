from __future__ import annotations

from dataclasses import dataclass

import torch

from tabula.data.datasets import TabularBatch


@dataclass
class EpisodeBatch:
    support: TabularBatch
    query: TabularBatch


def _index_text_values(values: list[list[str]], indices: torch.Tensor) -> list[list[str]]:
    index_list = indices.detach().cpu().tolist()
    return [list(values[idx]) for idx in index_list]


def _slice_batch(batch: TabularBatch, indices: torch.Tensor) -> TabularBatch:
    return TabularBatch(
        x_num=batch.x_num.index_select(0, indices),
        x_cat=batch.x_cat.index_select(0, indices),
        x_text_token_ids=batch.x_text_token_ids.index_select(0, indices),
        x_text_values=_index_text_values(batch.x_text_values, indices),
        x_num_mask=batch.x_num_mask.index_select(0, indices),
        x_cat_mask=batch.x_cat_mask.index_select(0, indices),
        x_text_mask=batch.x_text_mask.index_select(0, indices),
        num_schema_texts=batch.num_schema_texts,
        cat_schema_texts=batch.cat_schema_texts,
        text_schema_texts=batch.text_schema_texts,
        num_name_token_ids=batch.num_name_token_ids,
        cat_name_token_ids=batch.cat_name_token_ids,
        text_name_token_ids=batch.text_name_token_ids,
        num_profile_vectors=batch.num_profile_vectors,
        cat_profile_vectors=batch.cat_profile_vectors,
        text_profile_vectors=batch.text_profile_vectors,
        y=batch.y.index_select(0, indices),
    )


def sample_episode_batch(
    batch: TabularBatch,
    support_size: int,
    query_size: int,
    sample_with_replacement: bool = False,
    generator: torch.Generator | None = None,
) -> EpisodeBatch:
    row_count = batch.x_num.shape[0]
    total = support_size + query_size
    if row_count == 0:
        raise ValueError("Cannot sample an episode from an empty batch.")
    if not sample_with_replacement and total > row_count:
        raise ValueError(
            f"Episode requires {total} rows but batch only has {row_count}. "
            "Enable replacement or reduce support/query size."
        )
    if sample_with_replacement:
        indices = torch.randint(row_count, (total,), generator=generator, device=batch.x_num.device)
    else:
        indices = torch.randperm(row_count, generator=generator, device=batch.x_num.device)[:total]
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]
    return EpisodeBatch(
        support=_slice_batch(batch, support_indices),
        query=_slice_batch(batch, query_indices),
    )
