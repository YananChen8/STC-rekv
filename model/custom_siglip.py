import torch
from torch import nn
from typing import Optional, Tuple, Union,List

import torch.nn.functional as F
import os

from transformers.models.siglip.modeling_siglip import SiglipEncoder, SiglipEncoderLayer, SiglipConfig
from transformers.modeling_outputs import BaseModelOutput
from model.cache import *
import time
import math
import torch
from typing import Optional, Tuple
from collections import defaultdict
from logzero import logger  # 添加这行
from model.config import get_config
import types
import torch.distributed as dist
import json




_METRICS_DUMP_COUNT = 0
_METRICS_DUMP_PATH = None


def _record_cache_metrics(
    layer_idx: int,
    chunk_idx: int,
    similarity: Optional[torch.Tensor],
    update_indices: torch.Tensor,
    importance_scores: Optional[torch.Tensor],
    num_update_before_importance: int,
    num_update_after_importance: int,
) -> None:
    global _METRICS_DUMP_COUNT, _METRICS_DUMP_PATH

    cfg = get_config().cache
    if not cfg.metrics_dump_enabled:
        return
    if _METRICS_DUMP_COUNT >= cfg.metrics_dump_max_records:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    if _METRICS_DUMP_PATH is None:
        rank_suffix = ""
        if dist.is_available() and dist.is_initialized():
            rank_suffix = f"_rank{dist.get_rank()}"
        output_path = cfg.metrics_dump_path or f"./cache_token_metrics{rank_suffix}.jsonl"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _METRICS_DUMP_PATH = output_path
        with open(_METRICS_DUMP_PATH, "w", encoding="utf-8"):
            pass

    if similarity is None:
        return

    similarity_cpu = similarity.detach().float().cpu()
    update_indices_cpu = update_indices.detach().cpu()
    importance_cpu = None if importance_scores is None else importance_scores.detach().float().cpu()

    seq_len = similarity_cpu.shape[1]
    for frame_idx in range(similarity_cpu.shape[0]):
        if _METRICS_DUMP_COUNT >= cfg.metrics_dump_max_records:
            break

        importance_dense = [None] * seq_len
        if importance_cpu is not None:
            frame_indices = update_indices_cpu[frame_idx].tolist()
            frame_scores = importance_cpu[frame_idx].tolist()
            for token_idx, score in zip(frame_indices, frame_scores):
                importance_dense[token_idx] = float(score)

        record = {
            "chunk_idx": int(chunk_idx),
            "layer_idx": int(layer_idx),
            "frame_idx": int(frame_idx),
            "num_tokens": int(seq_len),
            "similarity": similarity_cpu[frame_idx].tolist(),
            "importance": importance_dense,
            "update_keep_ratio_before_importance": float(num_update_before_importance / seq_len),
            "update_prune_ratio_before_importance": float(1.0 - num_update_before_importance / seq_len),
            "update_keep_ratio_after_importance": float(num_update_after_importance / seq_len),
            "update_prune_ratio_after_importance": float(1.0 - num_update_after_importance / seq_len),
            "importance_computed_token_count": int(0 if importance_cpu is None else importance_cpu.shape[1]),
        }
        with open(_METRICS_DUMP_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _METRICS_DUMP_COUNT += 1


def register_cache_by_key_Siglip(vision_tower: nn.Module) -> None:
    for layer_idx, layer in enumerate(vision_tower.vision_model.encoder.layers):
        setattr(layer, "_old_forward", layer.forward)
        setattr(layer, "_stc_layer_idx", layer_idx)
        layer.forward = types.MethodType(forward_with_selective_key_recompute, layer)
        layer.new_attn= types.MethodType(new_siglip_sdpa_attn_forward, layer)
        
       
def register_cache_by_key_CLIP(vision_tower: nn.Module) -> None:
    for layer_idx, layer in enumerate(vision_tower.vision_model.encoder.layers):
        setattr(layer, "_old_forward", layer.forward)
        setattr(layer, "_stc_layer_idx", layer_idx)
        layer.forward = types.MethodType(forward_with_selective_key_recompute_clip, layer)
        layer.new_attn= types.MethodType(new_siglip_sdpa_attn_forward, layer)


def _apply_importance_filter(
    hidden_states_ln1: torch.Tensor,
    prev_layer_hidden_states: Optional[torch.Tensor],
    update_indices: torch.Tensor,
    importance_keep_ratio: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    在已选中的低相似度 token 内，按层间变化（L2）保留更重要的 token。
    """
    if prev_layer_hidden_states is None:
        return update_indices, None

    batch_size, _, embed_dim = hidden_states_ln1.shape
    num_update = update_indices.shape[1]

    if num_update <= 1:
        return update_indices, None

    update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
    current_selected = hidden_states_ln1.gather(1, update_idx_expanded)

    prev_selected = prev_layer_hidden_states.gather(1, update_idx_expanded)

    importance_scores = torch.norm(current_selected - prev_selected, p=2, dim=-1)
    keep_k = max(1, min(num_update, int(num_update * importance_keep_ratio)))
    topk_local = torch.topk(importance_scores, k=keep_k, dim=1, largest=True).indices
    final_indices = update_indices.gather(1, topk_local)
    return final_indices, importance_scores
        
def forward_with_selective_key_recompute(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):

    
    cache2 = STC_CACHE()
    chunk_idx = cache2.chunk_idx
    cache_interval=get_config().cache.cache_interval
    update_cache = (chunk_idx % cache_interval == 0)
    
    # ========== 偶数chunk：完整计算并保存reference frame ==========
    if update_cache :
        # 标准的Transformer层计算
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        o_proj = self.self_attn.out_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V, AttnOut, MLPOut作为reference
        # 注意：保存的是projection后的张量，shape为[T, C]
        
        self.reference_frame_key = key_states[-1].clone().detach()  # [T, C]
        self.reference_frame_value = value_states[-1].clone().detach()  # [T, C]  # 修复这里！
        self.reference_hidden_states_ln1 = hidden_states_ln1[-1].clone().detach()  # [T, C]

        
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的AttnOut, MLPOut作为reference
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 奇数chunk：基于Key相似度的选择性重计算 ==========
    else:
        cache2 = STC_CACHE()
        update_token_ratio = cache2.update_token_ratio  
             
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        cfg = get_config().cache
        current_layer_idx = getattr(self, "_stc_layer_idx", -1)
        shallow_reuse_enabled = cfg.shallow_layer_reuse_enabled and cfg.shallow_layer_count > 0
        use_reused_indices = (
            shallow_reuse_enabled
            and current_layer_idx >= cfg.shallow_layer_count
            and getattr(cache2, "shared_update_indices_chunk", -1) == chunk_idx
            and getattr(cache2, "shared_update_indices", None) is not None
        )

        similarity = None
        if use_reused_indices:
            update_indices = cache2.shared_update_indices
            num_update = update_indices.shape[1]
        else:
            # ========== 阶段1：基于Key识别需要更新的token ==========
            # 计算当前帧的Key向量（用于相似度计算）
            key_states_full = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
            
            # Reference frame的Key
            ref_key_for_sim = self.reference_frame_key  # [T, C]
            # 计算cosine相似度：[F, T, C] vs [T, C] -> [F, T]
            similarity = torch.nn.functional.cosine_similarity(
                key_states_full,
                ref_key_for_sim.unsqueeze(0),
                dim=-1
            )

            num_update = int(seq_len * update_token_ratio)
            num_update = max(1, min(num_update, seq_len))
            
            # 对每一帧，选择相似度最低的token索引
            update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]

        if cfg.importance_target_layers.strip():
            target_layers = {
                int(x.strip()) for x in cfg.importance_target_layers.split(",") if x.strip()
            }
        else:
            target_layers = None

        use_importance_filter = (not use_reused_indices) and cfg.importance_filter_enabled and (
            target_layers is None or current_layer_idx in target_layers
        )
        importance_scores = None
        num_update_before_importance = num_update
        if use_importance_filter:
            update_indices, importance_scores = _apply_importance_filter(
                hidden_states_ln1=hidden_states_ln1,
                prev_layer_hidden_states=hidden_states,
                update_indices=update_indices,
                importance_keep_ratio=cfg.importance_keep_ratio,
            )
            update_indices = torch.sort(update_indices, dim=1).values
            # 重要性筛选后，实际更新 token 数会变化；后续 reshape/scatter 必须使用最新长度
            num_update = update_indices.shape[1]
        _record_cache_metrics(
            layer_idx=current_layer_idx,
            chunk_idx=chunk_idx,
            similarity=similarity,
            update_indices=update_indices,
            importance_scores=importance_scores,
            num_update_before_importance=num_update_before_importance,
            num_update_after_importance=num_update,
        )
        if (
            shallow_reuse_enabled
            and current_layer_idx == cfg.shallow_layer_count - 1
        ):
            cache2.shared_update_indices = update_indices.detach()
            cache2.shared_update_indices_chunk = chunk_idx

    # ========== 阶段2：只为选定token计算Q和V ==========
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 提取需要更新的token的特征
        update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
        rank = dist.get_rank()
        
        if rank == 0:
    
            logger.info(f"SigLIP | Vocab size: 729, Tokens to update: {tokens_to_update.shape[1]}")
        # 只为这些token计算Q和V
        query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
        value_selected = v_proj(tokens_to_update)  # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        value_selected = value_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段3：更新V矩阵（Scatter Update） ==========
        # 从reference初始化完整的V矩阵
        value_states_full = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        value_states_full = value_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # Scatter: 将V_selected更新到对应位置
        update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, num_update, head_dim
        )  # [F, num_heads, num_update, head_dim]
        value_states_full.scatter_(2, update_idx_for_scatter, value_selected)
        
        # ========== 阶段4：完整计算K矩阵（全部使用新的） ==========
        key_states_full = k_proj(hidden_states_ln1)  # [F, T, C]
        key_states_full = key_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # ========== 阶段5：部分计算注意力 ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_full,
            value_states=value_states_full,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # ========== 阶段6：Scatter Update到缓存的Attention输出 ==========
        # 从reference初始化
        attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # Scatter更新选定token的输出
        attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
        
        # Residual connection
        hidden_states = residual1 + attn_output_full
        
        # ========== 阶段7：选择性MLP计算 ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 从reference初始化MLP输出
        mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # 提取需要更新的token（从ln2之后的特征）
        ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只计算选定token的MLP
        mlp_selected = self.mlp(ln2_tokens_to_update)  # [F, num_update, C]
        
        # Scatter更新
        mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
        
        # Residual connection
        hidden_states = residual2 + mlp_output_full

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 只计算了部分token的attention
        
        return outputs
    
def new_siglip_sdpa_attn_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    q_len=query_states.shape[-2]
    batch_size=query_states.shape[0]
    
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = False 

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

    attn_output = self.self_attn.out_proj(attn_output)
    return attn_output, None
def forward_with_selective_recompute(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):
    """
    选择性重计算cache策略：
    - 奇数chunk：完整计算，保存最后一帧的K, V, AttnOut, MLPOut
    - 偶数chunk：基于Value相似度选择变化最剧烈的token，只为这些token计算Q和Attention
    
    Args:
        hidden_states: [F, T, C] - F帧，T个token，C个通道
    """
    
    cache2 = STC_CACHE()
    chunk_idx = cache2.chunk_idx
    F, T, C = hidden_states.shape
    is_even_chunk = (chunk_idx % 4 == 0)
    # print("chunk_idx",chunk_idx)
    # ========== 奇数chunk：完整计算并保存reference frame ==========
    if is_even_chunk :
        # 标准的Transformer层计算
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        o_proj = self.self_attn.out_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
            # 保存最后一帧的K, V, AttnOut, MLPOut作为reference
        self.reference_frame_value = value_states[-1].clone().detach()  # [num_heads, T, head_dim]
        #=====================================================================================
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        #=====================================================================================================
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的K, V, AttnOut, MLPOut作为reference
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 偶数chunk：选择性重计算 ==========
    else:
        cache2 = STC_CACHE()
        update_token_ratio = cache2.update_token_ratio  
             
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # ========== 阶段1：识别需要更新的token ==========
        with torch.no_grad():
            # 计算当前帧的Value向量（用于相似度计算）
            value_states_full = self.self_attn.v_proj(hidden_states_ln1)  # [F, T, C]
            
            # 平均所有heads用于相似度计算
            value_for_sim = value_states_full  # [F, T, C]
            
            # Reference frame的value
            ref_value_for_sim = self.reference_frame_value  # [T, head_dim]
            # 计算cosine相似度：[F, T, head_dim] vs [T, head_dim] -> [F, T]
            similarity = torch.nn.functional.cosine_similarity(
                value_for_sim,
                ref_value_for_sim.unsqueeze(0),
                dim=-1
            )

            num_update = int(seq_len * update_token_ratio)
            num_update = max(1, min(num_update, seq_len))
            
            # 对每一帧，选择相似度最低的token索引 I_update
            update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
        
        # ========== 阶段2：只为选定token计算Q和K ==========
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        o_proj = self.self_attn.out_proj
        
        # 提取需要更新的token的特征
        update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
        print("[DEBUG ]tokens_to_update",tokens_to_update.shape)
        
        # 只为这些token计算Q和K
        query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
        key_selected = k_proj(tokens_to_update)    # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        key_selected = key_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段3：更新K矩阵（Scatter Update） ==========
        # 从reference初始化完整的K矩阵 Kr
        key_states_full = self.reference_frame_key.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()  # [F, num_heads, T, head_dim]
        
        # Scatter: 将K_selected更新到Kr的对应位置
        update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, num_update, head_dim
        )  # [F, num_heads, num_update, head_dim]
        key_states_full.scatter_(2, update_idx_for_scatter, key_selected)  # K_updated
        
        # ========== 阶段4：完整计算V矩阵（全部使用新的） ==========
        
        value_states_full = value_states_full.view(batch_size, key_states_full.shape[2], num_heads, head_dim).transpose(1, 2)
        # ========== 阶段5：部分计算注意力 ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_full,
            value_states=value_states_full,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        # ========== 阶段6：Scatter Update到缓存的Attention输出 ==========
        # 从reference初始化
        attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # Scatter更新选定token的输出
        attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
        
        # Residual connection
        hidden_states = residual1 + attn_output_full
        
        # ========== 阶段7：选择性MLP计算 ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 从reference初始化MLP输出
        mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # 提取需要更新的token（注意：是从ln2之后的特征中提取）
        ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只计算选定token的MLP
        mlp_selected = self.mlp(ln2_tokens_to_update)  # [F, num_update, C]
        
        # Scatter更新
        mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
        
        # Residual connection
        hidden_states = residual2 + mlp_output_full

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 因为我们只计算了部分token的attention
        
        return outputs

def siglip_sdpa_attn_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    q_len=query_states.shape[-2]
    batch_size=query_states.shape[0]
    
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = False 

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

    attn_output = self.self_attn.out_proj(attn_output)
    
    return attn_output, None
def forward_with_selective_key_recompute_clip(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
):
    """
    选择性重计算cache策略（K/V操作互换后）：
    - 偶数chunk：完整计算，保存最后一帧的K, V, AttnOut, MLPOut
    - 奇数chunk：基于Key相似度选择变化最剧烈的token，只为这些token计算Q和V
    
    Args:
        hidden_states: [F, T, C] - F帧，T个token，C个通道
    """
    
    cache2 = STC_CACHE()
    chunk_idx = cache2.chunk_idx
    is_even_chunk = (chunk_idx % 2 == 0)
    
    # ========== 偶数chunk：完整计算并保存reference frame ==========
    if is_even_chunk :
        # 标准的Transformer层计算
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        o_proj = self.self_attn.out_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V, AttnOut, MLPOut作为reference
        # 注意：保存的是projection后的张量，shape为[T, C]
        
        self.reference_frame_key = key_states[-1].clone().detach()  # [T, C]
        self.reference_frame_value = value_states[-1].clone().detach()  # [T, C]  # 修复这里！
        self.reference_hidden_states_ln1 = hidden_states_ln1[-1].clone().detach()  # [T, C]

        
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的AttnOut, MLPOut作为reference
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 奇数chunk：基于Key相似度的选择性重计算 ==========
    else:
        cache2 = STC_CACHE()
        update_token_ratio = cache2.update_token_ratio  
             
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # ========== 阶段1：基于Key识别需要更新的token ==========
        # 计算当前帧的Key向量（用于相似度计算）
        key_states_full = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
        
        # Reference frame的Key
        ref_key_for_sim = self.reference_frame_key  # [T, C]
        # 计算cosine相似度：[F, T, C] vs [T, C] -> [F, T]
        similarity = torch.nn.functional.cosine_similarity(
            key_states_full,
            ref_key_for_sim.unsqueeze(0),
            dim=-1
        )

        num_update = int(seq_len * update_token_ratio)
        num_update = max(1, min(num_update, seq_len))
        
        # 对每一帧，选择相似度最低的token索引
        update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
        cfg = get_config().cache
        current_layer_idx = getattr(self, "_stc_layer_idx", -1)
        if cfg.importance_target_layers.strip():
            target_layers = {
                int(x.strip()) for x in cfg.importance_target_layers.split(",") if x.strip()
            }
        else:
            target_layers = None

        use_importance_filter = cfg.importance_filter_enabled and (
            target_layers is None or current_layer_idx in target_layers
        )
        importance_scores = None
        num_update_before_importance = num_update
        if use_importance_filter:
            update_indices, importance_scores = _apply_importance_filter(
                hidden_states_ln1=hidden_states_ln1,
                prev_layer_hidden_states=hidden_states,
                update_indices=update_indices,
                importance_keep_ratio=cfg.importance_keep_ratio,
            )
            update_indices = torch.sort(update_indices, dim=1).values
            num_update = update_indices.shape[1]
        _record_cache_metrics(
            layer_idx=current_layer_idx,
            chunk_idx=chunk_idx,
            similarity=similarity,
            update_indices=update_indices,
            importance_scores=importance_scores,
            num_update_before_importance=num_update_before_importance,
            num_update_after_importance=num_update,
        )
    
        # ========== 阶段2：只为选定token计算Q和V ==========
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 提取需要更新的token的特征
        update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只为这些token计算Q和V
        query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
        value_selected = v_proj(tokens_to_update)  # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        value_selected = value_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段3：更新V矩阵（Scatter Update） ==========
        # 从reference初始化完整的V矩阵
        value_states_full = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        value_states_full = value_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # Scatter: 将V_selected更新到对应位置
        update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, num_update, head_dim
        )  # [F, num_heads, num_update, head_dim]
        value_states_full.scatter_(2, update_idx_for_scatter, value_selected)
        
        # ========== 阶段4：完整计算K矩阵（全部使用新的） ==========
        key_states_full = k_proj(hidden_states_ln1)  # [F, T, C]
        key_states_full = key_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # ========== 阶段5：部分计算注意力 ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_full,
            value_states=value_states_full,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # ========== 阶段6：Scatter Update到缓存的Attention输出 ==========
        # 从reference初始化
        attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # Scatter更新选定token的输出
        attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
        
        # Residual connection
        hidden_states = residual1 + attn_output_full
        
        # ========== 阶段7：选择性MLP计算 ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 从reference初始化MLP输出
        mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # 提取需要更新的token（从ln2之后的特征）
        ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只计算选定token的MLP
        mlp_selected = self.mlp(ln2_tokens_to_update)  # [F, num_update, C]
        
        # Scatter更新
        mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
        
        # Residual connection
        hidden_states = residual2 + mlp_output_full

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 只计算了部分token的attention
        
        return outputs
