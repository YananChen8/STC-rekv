"""
全局配置管理器 - 优雅地管理所有实验配置
"""
from dataclasses import dataclass, field
from typing import Optional, Literal

import os
@dataclass
class CacheConfig:
    """缓存相关配置"""
    strategy: Literal['none', 'cacher'] = 'cacher'
    update_token_ratio: float = 0.25
    cache_interval=2
    importance_filter_enabled: bool = True
    importance_keep_ratio: float = 0.5
    importance_target_layers: str = "3,17,22"
    metrics_dump_enabled: bool = False
    metrics_dump_path: str = ""
    metrics_dump_max_records: int = 500
    

        

@dataclass
class ModelConfig:
    """模型相关配置"""
    token_per_frame: int = 60
    prune_strategy: str = 'full_tokens'
    encode_chunk_size: int = 1
        




@dataclass
class GlobalConfig:
    """全局配置单例"""
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    _instance: Optional['GlobalConfig'] = None
    
    @classmethod
    def get_instance(cls) -> 'GlobalConfig':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize_from_args(cls, args):
        instance = cls.get_instance()
        if hasattr(args, "cache_strategy"):
            instance.cache.strategy = args.cache_strategy
        if hasattr(args, "update_token_ratio"):
            instance.cache.update_token_ratio = args.update_token_ratio
        if hasattr(args, "cache_interval"):
            instance.cache.cache_interval = args.cache_interval
        if hasattr(args, "importance_filter_enabled"):
            instance.cache.importance_filter_enabled = args.importance_filter_enabled
        if hasattr(args, "importance_keep_ratio"):
            instance.cache.importance_keep_ratio = args.importance_keep_ratio
        if hasattr(args, "importance_target_layers"):
            instance.cache.importance_target_layers = args.importance_target_layers
        if hasattr(args, "metrics_dump_enabled"):
            instance.cache.metrics_dump_enabled = args.metrics_dump_enabled
        if hasattr(args, "metrics_dump_path"):
            instance.cache.metrics_dump_path = args.metrics_dump_path
        if hasattr(args, "metrics_dump_max_records"):
            instance.cache.metrics_dump_max_records = args.metrics_dump_max_records
        if hasattr(args, "token_per_frame"):
            instance.model.token_per_frame = args.token_per_frame
        if hasattr(args, "prune_strategy"):
            instance.model.prune_strategy = args.prune_strategy
        if hasattr(args, "retrieve_chunk_size"):
            instance.model.encode_chunk_size = args.retrieve_chunk_size

        return instance
    
    def to_dict(self):
        return {
            'cache': {
                'strategy': self.cache.strategy,
                'update_token_ratio': self.cache.update_token_ratio,
                'cache_interval': self.cache.cache_interval,
                'importance_filter_enabled': self.cache.importance_filter_enabled,
                'importance_keep_ratio': self.cache.importance_keep_ratio,
                'importance_target_layers': self.cache.importance_target_layers,
                'metrics_dump_enabled': self.cache.metrics_dump_enabled,
                'metrics_dump_path': self.cache.metrics_dump_path,
                'metrics_dump_max_records': self.cache.metrics_dump_max_records,
            },
            'model': {
                'token_per_frame': self.model.token_per_frame,
                'prune_strategy': self.model.prune_strategy,
                'encode_chunk_size': self.model.encode_chunk_size,
                
            }
        }
    
    def __str__(self):
        import json
        return json.dumps(self.to_dict(), indent=2)


# 便捷访问函数
def get_config() -> GlobalConfig:
    return GlobalConfig.get_instance()
