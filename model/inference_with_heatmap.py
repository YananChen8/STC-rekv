#!/usr/bin/env python3
"""
推理后热力图生成示例

这个脚本展示如何在推理完成后自动生成热力图。
可以集成到你的推理脚本中。
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.custom_siglip import visualize_heatmaps_from_logs
from logzero import logger


def generate_heatmaps_after_inference(
    log_dir="./ACR/heatmap_logs",
    output_dir="./ACR/heatmap_visualizations",
    skip_on_empty=True
):
    """
    推理后自动生成热力图
    
    参数:
        log_dir: 热力图日志目录
        output_dir: 热力图输出目录
        skip_on_empty: 如果日志目录为空是否跳过
    
    返回:
        True if successful, False otherwise
    """
    
    # 检查日志目录
    if not os.path.exists(log_dir):
        logger.warning(f"Heatmap log directory not found: {log_dir}")
        return False
    
    # 检查是否有日志文件
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.pt')]
    if not log_files:
        if skip_on_empty:
            logger.warning(f"No heatmap logs found in {log_dir}, skipping visualization")
            return False
        else:
            logger.error(f"No heatmap logs found in {log_dir}")
            return False
    
    logger.info(f"Found {len(log_files)} heatmap logs, generating visualizations...")
    
    try:
        visualize_heatmaps_from_logs(log_dir=log_dir, output_dir=output_dir)
        logger.info(f"Heatmap visualizations saved to {output_dir}")
        
        # 统计生成的文件
        if os.path.exists(output_dir):
            png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            logger.info(f"Generated {len(png_files)} visualization files")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to generate heatmaps: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 集成示例
# ============================================================================

def inference_with_heatmap_generation():
    """
    推理 + 热力图生成的完整流程示例
    """
    
    logger.info("=" * 60)
    logger.info("Starting inference with heatmap visualization")
    logger.info("=" * 60)
    
    # ========== 推理阶段 ==========
    # 这里替换为实际的推理代码
    logger.info("\n[Phase 1] Running inference...")
    try:
        # 你的推理代码在这里
        # inference_model(args)
        logger.info("Inference completed successfully")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return False
    
    # ========== 热力图生成阶段 ==========
    logger.info("\n[Phase 2] Generating heatmap visualizations...")
    success = generate_heatmaps_after_inference(
        log_dir="./ACR/heatmap_logs",
        output_dir="./ACR/heatmap_visualizations",
        skip_on_empty=False  # 如果没有日志则报错
    )
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("All tasks completed successfully!")
        logger.info("Heatmaps saved to: ./ACR/heatmap_visualizations/")
        logger.info("=" * 60)
    else:
        logger.warning("\n" + "=" * 60)
        logger.warning("Inference completed but heatmap generation failed")
        logger.warning("=" * 60)
    
    return success


if __name__ == "__main__":
    # 直接生成热力图（假设已有推理日志）
    logger.info("Post-inference heatmap generation mode")
    success = generate_heatmaps_after_inference()
    
    if not success:
        logger.info("\nAlternatively, run inference_with_heatmap_generation() to:")
        logger.info("  1. Run complete inference")
        logger.info("  2. Automatically generate heatmaps after inference")
