#!/usr/bin/env python3
"""
热力图可视化示例脚本

这个脚本展示如何使用新的热力图可视化功能。

使用方式：
    1. 运行完整个推理后，调用此脚本
    2. 或在推理脚本中直接调用 visualize_heatmaps_from_logs()
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.custom_siglip import visualize_heatmaps_from_logs
from model.custom_siglip import visualize_similarity_grouped_by_chunk
from model.custom_siglip import visualize_attention_token_scores_grouped_by_chunk


def main():
    """主函数"""

    # visualize_similarity_grouped_by_chunk(
    #     log_dir="/home/chenyanan-20260210/STC/STC/model/online_bench_inference/ovobench/EPM/heatmap_logs",
    #     output_dir="/home/chenyanan-20260210/STC/STC/model/online_bench_inference/ovobench/EPM/heatmap_logs/heatmap_visualizations_grouped",
    #     target_chunk_idx=5,
    # )

    visualize_attention_token_scores_grouped_by_chunk(
        log_dir="/home/chenyanan-20260210/STC/STC/model/online_bench_inference/ovobench/EPM/heatmap_logs",
        output_dir="/home/chenyanan-20260210/STC/STC/model/online_bench_inference/ovobench/EPM/heatmap_logs/heatmap_visualizations_grouped",
        target_chunk_idx=4,
        attention_reduce="sum_over_query",
    )

    # # 默认日志和输出目录
    # log_dir = "/home/chenyanan-20260210/STC/STC/model/online_bench_inference/ovobench/EPM/heatmap_logs"
    # output_dir = "/home/chenyanan-20260210/STC/STC/model/online_bench_inference/ovobench/EPM/heatmap_logs/heatmap_visualizations"
    
    # print("=" * 60)
    # print("STC Heatmap Visualization Tool")
    # print("=" * 60)
    
    # # 检查日志目录是否存在
    # if not os.path.exists(log_dir):
    #     print(f"\n[ERROR] Log directory not found: {log_dir}")
    #     print("Please run inference first to generate heatmap logs.")
    #     return
    
    # # 检查日志目录中是否有文件
    # log_files = [f for f in os.listdir(log_dir) if f.endswith('.pt')]
    # if not log_files:
    #     print(f"\n[ERROR] No heatmap logs found in {log_dir}")
    #     print("Please run inference with heatmap recording enabled.")
    #     return
    
    # print(f"\n[INFO] Found {len(log_files)} heatmap log files")
    # print(f"[INFO] Log directory: {log_dir}")
    # print(f"[INFO] Output directory: {output_dir}")
    
    # # 生成可视化
    # print(f"\n[INFO] Generating heatmap visualizations...")
    # try:
    #     visualize_heatmaps_from_logs(log_dir=log_dir, output_dir=output_dir)
    #     print(f"\n[SUCCESS] Heatmap visualizations generated successfully!")
    #     print(f"[INFO] Output saved to: {output_dir}")
        
    #     # 列出生成的文件
    #     if os.path.exists(output_dir):
    #         output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    #         if output_files:
    #             print(f"\n[INFO] Generated files:")
    #             for file in output_files:
    #                 file_path = os.path.join(output_dir, file)
    #                 file_size = os.path.getsize(file_path) / 1024  # KB
    #                 print(f"  - {file} ({file_size:.1f} KB)")
    
    # except Exception as e:
    #     print(f"\n[ERROR] Failed to generate visualizations: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return


if __name__ == "__main__":
    main()
