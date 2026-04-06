# 快速开始指南 - 热力图可视化

## 💡 一句话总结
在你的推理脚本中，只需在推理完成后加一行代码即可自动生成热力图！

## 🚀 快速开始 (3步)

### 步骤1: 运行推理
```bash
# 运行你的推理脚本（代码已自动记录前5帧的热力图数据）
python your_inference_script.py
```

### 步骤2: 生成可视化
```bash
# 方案A: 运行可视化脚本
cd /path/to/STC/model
python visualize_heatmaps.py

# 或 方案B: 在推理脚本中添加一行代码
from model.custom_siglip import visualize_heatmaps_from_logs
visualize_heatmaps_from_logs()
```

### 步骤3: 查看结果
```bash
# 打开生成的热力图
open ./ACR/heatmap_visualizations/frame_000_all_layers.png
```

## 📊 输出文件说明

```
./ACR/heatmap_visualizations/
├── frame_000_all_layers.png  # 第0帧，所有层
├── frame_001_all_layers.png  # 第1帧，所有层
├── frame_002_all_layers.png  # 第2帧，所有层
├── frame_003_all_layers.png  # 第3帧，所有层
└── frame_004_all_layers.png  # 第4帧，所有层
```

每个PNG包含：
- **左列**: 相似度热力图（蓝≈低相似度, 红≈高相似度）
- **右列**: 注意力热力图（紫≈低attention, 黄≈高attention）
- **每行**: 一个Transformer层的热力图

## 🔧 常见命令

### 只生成热力图（不运行推理）
```python
from model.custom_siglip import visualize_heatmaps_from_logs

# 使用默认路径
visualize_heatmaps_from_logs()

# 使用自定义路径
visualize_heatmaps_from_logs(
    log_dir="./ACR/heatmap_logs",
    output_dir="./ACR/heatmap_visualizations"
)
```

### 在推理脚本中集成
```python
# 推理完成后
from model.custom_siglip import visualize_heatmaps_from_logs

print("Running inference...")
# 你的推理代码...

print("Generating heatmaps...")
visualize_heatmaps_from_logs()

print("Done! Check ./ACR/heatmap_visualizations/")
```

### 查看所有生成的文件
```bash
ls -lh ./ACR/heatmap_visualizations/
```

## ⚙️ 自定义配置

### 改变保存的帧数（默认5帧）
编辑 `custom_siglip.py` 中的 `record_attention_and_similarity_heatmap()` 函数：
```python
max_frames = 10  # 改为10帧
```

### 改变热力图大小
编辑 `custom_siglip.py` 中的 `visualize_heatmaps_from_logs()` 函数：
```python
figsize=(14, 5 * num_layers)  # (宽度, 高度)
```

### 改变保存质量
```python
plt.savefig(output_path, dpi=200, bbox_inches='tight')  # dpi越大质量越好
```

### 改变颜色方案
```python
# 相似度 - 改为其他如 'RdYlBu', 'bwr'
sns.heatmap(sim_2d, cmap='coolwarm', ...)

# attention - 改为其他如 'hot', 'bone'  
sns.heatmap(attention_mean, cmap='viridis', ...)
```

## 🎯 热力图解读示例

### 相似度热力图（左）
```
蓝色 ← 与参考帧不相似 → 红色
                ↓
           需要更新的token（系统优先计算这些token）
```

### 注意力热力图（右）
```
查询token → 该token关注哪些Key token
           ↓
颜色越亮 ← 注意力越强
```

## 💾 存储空间

- 每帧热力图日志: ~100-500KB
- 生成的PNG: ~50-200KB
- 总计（5帧）: ~500KB-2MB

## ⚡ 性能

| 指标 | 值 |
|-----|-----|
| 热力图记录开销 | <1% 推理时间 |
| 内存占用 | ~50MB |
| 可视化生成时间 | 30-60秒（5帧，所有层）|

## ❌ 常见问题

### Q: 没有生成热力图？
A: 检查是否有 `/ACR/heatmap_logs/*.pt` 文件，如果没有说明推理未正常记录。

### Q: 热力图显示很奇怪？
A: 可能是数据问题。检查日志文件中的相似度和attention值是否在合理范围内。

### Q: 如何只保存某些层的热力图？
A: 修改 `visualize_heatmaps_from_logs()` 中的循环逻辑来过滤层索引。

### Q: 能否处理所有帧而不是5帧？
A: 修改 `record_attention_and_similarity_heatmap()` 中的 `max_frames` 参数。

## 📚 详细文档

- **详细使用指南**: [HEATMAP_VISUALIZATION_GUIDE.md](./HEATMAP_VISUALIZATION_GUIDE.md)
- **修改总结**: [MODIFICATION_SUMMARY.md](./MODIFICATION_SUMMARY.md)
- **独立脚本**: [visualize_heatmaps.py](./visualize_heatmaps.py)
- **集成示例**: [inference_with_heatmap.py](./inference_with_heatmap.py)

## 📞 支持

如遇到问题：
1. 查看 `./ACR/heatmap_logs/` 目录是否有数据
2. 查看推理脚本的日志输出
3. 检查所有必要的Python包是否已安装（matplotlib, seaborn）
