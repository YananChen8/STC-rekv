# 热力图可视化使用指南

本指南说明如何使用修改后的 `custom_siglip.py` 来生成每一帧的相似度和注意力热力图。

## 功能概述

代码现在可以自动记录每一帧每一层的：
1. **相似度热力图** - 显示当前帧的token与参考帧的相似度
2. **注意力热力图** - 显示attention权重分布（对所有attention head求平均）

系统会自动限制只保存前5帧的数据，以避免产生过多的图表。

## 代码修改说明

### 1. 新增导入
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

### 2. 新增函数

#### `record_attention_and_similarity_heatmap()`
- **功能**: 记录每一帧每一层的相似度和attention数据
- **参数**:
  - `cache2`: 缓存对象
  - `layer_idx`: 层索引
  - `chunk_idx`: chunk索引
  - `similarity`: 相似度张量 [batch_size, seq_len]
  - `attention_weights`: attention权重张量 [batch_size, num_heads, seq_len, seq_len] 或 None
- **自动限制**: 最多保存5帧（可修改 `max_frames` 参数）

#### `visualize_heatmaps_from_logs()`
- **功能**: 读取保存的日志文件并生成热力图
- **参数**:
  - `log_dir`: 日志目录（默认: `./ACR/heatmap_logs`）
  - `output_dir`: 输出目录（默认: `./ACR/heatmap_visualizations`）
- **输出**: 为每一帧生成一个包含所有层的热力图组合文件

### 3. 修改的Forward函数

在以下三个函数中都添加了热力图记录：
- `forward_with_selective_key_recompute()`
- `forward_with_selective_key_recompute_clip()`
- `forward_with_selective_recompute()`

热力图记录会在下列位置进行：
- 完整计算阶段（even chunk）结束后
- 选择性重计算阶段（odd chunk）结束后

### 4. Attention权重计算更新

`new_siglip_sdpa_attn_forward()` 和 `siglip_sdpa_attn_forward()` 现在会：
- 手动计算attention权重（scores = Q @ K^T / sqrt(d_k)）
- 应用attention mask
- 计算softmax得到最终权重
- 返回attention权重供可视化使用

## 使用步骤

### 步骤1: 运行推理
正常运行推理脚本。代码会自动在以下目录保存热力图数据：
```
./ACR/heatmap_logs/
```

每个保存的文件格式为：
```
frame{frame_idx}_layer{layer_idx}_chunk{chunk_idx}.pt
```

### 步骤2: 生成可视化
在推理完成后，调用可视化函数生成热力图PNG文件：

```python
from model.custom_siglip import visualize_heatmaps_from_logs

# 使用默认路径
visualize_heatmaps_from_logs()

# 或指定自定义路径
visualize_heatmaps_from_logs(
    log_dir="./ACR/heatmap_logs",
    output_dir="./ACR/heatmap_visualizations"
)
```

### 步骤3: 查看结果
输出的热力图文件会保存到 `./ACR/heatmap_visualizations/`，格式为：
```
frame_000_all_layers.png  # Frame 0, 所有层
frame_001_all_layers.png  # Frame 1, 所有层
frame_002_all_layers.png  # Frame 2, 所有层
frame_003_all_layers.png  # Frame 3, 所有层
frame_004_all_layers.png  # Frame 4, 所有层
```

每个PNG文件包含：
- 左列: 相似度热力图（1D token相似度可视化为2D）
- 右列: 注意力热力图（所有attention head的平均）
- 每行代表一个transformer层

## 图表解读

### 相似度热力图
- **颜色范围**: 从蓝色（低相似度）到红色（高相似度）
- **含义**: 越接近红色的token表示与参考帧更相似（需要更新的token）

### 注意力热力图
- **颜色范围**: 从紫色（低attention）到黄色（高attention）
- **X轴**: Key token位置
- **Y轴**: Query token位置
- **含义**: 显示各个query token对哪些key token的关注程度

## 配置参数

### 限制帧数
在 `record_attention_and_similarity_heatmap()` 中修改：
```python
max_frames = 5  # 改为需要的帧数
```

### 图表分辨率
在 `visualize_heatmaps_from_logs()` 中修改：
```python
fig, axes = plt.subplots(num_layers, 2, figsize=(14, 5 * num_layers))
# 可修改figsize来改变输出图表大小

plt.savefig(output_path, dpi=100, bbox_inches='tight')
# 可修改dpi来改变分辨率
```

### 颜色方案
在 `visualize_heatmaps_from_logs()` 中修改：
```python
sns.heatmap(sim_2d, ax=ax_sim, cmap='coolwarm', cbar=True)  # 改为其他如'RdYlBu'等
sns.heatmap(attention_mean, ax=ax_attn, cmap='viridis', cbar=True)  # 改为其他如'hot'等
```

## 故障排除

### 问题1: 没有生成热力图
- 检查推理是否成功完成
- 确保 `./ACR/heatmap_logs/` 目录中有 `.pt` 文件
- 查看日志中是否有警告信息

### 问题2: 热力图显示异常
- 检查相似度数据的范围是否正常（应该在-1到1之间）
- 检查attention权重是否为有效值（应该在0到1之间）
- 尝试增加图表大小以看清细节

### 问题3: 内存不足
- 减少 `max_frames` 的值
- 减少每个图表的大小（修改figsize参数）
- 逐个帧生成可视化而不是批量生成

## 性能提示

- 热力图记录只会记录前5帧，对总体性能影响最小
- 可视化生成在推理完成后进行，不会影响推理速度
- 如果需要处理大量帧，建议分批生成可视化
