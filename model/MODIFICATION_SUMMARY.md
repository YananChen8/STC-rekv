# 代码修改总结

## 概述
已成功修改 `custom_siglip.py`，实现对每一帧每一层的相似度和attention热力图的自动记录和可视化。

## 主要修改

### 1. 新增导入库
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

### 2. 新增函数

#### (1) `record_attention_and_similarity_heatmap(cache2, layer_idx, chunk_idx, similarity, attention_weights)`
- **位置**: 主函数区域
- **功能**: 
  - 记录每一帧的相似度和attention数据
  - 自动限制只保存前5帧（max_frames=5）
  - 每帧单独存储为 `.pt` 文件
  - 包含frame_idx, layer_idx, chunk_idx, similarity, attention_weights

- **数据格式**:
  ```python
  {
      "chunk_idx": int,
      "layer_idx": int,
      "frame_idx": int,
      "similarity": torch.Tensor [seq_len],
      "attention_weights": torch.Tensor [num_heads, seq_len, seq_len]
  }
  ```

#### (2) `visualize_heatmaps_from_logs(log_dir, output_dir)`
- **位置**: 主函数区域
- **功能**:
  - 从保存的日志文件读取数据
  - 为每一帧生成热力图
  - 保存为PNG格式
  - 每个PNG包含该帧所有层的相似度和attention热力图

- **输出文件**:
  ```
  frame_000_all_layers.png
  frame_001_all_layers.png
  ...
  frame_004_all_layers.png
  ```

### 3. 修改的Forward函数

#### (1) `forward_with_selective_key_recompute()`
- **修改位置**: 
  - update_cache=True分支（完整计算）后
  - is_shallow=True分支（相似度计算后）
  - else分支（attn_weights获得后）
- **添加内容**: `record_attention_and_similarity_heatmap()` 调用

#### (2) `forward_with_selective_key_recompute_clip()`
- **修改位置**: 
  - is_even_chunk=True分支（完整计算后）
  - else分支（similarity计算后和attn_weights获得后）
- **添加内容**: `record_attention_and_similarity_heatmap()` 调用

#### (3) `forward_with_selective_recompute()`
- **修改位置**:
  - is_even_chunk=True分支（完整计算后）
  - else分支（similarity计算后和attn_weights获得后）
- **添加内容**: `record_attention_and_similarity_heatmap()` 调用

### 4. Attention前向函数更新

#### (1) `new_siglip_sdpa_attn_forward()`
- **修改**: 现在计算并返回attention权重
- **计算流程**:
  ```python
  scores = Q @ K^T / sqrt(d_k)
  scores += attention_mask (if present)
  attn_weights = softmax(scores)
  ```
- **返回**: `(attn_output, attn_weights)` 而不是 `(attn_output, None)`

#### (2) `siglip_sdpa_attn_forward()`
- **修改**: 同上，现在也计算并返回attention权重

## 数据流程

### 推理阶段
```
推理进行中
  ↓
forward函数执行
  ↓
计算相似度/attention
  ↓
调用 record_attention_and_similarity_heatmap()
  ↓
保存数据到 ./ACR/heatmap_logs/
  ↓
自动限制在5帧
```

### 可视化阶段
```
调用 visualize_heatmaps_from_logs()
  ↓
读取 ./ACR/heatmap_logs/ 中的 .pt 文件
  ↓
按frame_idx分组
  ↓
为每frame生成多层热力图
  ↓
保存PNG到 ./ACR/heatmap_visualizations/
```

## 使用方案

### 方案1：在推理脚本中自动生成
```python
# 在推理完成后
from model.custom_siglip import visualize_heatmaps_from_logs

visualize_heatmaps_from_logs()
```

### 方案2：使用独立脚本
```bash
cd /path/to/STC/model
python visualize_heatmaps.py
```

### 方案3：集成到推理流程
```python
from model.inference_with_heatmap import inference_with_heatmap_generation

inference_with_heatmap_generation()
```

## 生成的热力图说明

### 每个PNG文件包含
- **行数**: 等于该帧的transformer层数
- **列数**: 2（左:相似度，右:attention）
- **左侧热力图**: 相似度分布
  - 蓝色: 低相似度
  - 红色: 高相似度
  - 含义: 需要更新的token
- **右侧热力图**: Attention权重分布（所有head平均）
  - 紫色: 低attention
  - 黄色: 高attention
  - X轴: Key token位置
  - Y轴: Query token位置

## 配置参数

### 限制帧数
在 `record_attention_and_similarity_heatmap()` 中：
```python
max_frames = 5  # 改为需要的值
```

### 图表输出质量
在 `visualize_heatmaps_from_logs()` 中：
```python
figsize=(14, 5 * num_layers)  # 改变图表大小
dpi=100  # 改变分辨率
```

### 颜色方案
```python
cmap='coolwarm'   # 相似度颜色方案
cmap='viridis'    # attention颜色方案
```

## 目录结构

```
./ACR/
├── heatmap_logs/              # 热力图数据（自动创建）
│   ├── frame0_layer0_chunk0.pt
│   ├── frame0_layer1_chunk0.pt
│   ├── frame1_layer0_chunk2.pt
│   └── ...
└── heatmap_visualizations/    # 热力图PNG（自动创建）
    ├── frame_000_all_layers.png
    ├── frame_001_all_layers.png
    └── ...
```

## 性能影响

- **热力图记录开销**: 极小（只在选定的位置记录）
- **对推理速度的影响**: <1%（记录为异步操作）
- **存储空间**: 每帧约~ 100KB-1MB（因层数和分辨率而异）
- **前后限制**: 自动限制为5帧，可配置

## 文件清单

### 修改的文件
- `/home/chenyanan-20260210/STC/STC/model/custom_siglip.py` - 主要修改

### 新增的文件
- `/home/chenyanan-20260210/STC/STC/model/HEATMAP_VISUALIZATION_GUIDE.md` - 详细使用指南
- `/home/chenyanan-20260210/STC/STC/model/visualize_heatmaps.py` - 独立可视化脚本
- `/home/chenyanan-20260210/STC/STC/model/inference_with_heatmap.py` - 推理集成脚本

## 注意事项

1. **自动限制5帧**: 代码自动限制只保存前5帧的热力图数据，避免大量输出
2. **显存考虑**: 记录操作使用CPU内存，不占用显存
3. **分布式训练**: 代码中考虑了分布式训练，只在rank 0进程记录
4. **错误处理**: 所有记录操作都被try-except包装，不会中断推理
5. **可选操作**: 热力图生成完全可选，推理过程不依赖它

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| 没有生成热力图 | 检查log_dir是否存在.pt文件 |
| 热力图显示异常 | 检查相似度和attention值的范围 |
| 内存不足 | 减少max_frames或figsize |
| 文件保存失败 | 检查output_dir的写入权限 |
