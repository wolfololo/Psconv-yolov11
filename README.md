# PSconv-YOLO11

基于 Ultralytics YOLO11 的海面目标检测模型，在 YOLO11n 上集成 **PSConv、CBAM、FEM 与 BiFPN 加权融合**。

模型配置文件：[yolo11n_psconv_cbam_fem_bifpn_aligned.yaml](ultralytics/cfg/models/11/yolo11n_psconv_cbam_fem_bifpn_aligned.yaml)

## 模型结构

| 位置 | 模块 | 说明 |
| --- | --- | --- |
| Backbone 下采样 | PSConv | 风车形非对称填充卷积，替换 P1–P5 的下采样卷积 |
| Backbone P2 / P3 | CBAM | 通道 + 空间注意力，插入在 C3k2 之后 |
| Neck 输入端 | FEM | 四分支特征增强模块，作用于 P3 / P4 / P5 |
| Neck 融合 | BiFPN | 通过 `Fusion(fusion='bifpn')` 实现可学习权重的加权融合 |
| Head | Detect | P3 / P4 / P5 三尺度检测头 |

模型复杂度（`imgsz=640`，由 `model.info()` 实测）：

| 参数量 | GFLOPs | 层数 |
| --- | --- | --- |
| 1,928,583 | 6.7 | 593 |

> 权重文件与训练结果不随仓库提供，可按下方步骤自行训练，或从 Releases 下载。

## 目录结构

```text
.
├── ultralytics/                 # 修改版 Ultralytics（新增自定义模块与注册逻辑）
│   ├── cfg/models/11/yolo11n_psconv_cbam_fem_bifpn_aligned.yaml
│   └── nn/
│       ├── modules/fem.py       # FEM
│       ├── modules/conv.py      # CBAM
│       └── extra_modules/block.py  # PSConv、Fusion(BiFPN)
├── train.py                     # 训练脚本
├── val.py                       # 验证脚本（输出参数量/FLOPs/mAP 表格）
├── sea.yaml                     # 数据集配置
├── requirements.txt
└── LICENSE
```

代码中新增模块在 [tasks.py](ultralytics/nn/tasks.py) 的 `parse_model` 中注册；`Fusion` 的融合方式由模型 YAML 顶层的 `fusion_mode` 键解析（当前为 `bifpn`）。

## 环境安装

- Python >= 3.8，PyTorch >= 2.0（建议使用 GPU）
- 在仓库根目录执行：

```bash
pip install -r requirements.txt
```

> 必须从仓库根目录运行脚本，以保证 `ultralytics` 使用本仓库中的修改版而不是 pip 安装的官方版本。

## 数据集准备

`sea.yaml` 定义了 5 个类别：`swimmer`、`boat`、`jetski`、`life_saving_appliances`、`buoy`。

数据集不随仓库提供，请按如下结构放置（目录名需与 `sea.yaml` 中的相对路径一致）：

```text
sea/
├── train/
│   ├── images/   # *.jpg / *.png
│   └── labels/   # 同名 *.txt（YOLO 格式：cls cx cy w h，均归一化到 0-1）
└── val/
    ├── images/
    └── labels/
```

`sea.yaml` 内容：

```yaml
train: ./sea/train/images
val: ./sea/val/images

nc: 5
names: ['swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy']
```

## 训练

```bash
python train.py
```

默认参数：`imgsz=640`、`epochs=300`、`batch=16`、`Adam`、`close_mosaic=10`，输出保存在 `runs/sea/exp/`。
需要加载官方预训练权重时，可在 `train.py` 中 `YOLO(...)` 之后加入：

```python
model.load('yolo11n.pt')
```

Windows 下若训练卡住，将 `workers` 改为 `0`。

## 验证

```bash
python val.py
```

脚本会输出每类 Precision / Recall / F1 / mAP50 / mAP75 / mAP50-95，以及参数量、GFLOPs、FPS 和模型文件大小，并保存到 `runs/valsea/exp/paper_data.txt`。

## 推理与导出

```python
from ultralytics import YOLO

model = YOLO('runs/sea/exp/weights/best.pt')
model.predict(source='path/to/images', imgsz=640, conf=0.4, save=True)
model.export(format='onnx', opset=13, simplify=True)
```

PSConv、FEM、CBAM 均为常规卷积/注意力算子，不包含自定义 CUDA 扩展，可直接导出 ONNX（导出前安装 `pip install onnx onnxslim onnxruntime`）。

## 致谢

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)：检测框架与基础模块。
- PSConv 参考 *Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection*。
- 其余对比模块来自开源社区的 YOLO 改进仓库，已在代码中保留对应来源注释。

## 许可证

本项目基于 Ultralytics 修改，遵循 [AGPL-3.0](LICENSE) 许可证。
