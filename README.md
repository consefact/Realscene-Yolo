## 基于靶标组件拼贴的 YOLO 真实场景数据增强工具（通用版）

一个「把目标当组件、在真实背景上拼贴并自动生成 YOLO 标注」的数据增强流水线。
**换检测任务时，原则上只改根目录的 `config.yaml`，无需改任何脚本源码。**

### 1. 核心思想

- 把「目标图」（整张图就是目标，或"底板 + 内部识别区"）在随机裁剪、增强过的背景图上拼贴；
- 拼贴的同时按目标位置自动写出 YOLO 标注；
- 支持大量几何/颜色/噪声增强，快速造出规模化训练集。

### 2. 唯一配置源：`config.yaml`

所有类别、路径、数量、增强开关、训练超参都集中在 `config.yaml`。脚本通过
`from config import load_config` 读取（`config.py` 负责定位、解析、把相对路径解析为绝对路径）。

关键字段：

- `classes`：类别列表，**下标即 class_id**；目标图须按 `<目标目录>/<类名>/*.png` 组织。
- `paths.*`：背景、合成输出、数据集、训练结果等目录。
- `synth.target_types`：每类目标的来源目录 `dir`、`roi`、缩放范围 `scale`、旋转 `rotate`、
  羽化 `feather`、透明裁剪 `crop_transparent`。
  - `roi: null` → **整张目标图就是检测框**（最常用）。
  - `roi: [rx, ry, rw, rh]` → 目标是"底板"，**只框内部 ROI**（相对目标框的比例）。
- `synth.aug`：**增强菜单**（分 base/target/final 三层，每层 `{num, extra_prob, menu}`；
  每张图从菜单抽 num 种增强，每种按自身 prob 触发；全部可配置，换任务只改这里）。
- `synth.profile`：可选，指向 `tools/calibrate_augment_profile.py` 生成的校准 YAML，
  加载时与 `synth.aug` 深合并（数据驱动标定增强参数）。
- `synth.num_workers`：`1` 用单进程 `realscene.py`；`>1` 用多进程 `multi_rs.py`。
- `train.*`：训练超参；`yolo/dataset.yaml` 由 `y8train.py` 依据本文件自动生成。

### 3. 代码结构

- `realscene/realscene.py`（单进程）/ `realscene/multi_rs.py`（多进程）：拼贴 + 自动标注核心。
- `generate_letters.py`：**「整图即目标」范例生成器**（为每个类别画类名文字图，demo 用）。
- `target/smooth_target.py`：**「底板 + ROI」范例生成器**（把识别区贴到底板中心，demo 用）。
- `tools/`：采集（`manual_shot.py`）、裁剪（`auto_crop.py`）、重命名（`rename.py`）、
  数据集整理（`organize.py`）、标签转换（`json2yolo.py`）、增强标定（`calibrate_augment_profile.py`）、
  计数（`count.py`）。
- `check/`：可视化检查（`label/` 看标注框，`yolo/quick_test.py` 批量推理验证）。
- `yolo/`：训练（`y8train.py`）、混合微调（`finetune.py`）。
- `run.py`：对着场景自动连拍背景图。
- `target/target_new.py`、`target/fast_target_new.py`、`target/make.py`：**遗留**的 CIFAR 期实现，未迁移到配置，仅作参考。

### 4. 工作流

```mermaid
graph TD
    A["拍摄背景/目标(run.py, manual_shot.py)"] --> B["裁剪整理(auto_crop.py, rename.py)"]
    B --> C["准备目标图: 整图 generate_letters.py / 底板 smooth_target.py / 或自备裁剪图"]
    C --> D["拼贴+自动标注(realscene.py 或 multi_rs.py) → train_output/"]
    D --> E["可视化检查标注(check/label/partial_check.py)"]
    E --> F["整理数据集(organize.py) → HDATASET/"]
    F --> G["训练(y8train.py，自动生成 dataset.yaml) → yolo_run/"]
```

### 5. 快速开始（当前 config.yaml 即一个任务实例，改它就是换任务）

```bash
conda env create -f yolo_env.yml && conda activate <你的环境>

python run.py                     # 1. 拍背景图（按 q 退出）→ save_frame/，整理进 paths.backgrounds
python generate_letters.py        # 2. 生成示例目标图（仅演示，实际任务请自备目标图）
python realscene/realscene.py     # 3. 拼贴+标注 → paths.synth_output（先 epochs=1~2 小规模验证，再放量）
python check/label/partial_check.py  # 4. 抽查标注框是否正确
python tools/organize.py          # 5. 切分数据集 → paths.dataset
python yolo/y8train.py            # 6. 训练 → paths.yolo_runs
```

### 6. 换成你自己的任务

1. 改 `config.yaml` 的 `classes`。
2. 准备目标图放进 `target_types.<类型>.dir/<类名>/`（整图即目标就设 `roi: null`）。
3. 把背景照片放进 `paths.backgrounds`。
4. 跑第 3~6 步即可。

### 7. 增强功能全集

所有增强（像素级菜单 / 几何增强 / 隐性机制 / 数据驱动校准）的详细清单、参数、用法见

**[docs/augmentation.md](docs/augmentation.md)**

要点速览：`config.yaml` 的 `synth.aug`（base/target/final 三层菜单）控制像素级增强；
`synth.target_types` 控制目标几何增强（缩放/旋转/透视/羽化）；`label_dir` 是"打框 alpha"
机制（透视/旋转下标注框贴合内容）；背景随机裁剪本身是最强的背景增强。

### 8. 注意点

- **少图实例模式**：识别特定物体（实例即类，如"场地上的那个橙子/球门"）且每类只有 1~几张目标图时，
  用 `synth.target_repeat`（每目标每 epoch 复用 N 次）直接放大数据量：
  单 epoch 产出 ≈ `repeat × 目标数 ÷ 每图贴入数`；再配合放宽 `scale`/`rotate`、降低 `num_targets_per_image`
  让每张图聚焦少数目标。
- **先验证再放量**：数据量开关是 **`synth.epochs`**（每 epoch 扫一遍全目标池；总图数 ≈ epochs × 单 epoch 产出）。
  `synth.num_rounds` 只是单 epoch 的图数**上限**（目标池用完即停），调大它不会多产、保持默认即可。
  验证阶段把 epochs 调到 1~2，跑完用 `partial_check.py` 确认标注框正确，再把 epochs 加到想要的数据量。
- 开启 `synth.apply_geometric_aug` 等几何增强可能让识别框与目标出现细微偏移。
- 背景图的多样性直接决定模型泛化能力——实际会在什么背景上出现，就拍什么背景。

### 8. GPU 批量合成引擎（大规模数据）

数据量需求大（万~十万级）时用单进程批量 GPU 引擎（替代多进程 multi_rs，无 pool 卡死面）：

```bash
python realscene/gpu_synth.py --backend gpu          # 或 config synth.backend: auto
python realscene/gpu_synth.py --backend gpu --epochs 20 --batch 32 --jpeg-workers 8
```

- `synth.backend: gpu|auto|cpu`（auto：有 CUDA 则 gpu 否则 cpu）；`gpu_batch`（每批图数，32-64）；
  `gpu_jpeg_workers`（JPEG 保存线程数）。
- 语义与 `realscene.py` 全一致（目标池/repeat/空镜/紧框/ROI/羽化/透视/增强菜单）；像素批量张量化，
  随机采样/放置/标注保留 CPU（同 seed 同进程复现，跨进程 final 噪声层存在轻微进程级差异，分布无偏）。
- 数据量旋钮不变：`target_repeat × num_rounds × epochs`（如 500×20=1 万张/epoch×20）。
- 实测（5070Ti）：30–88 img/s；服务器 4090 预计 2-3×。未装 CUDA 时自动回退 CPU 后端。
