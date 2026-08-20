# 适配指南：把本工具用到你自己的检测任务

本工具已通用化——**换任务只改根目录 `config.yaml`，不用动脚本源码**。
下面按"你手里有什么样的目标"分两种情况说明。

---

## 一、先理解两种「目标定义」

`config.yaml` 里 `synth.target_types.<类型>.roi` 决定自动标注怎么画框：

| roi 取值 | 含义 | 适用场景 |
|---|---|---|
| `null` | **整张目标图就是检测框** | 目标裁剪图 / 渲染图，整图即物体 |
| `[rx, ry, rw, rh]` | 目标是"底板"，**只框内部 ROI**（相对目标框的比例） | 物体贴在一块底板/靶标上，只想框物体本身 |

一个 target_type = 一个来源目录 + 一种 roi + 一段缩放范围。可以配多个类型，realscene 会全部加载混合使用。

目标图统一按 `<dir>/<类名>/*.png` 组织，**子文件夹名必须和 `classes` 里的名字完全一致**。

---

## 二、情况 A：整图即目标（最常见）

例：检测「猫 / 狗」，你有一堆猫、狗的抠图。

1. 改 `config.yaml`：
   ```yaml
   classes: [cat, dog]

   synth:
     target_types:
       real_synthetic:
         dir: ./my_targets
         roi: null
         scale: [0.5, 1.0]
   ```
2. 把抠图放好：`my_targets/cat/*.png`、`my_targets/dog/*.png`（建议带透明通道 PNG）。
3. 背景照片放进 `backgrounds/`（`paths.backgrounds`）。
4. 跑流水线（见下方"操作步骤"）。

> 没有现成抠图、只想快速试通？直接 `python generate_letters.py`——它会按 `classes` 给每个类别画一张"类名文字图"作为范例目标。

---

## 三、情况 B：底板 + ROI（靶标场景）

例：物体印在一块圆形/方形底板上，只想框住底板中心的识别区。

1. 用内置范例生成器造底板目标（`target/smooth_target.py`）：把识别区裁剪图放进
   `smooth_target.objects_dir/<类名>/`，运行后会把它们贴到底板中心，输出到
   `smooth_target.output_dir`。关键参数（`config.yaml` 的 `smooth_target` 段）：
   ```yaml
   smooth_target:
     objects_dir: ./recognition_regions
     output_dir: ./synthetic_targets
     board_size: 100      # 底板边长
     object_size: 32      # 识别区边长
     paste_offset: 34     # 粘贴左上角偏移 = (board-object)/2 居中
   ```
2. 在 `target_types` 里加一个指向底板目录、并带 `roi` 的类型。**roi 要和底板几何一致**：
   `roi = [offset/board, offset/board, object/board, object/board]`，默认 100/32/34 即
   `[0.34, 0.34, 0.32, 0.32]`：
   ```yaml
   synth:
     target_types:
       synthetic:
         dir: ./synthetic_targets
         roi: [0.34, 0.34, 0.32, 0.32]
         scale: [0.5, 1.0]
   ```
3. 背景、跑流水线同上。这样标注框只会框住底板内部的识别区，而不是整块底板。

---

## 四、操作步骤

```bash
# 0. 环境
conda env create -f yolo_env.yml
conda activate yolo

# 1. 准备背景图（对着实际场景拍）
python run.py                 # 自动连拍，按 q 退出；save_frame/ 里的图整理进 backgrounds/
#   或 python tools/manual_shot.py  手动按 c 拍

# 2. 准备目标图（三选一）
python generate_letters.py            # 整图范例：按 classes 画类名文字
python target/smooth_target.py        # 底板范例：识别区贴底板
#   或：自己把目标裁剪图放进 target_types.<类型>.dir/<类名>/

# 3. 合成 + 自动标注（★先把 config 的 synth.num_rounds 调到几十做验证）
python realscene/realscene.py         # 单进程；或把 synth.num_workers>1 后用 multi_rs.py
python realscene/multi_rs.py

# 4. 验证标注框（关键！）
python check/label/partial_check.py   # 随机抽样画框，确认框住的是正确区域

# 5. 确认无误后，调大 synth.num_rounds 重跑第 3 步

# 6. 整理数据集
python tools/organize.py              # 按 dataset.val_ratio 切分 → HDATASET/

# 7. 训练（自动依据 config 生成 yolo/craic.yaml）
python yolo/y8train.py

# 8.（可选）看模型预测
python check/yolo/yolo_check.py <图片目录>   # 默认取 yolo_runs/<run_name>/weights/best.pt
```

---

## 五、注意事项

1. **先验证再放量**：务必先小规模跑、用 `partial_check.py` 确认标注正确，再加量。
2. **类名要对齐**：目标子文件夹名 ↔ `classes` 列表，完全一致，否则该类目标会被跳过。
3. **底板 roi 要对齐几何**：情况 B 里 roi 比例必须匹配底板/识别区尺寸，否则框会偏。
4. **背景多样性决定泛化**：实际会出现在什么背景上，就拍什么背景。
5. **多进程分辨率**：`multi_rs.py` 与 `realscene.py` 现在共用 `synth.base_size`；需要更大尺寸就改这一处。
6. **几何增强**：开启 `synth.apply_geometric_aug` 可能让框出现细微偏移，按需权衡。
