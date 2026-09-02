# 增强功能全集与使用方法

本工具的增强体系全部通过 `config.yaml` 配置（唯一配置源，换任务不改代码）。
增强分四大来源：**像素级菜单**（`synth.aug`）、**几何增强**（`synth.target_types`）、
**隐性机制**（背景裁剪/空镜/复用/JPEG）、**数据驱动校准**（`tools/calibrate_augment_profile.py`）。

```
config.yaml
├── synth.aug            ← 像素级增强菜单（三段：base / target / final）
├── synth.target_types   ← 每类目标的几何增强（缩放/旋转/透视/羽化/裁剪/标注方式）
├── synth.profile        ← 可选：校准 profile（深合并进 synth.aug）
└── paths                ← 输入/输出路径
```

---

## 一、像素级增强菜单（`synth.aug`）

三层按合成时序执行：**base（背景）→ target（目标贴入前）→ final（整图）**。

### 调度规则

- 带 `num: [lo, hi]` → 每张图抽 lo~hi 种，再以 `extra_prob` 概率补抽一种；
- 不带 `num` → 每项按自己的 `prob` 逐项独立触发；
- 每项格式统一：`- {type: <算子>, prob: <触发概率 0~1>, <参数>: <值>}`。

### ① base 层（背景，`config.yaml` aug.base）

背景裁剪缩放到 base_size 之后、贴目标之前。当前默认 `num: [2, 6]`、`extra_prob: 0.3`。

| 算子 | 参数 | 当前默认 | 语义 |
|---|---|---|---|
| `brightness` | `range: [lo, hi]` | 0.5 / `[0.6, 1.35]` | 乘性亮度 `img × U(lo,hi)` |
| `contrast` | `range` | 0.5 / `[0.8, 1.2]` | 围绕通道均值缩放 `(img−mean)×f+mean` |
| `color` | `range` | 0.5 / `[0.5, 1.5]` | 饱和度：向灰度混合，f=0 全灰 |
| `hsv` | `delta` | 0.5 / `0.1` | HSV 三通道各 ±delta 扰动 |
| `cutout` | `mask_size`, `num_masks`, `fill` | 0.5 / `[50,100]` `[1,5]` | 随机矩形遮挡。`fill: noise`（默认）或 `color`（随机纯色块，对齐 4.py 色块） |
| `geometric` | `distortion` | **0.0（默认关）** | 背景透视扰动（CUBIC/REPLICATE）。**只影响背景，不影响标注**；开启会改变背景边角特征 |

### ② target 层（目标，`config.yaml` aug.target）

贴入前、只作用于目标 RGB 通道——**alpha 通道（贴图+打框）保持不变**，不破坏紧框。

| 算子 | 参数 | 当前默认 | 语义 |
|---|---|---|---|
| `brightness` | `range` | 1.0 / `[0.7, 1.35]` | 目标必过一遍，融合背景光线 |
| `contrast` | `range` | 1.0 / `[0.8, 1.2]` | 同上 |
| `sharpness` | `range` | 0.25 / `[0.8, 1.2]` | unsharp mask 锐/柔化 |
| `color` | `range` | 0.35 / `[0.5, 1.5]` | 饱和度 |
| `flip` | 无参数 | 0.5 | 水平翻转（alpha + 打框 alpha 通道同步翻转） |

### ③ final 层（整图，`config.yaml` aug.final）

贴完所有目标后对整图施加。

| 算子 | 参数 | 当前默认 | 语义 |
|---|---|---|---|
| `gaussian` | `var` | 0.5 / `[1, 10]` | 高斯噪声，σ=√var |
| `salt_pepper` | `amount` | 0.3 / `[0.001, 0.005]` | 椒盐（黑白各半的像素比例） |
| `poisson` | `intensity` | 0.3 / `0.1` | 泊松噪声（传感器光子统计） |
| `ink_reflection` | 无参数 | **0.0（默认关）** | 椭圆油墨反光斑 |
| `motion_blur` | `length`, `angle` | 0.3 / `[5,15]` `[0,180]` | 动态模糊（相机抖动/目标运动拖影）。length=拖影长度 px，angle=方向°。（对齐 4.py） |

---

## 二、几何增强（`synth.target_types`）

每个目标类型独立配置，作用于单张目标图。**几何变换后标注自动贴合**（见"打框方案"）。

| 参数 | 当前默认 | 说明 |
|---|---|---|
| `dir` | 必填 | 目标图目录：`<dir>/<类名>/*.png`（文件夹类）或 `<dir>/<类名>.png`（单图类），子目录/文件名必须与 `classes` 一致 |
| `label_dir` | 可选 | **打框 alpha 目录**：与 `dir` 同布局、同相对路径/文件名、同画布。配了它 → 标注紧框取自该图 alpha（透视/旋转下贴合内容，**ROI 自动失效**） |
| `roi` | `null` | 无 `label_dir` 时：`null`=整图即目标；`[rx, ry, rw, rh]`=底板只框内部比例 |
| `scale` | `[0.5, 1.0]` | 目标初始缩放范围（再受 min/max_target_ratio 约束） |
| `rotate` | `[-45, 45]` | 旋转角度（紧框=旋转后 alpha 真实内容轴对齐框） |
| `perspective` | `[0.5, 0.20]` | **[prob, distortion]** 透视形变（模拟相机畸变/倾斜视角）：四角独立扰动 ±distortion×最短边。**配 label_dir 下框贴合；仅配 roi 时框可能轻微漂移**（单应变换不保比例，见下文） |
| `feather` | `4` | 边缘羽化半径 px（去抠图硬边；0=关闭） |
| `crop_transparent` | `true` | 贴入前裁掉 alpha=0 的透明边距（紧框更准；5 通道时以贴图 alpha bbox 裁剪、各通道同步） |

### 打框方案选择（关键）

| 场景 | 选用 |
|---|---|
| 目标图本身=检测框（单图目标） | `roi: null` |
| 底板+中心（环保留、中心换图） | **`label_dir`**（推荐；透视/旋转下框贴合，IoU 1.000） |
| 底板+中心（无 label 素材，只有单图） | `roi: [rx, ry, rw, rh]`（透视下轻微漂移，可用 `tools/auto_roi.py` 测量） |

> **为什么透视下 ROI 会漂移**：透视是单应变换，**不保比例**——"源坐标系固定比例内缩"在变形后不再对应内容的实际像。
> `label_dir` 直接定义拳框（alpha 即内容本身），数学上必然贴合（实验：透视±20%+一半旋转，ROI 比例内缩 IoU 中位 0.737/36% 失配；label_dir IoU 1.000/0% 失配）。

---

## 三、隐性增强机制（不在菜单里）

| 机制 | 配置键 | 当前默认 | 说明 |
|---|---|---|---|
| 背景随机裁剪 | `synth.min_crop_ratio` / `max_crop_ratio` | 0.4 / 0.9 | 每张背景裁一块再缩放——**本身是最强的背景增强** |
| 空镜负样本 | `synth.background_ratio` | 0.05 | 5% 图不贴目标、标签为空，压误检 |
| 目标复用放大 | `synth.target_repeat` | 30 | 少图实例模式：每目标每 epoch 复用 N 次（数据量≈repeat×num_rounds×epochs） |
| JPEG 压缩 | `synth.jpeg_quality` | 95 | 压缩痕迹 |
| 目标比例约束 | `synth.min_target_ratio` / `max_target_ratio` | 0.06 / 0.65 | 目标最终占画布比例（与 scale 联合决定实际大小） |
| 重叠/边界控制 | `max_overlap_attempts` / `to_border` | 10 / 0.02 | 放置拒绝次数、标注边距安全距离 |

---

## 四、边缘残缺增强（`edge_clip`）

模拟真实相机画面里目标被镜头边缘裁切（部分可见、部分缺失，贴边残缺）。

```yaml
# config.yaml synth 段
edge_clip: [0.3, 0.4]   # [prob, max_frac]
```

| 键 | 含义 |
|---|---|
| `prob` | 每个目标放置时触发"允许越界采样"的概率（0=关闭，向后兼容；示例 0.3） |
| `max_frac` | 越界量相对**目标自身宽/高**的最大比例，两轴各方向独立 `uniform(0, max_frac*tw/th)` |

**保底规则**：残缺目标可见宽/高必须 ≥ **30%** 目标自身，否则换样重采样（不会全越界/产生负宽标注）。
数学上 `max_frac ≤ 0.7` 时保底**恒成立**（单侧可见 ≥ tw − max_frac·tw ≥ 0.3·tw），只有 `max_frac > 0.7`
才真正触发重采样；取 `0.9` 可验证该路径。

**行为说明**：
- 触发越界采样时目标可能贴任一画布边缘被裁；未触发时目标完全在画布内（现状行为不变）
- 残缺后的检测框**与可见部分一致**（标注 clamp 自动收缩到画布内）；全越界/碎屑框被丢弃不出标注
- 所有目标类型一视同仁（环类、H/cross 均可残缺）
- 三引擎（realscene / multi_rs / gpu_engine）行为一致；GPU 紧框批算晚，极罕见出现"已贴目标但
  紧框整体出画布"的无标注幻影（CPU 路径有额外预检几乎消除），属已知特性
- 放置失败率上升时（像 `max_frac=0.9` 极限）目标可能放不下，合成密度略降，属预期

**验证命令**：
```bash
# 强制触发 + 保底重采样路径
# config.yaml 设 edge_clip: [1.0, 0.9]，然后
python realscene/gpu_synth.py --backend gpu --epochs 1 --batch 8
# 检查：输出存在贴边被裁目标，且所有 .txt 无负宽/越界标注行
```

---

## 五、数据驱动校准（高级）

```bash
# 统计真实标注图的亮度/清晰度/噪声/压缩痕迹分布 → 推荐增强参数的 YAML
python tools/calibrate_augment_profile.py --mode single \
    --images_dir <真实图目录> --labels_dir <标注txt目录> \
    --out_yaml profiles/xxx_aug.yaml

# 真实域 vs 合成域对齐
python tools/calibrate_augment_profile.py --mode compare \
    --images_dir <真实图> --labels_dir <标注> \
    --synthetic_images_dir <合成图输出> --synthetic_labels_dir <合成标注> \
    --align_strength 1.0 --out_yaml profiles/xxx_aug.yaml
```

生成后把 `config.yaml` 的 `synth.profile` 指向它（`config.py` 加载时与 `synth.aug` 深合并，profile 覆盖 config）。
⚠️ 当前配置 `profile: ./profiles/delivery-2_aug.yaml` 指向的文件**不存在**（`config.py` 仅警告、实际用内嵌增段）——应删除该行或重新生成。

---

## 六、使用流程

1. **改配置**：编辑 `config.yaml` 的 `synth.aug` 三段（`prob` 调触发率、`range/var/amount` 调强度）。
2. **小规模验证**：`python realscene/gpu_synth.py --backend gpu --epochs 1`（几秒），
   用 `check/label/partial_check.py` 抽查标注框是否贴合。
3. **放量**：确认无误后调大 `synth.epochs`（总数据量≈epochs × 单 epoch 产出）。
4. **强度/概率定不准时**：采集真实标注图 → 跑第四节的校准工具 → 指向生成的 profile。

### 运行入口

```bash
# GPU 引擎（推荐，单进程，规避多进程 pool 卡死）
python realscene/gpu_synth.py --backend gpu [--epochs 2000] [--batch 32] [--jpeg-workers 8]

# CPU 引擎
python realscene/realscene.py          # 单进程
python realscene/multi_rs.py           # 多进程（<=24 核）
```

### 后端算子覆盖（已完全对齐）

| 算子 | CPU (realscene.py) | GPU (gpu_engine.py) |
|---|---|---|
| brightness / contrast / color / hsv | ✅ | ✅ |
| sharpness | ✅ | ✅ |
| cutout（含 `fill`） | ✅ | ✅ |
| motion_blur | ✅ | ✅ |
| geometric（背景透视） | ✅ | ✅ |
| flip（target） | ✅ | ✅ |
| gaussian / salt_pepper / poisson（final） | ✅ | ✅ |
| ink_reflection | ✅ | ✅ |

两端共用同一套菜单配置（AUG_MENUS），无需分别开关。

### 常见问题

- **几何增强与打框的关系**：`aug.geometric`（base）作用于**背景层**，不影响标注；`target_types.perspective` 作用于**目标本身**，会影响标注（配 `label_dir` 后无漂移）。
- **加新类别**：`classes` + `target_types` 各加一项；有底板+中心的类记得配 `label_dir` 指向同画布的中心 alpha 图。
- **少图实例模式**：每类 1~几张图时靠 `target_repeat` 放大；配合放宽 `scale`/`rotate` 提高多样性。
