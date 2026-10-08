"""YOLO 标签类别筛选与重编号：删掉不要的类别，剩余类编号自动接续（0..n-1）。

用途（delivery-5/6 的"贴入但不打框"实验就是这样做的）：
合成完成（或 organize 切分）之后、训练之前，把 labels 里不需要的类别行删掉——
目标仍留在图里当"无标注干扰项/难负样本"——剩余类别按原顺序压缩重编号为 0..n-1；
脚本会打印新的 classes 列表，直接粘回 config.yaml 即可。

用法：
  # 只保留 H（其余类贴了但不参与训练 → H 专属检测器 + 全类难负样本）
  python tools/filter_labels.py delivery-6/HDATASET --keep H --in-place
  # 删掉 H（H 当干扰项，训练其余类 → 治 H 误检）
  python tools/filter_labels.py delivery-5/HDATASET --drop H --out /tmp/filtered
  # 先看统计、不写盘
  python tools/filter_labels.py <路径...> --keep H --dry-run

参数：
  paths            一个或多个目录（递归找 *.txt）或单个 .txt 文件
  --keep / --drop  保留/删除的类（类名或下标，逗号分隔；二者互斥）
  --classes        原类别表（逗号分隔；默认读项目根 config.yaml 的 classes）
  --out DIR        输出到新目录（保持相对结构）；与 --in-place 互斥
  --in-place       就地修改（每个文件先写 .bak 备份；已有 .bak 不覆盖）
  --drop-empty     标签被清空时把对应图片一并删掉（默认保留空标签=背景负样本）
  --dry-run        只统计不写盘
"""
import argparse
import os
import shutil
import sys

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def load_classes_from_config():
    """读项目根 config.yaml 的 classes（找不到就返回 None，由 --classes 兜底）。"""
    root = os.path.dirname(os.path.abspath(__file__))
    while root != os.path.dirname(root) and not os.path.exists(os.path.join(root, "config.yaml")):
        root = os.path.dirname(root)
    cfg = os.path.join(root, "config.yaml")
    if not os.path.exists(cfg):
        return None
    sys.path.insert(0, root)
    try:
        from config import load_config
        return list(load_config(cfg).classes)
    except Exception as e:
        print(f"⚠️ 读取 config.yaml 类别失败（可用 --classes 指定）：{e}")
        return None


def parse_selector(sel, classes):
    """--keep/--drop 的值 → 类 id 集合。支持类名或下标混用。"""
    out = set()
    for tok in sel.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lstrip('-').isdigit():
            out.add(int(tok))
        else:
            if classes is None:
                raise SystemExit(f'❌ 类名 "{tok}" 需要类别表：请用 --classes 指定或补 config.yaml')
            if tok not in classes:
                near = [c for c in classes if tok.lower() in c.lower()]
                hint = f"，相近的：{near}" if near else ""
                raise SystemExit(f'❌ 类名 "{tok}" 不在类别表 {classes} 中{hint}')
            out.add(classes.index(tok))
    return out


def collect_txt(paths):
    """→ [(基准目录, 相对路径)]；目录递归找 *.txt，保持 --out 时的相对结构。"""
    items = []
    for p in paths:
        if os.path.isfile(p):
            if p.lower().endswith('.txt'):
                items.append((os.path.dirname(os.path.abspath(p)), os.path.basename(p)))
            else:
                print(f"⚠️ 跳过非 .txt 文件：{p}")
        elif os.path.isdir(p):
            base = os.path.abspath(p)
            for root, _, files in os.walk(base):
                for f in sorted(files):
                    if f.lower().endswith('.txt'):
                        items.append((base, os.path.relpath(os.path.join(root, f), base)))
        else:
            print(f"⚠️ 跳过不存在的路径：{p}")
    return items


def find_image(txt_path):
    """找标签对应的图片：同目录 或 兄弟 images/ 目录（HDATASET 结构）。"""
    stem = os.path.splitext(txt_path)[0]
    for ext in IMG_EXT:
        if os.path.exists(stem + ext):
            return stem + ext
    d, f = os.path.split(stem)
    sib = os.path.join(os.path.dirname(d), 'images', f)
    for ext in IMG_EXT:
        if os.path.exists(sib + ext):
            return sib + ext
    return None


def main():
    ap = argparse.ArgumentParser(description="YOLO 标签类别筛选与重编号（删除类别 + 序号自动接续）")
    ap.add_argument('paths', nargs='+', help='目录（递归）或 .txt 文件')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--keep', help='保留的类（类名/下标，逗号分隔）')
    g.add_argument('--drop', help='删除的类（类名/下标，逗号分隔）')
    ap.add_argument('--classes', help='原类别表（逗号分隔；默认读 config.yaml）')
    ap.add_argument('--out', help='输出目录（保持相对结构）；与 --in-place 互斥')
    ap.add_argument('--in-place', action='store_true', help='就地修改（写 .bak 备份）')
    ap.add_argument('--drop-empty', action='store_true', help='标签清空时连图片一起删')
    ap.add_argument('--dry-run', action='store_true', help='只统计不写盘')
    args = ap.parse_args()

    if not args.keep and not args.drop:
        raise SystemExit('❌ 需要 --keep 或 --drop 指定要保留/删除的类别')
    if args.in_place and args.out:
        raise SystemExit('❌ --in-place 与 --out 互斥')
    if not args.in_place and not args.out and not args.dry_run:
        raise SystemExit('❌ 需要 --in-place（就地改，写 .bak）或 --out DIR（输出到新目录）；只统计加 --dry-run')

    classes = ([c.strip() for c in args.classes.split(',') if c.strip()]
               if args.classes else load_classes_from_config())

    items = collect_txt(args.paths)
    if not items:
        raise SystemExit('❌ 没找到任何 .txt 标签文件')

    keep = parse_selector(args.keep, classes) if args.keep else None

    # 读入 + 预扫描出现的类（--drop 且无类别表时按出现集合推断保留集）
    parsed, seen = [], set()
    for base, rel in items:
        p = os.path.join(base, rel)
        try:
            lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
        except OSError as e:
            print(f"⚠️ 读取失败 {p}：{e}")
            continue
        parsed.append((base, rel, lines))
        for ln in lines:
            parts = ln.split()
            if parts:
                try:
                    seen.add(int(parts[0]))
                except ValueError:
                    pass

    if keep is None:                      # --drop 模式
        universe = set(range(len(classes))) if classes else seen
        keep = universe - parse_selector(args.drop, classes)

    kept_sorted = sorted(keep)
    if not kept_sorted:
        raise SystemExit('❌ 保留集合为空（全部类别都被删掉了？）')
    remap = {old: i for i, old in enumerate(kept_sorted)}
    unknown = [o for o in kept_sorted if classes and o >= len(classes)]
    if unknown:
        print(f"⚠️ 保留集合含类别表外的下标 {unknown}（config 的 classes 只有 {len(classes)} 类）")
    new_classes = ([classes[o] for o in kept_sorted] if classes and not unknown else None)

    # 逐文件处理
    n_files = n_kept = n_dropped = n_empty = n_img_del = 0
    before, after = {}, {}
    for base, rel, lines in parsed:
        out_lines = []
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except ValueError:            # 非标准行：原样保留（保守）
                out_lines.append(ln)
                continue
            before[cls] = before.get(cls, 0) + 1
            if cls in remap:
                out_lines.append(' '.join([str(remap[cls])] + parts[1:]))
                n_kept += 1
                after[remap[cls]] = after.get(remap[cls], 0) + 1
            else:
                n_dropped += 1
        n_files += 1
        empty = not out_lines
        if empty:
            n_empty += 1
        if args.dry_run:
            continue

        text = '\n'.join(out_lines) + ('\n' if out_lines else '')
        if args.out:
            if empty and args.drop_empty:
                continue                  # 输出模式：空标签不写（图片本就不复制）
            dst = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
            with open(dst, 'w', encoding='utf-8') as f:
                f.write(text)
        else:                             # --in-place
            src = os.path.join(base, rel)
            bak = src + '.bak'
            if not os.path.exists(bak):
                shutil.copy2(src, bak)
            if empty and args.drop_empty:
                os.remove(src)
                img = find_image(src)
                if img:
                    os.remove(img)
                    n_img_del += 1
            else:
                with open(src, 'w', encoding='utf-8') as f:
                    f.write(text)

    # 报告
    print(f"文件 {n_files} 个；保留 {n_kept} 行，删除 {n_dropped} 行；清空 {n_empty} 个"
          + (f"（连图删除 {n_img_del} 张）" if args.drop_empty else ""))
    def _fmt(d, names):
        return ', '.join(f"{k}{'(' + names[k] + ')' if names and k < len(names) else ''}:{v}"
                         for k, v in sorted(d.items())) or '（无）'
    print(f"处理前每类行数：{_fmt(before, classes)}")
    print(f"处理后每类行数：{_fmt(after, new_classes)}   # 编号为新的")
    print("旧类→新类映射：" + ', '.join(
        f"{o}{'(' + classes[o] + ')' if classes and o < len(classes) else ''}→{n}"
        for o, n in sorted(remap.items())))
    if classes:
        removed = sorted(set(range(len(classes))) - keep)
        print("删除的类：" + (', '.join(f"{i}({classes[i]})" for i in removed) or '无'))
    if new_classes:
        print()
        print("新 classes（粘回 config.yaml）：")
        print("classes: [" + ', '.join(new_classes) + "]")
    if args.dry_run:
        print("（--dry-run：未写盘）")


if __name__ == "__main__":
    main()
