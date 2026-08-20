"""通用流水线的配置加载器。

所有脚本通过 `from config import load_config; CFG = load_config()` 读取 `config.yaml`。
- 返回点号可访问的配置（如 `CFG.synth.num_rounds`、`CFG.classes`）。
- 已知的路径字段会被解析为相对 config.yaml 目录（项目根）的绝对路径，
  这样脚本从任意工作目录运行都能找到正确的文件。
"""
import os
import yaml

# config.py 与 config.yaml 同在项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "config.yaml")

_cache = {}


class AttrDict(dict):
    """既能按键访问、也能按属性访问的 dict。缺失属性抛 AttributeError。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def _wrap(obj):
    """把嵌套 dict 递归转成 AttrDict；list 逐项递归；其余原样返回。"""
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _abs(root, p):
    """把相对路径按 root 解析为绝对路径；绝对路径原样返回；空/None 返回原值。"""
    if not p or not isinstance(p, str):
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


def _resolve_paths(cfg, root):
    """就地把已知的路径字段解析为绝对路径。"""
    paths = cfg.get("paths")
    if isinstance(paths, dict):
        for k, v in paths.items():
            paths[k] = _abs(root, v)

    for section, key in (
        ("capture", "save_dir"),
        ("generate_letters", "output_dir"),
    ):
        sec = cfg.get(section)
        if isinstance(sec, dict) and sec.get(key):
            sec[key] = _abs(root, sec[key])

    st = cfg.get("smooth_target")
    if isinstance(st, dict):
        for key in ("objects_dir", "real_targets_dir", "output_dir"):
            val = st.get(key)
            # 保留 "none"/"no" 这类占位关键字，仅解析真实路径
            if val and val.lower() not in ("none", "no"):
                st[key] = _abs(root, val)

    synth = cfg.get("synth")
    if isinstance(synth, dict):
        tt = synth.get("target_types")
        if isinstance(tt, dict):
            for spec in tt.values():
                if isinstance(spec, dict) and spec.get("dir"):
                    spec["dir"] = _abs(root, spec["dir"])
    return cfg


def load_config(path=None):
    """加载并缓存配置。path 缺省时使用项目根的 config.yaml。"""
    cfg_path = os.path.abspath(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path in _cache:
        return _cache[cfg_path]

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = os.path.dirname(cfg_path)
    raw = _resolve_paths(raw, root)
    raw["_root"] = root  # 项目根，供脚本按需拼路径
    cfg = _wrap(raw)
    _cache[cfg_path] = cfg
    return cfg


if __name__ == "__main__":
    c = load_config()
    print("classes:", c.classes)
    print("backgrounds:", c.paths.backgrounds)
    print("synth.num_rounds:", c.synth.num_rounds)
    print("target_types:", dict(c.synth.target_types))
