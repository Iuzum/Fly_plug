"""
coarse_grain.py v5 — smart top-N по 4 срезам.

Порядок:
  1. bodyId -> type (гибрид type/instance).
  2. Агрегация синапсов по (pre_type, post_type).
  3. Удаление self-loops.
  4. Log-нормализация.
  5. Фильтры: min_count, threshold.
  6. Отбор типов (одно из):
       --top-types N   — обычный топ-N по количеству нейронов
       --smart-top N   — топ-N по каждому из 4 срезов (n_neurons, degree_in,
                          degree_out, weight_total), объединение
  7. Сохранение JSON.

Примеры:
  python3 tools/coarse_grain.py --smart-top 150 --min-count 5 --threshold 0.3
  python3 tools/coarse_grain.py --top-types 500 --min-count 5 --threshold 0.3
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/hemibrain.pkl")
    p.add_argument("--output", default="data/coarse_graph.json")
    p.add_argument("--min-count", type=int, default=1,
                   help="Мин. число синапсов в ребре (1 = без фильтра)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Мин. нормализованный вес (0.0 = без фильтра)")
    p.add_argument("--top-types", type=int, default=0,
                   help="Обычный top-N по количеству нейронов (0 = выкл.)")
    p.add_argument("--smart-top", type=int, default=0,
                   help="Smart top-N по 4 срезам (0 = выкл.)")
    return p.parse_args()


# ============================================================
# Загрузка
# ============================================================
def load_data(path):
    print(f"Загружаю {path} ...", flush=True)
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"  neurons: {d['neurons'].shape}, synapses: {d['synapses'].shape}", flush=True)
    return d["neurons"], d["synapses"]


# ============================================================
# Маппинг bodyId -> type
# ============================================================
def clean_str(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "unknown"):
        return None
    return s


def extract_type(row):
    t = clean_str(row.get("type"))
    if t:
        return t
    inst = clean_str(row.get("instance"))
    if inst:
        parts = inst.split("_")
        if parts and parts[0]:
            return parts[0]
    return None


def build_type_map(neurons):
    print("Маппинг bodyId -> type ...", flush=True)
    types = neurons.apply(extract_type, axis=1)
    n_total = len(neurons)
    n_with = int(types.notna().sum())
    print(f"  всего: {n_total}, с типом: {n_with}, без типа: {n_total - n_with}", flush=True)

    typed = pd.DataFrame({
        "bodyId": neurons["bodyId"].astype(np.int64),
        "type": types,
    }).dropna(subset=["type"])

    type_map = dict(zip(typed["bodyId"], typed["type"]))
    vc = typed["type"].value_counts()
    print(f"  уникальных типов: {len(vc)}", flush=True)
    return type_map, vc


# ============================================================
# Агрегация
# ============================================================
def aggregate(synapses, type_map):
    print("Агрегация ...", flush=True)
    pre = synapses["pre"].map(type_map)
    post = synapses["post"].map(type_map)
    mask = pre.notna() & post.notna()

    n_kept = int(mask.sum())
    print(f"  синапсов: {len(synapses)}, использовано: {n_kept}", flush=True)

    df = pd.DataFrame({
        "pre_type": pre[mask].astype(str),
        "post_type": post[mask].astype(str),
        "weight": synapses.loc[mask, "weight"].values,
    })

    grouped = df.groupby(["pre_type", "post_type"], sort=False).agg(
        weight=("weight", "sum"),
        count=("weight", "size"),
    ).reset_index()
    print(f"  групп до self-loops: {len(grouped)}", flush=True)

    # Убираем self-loops
    before = len(grouped)
    grouped = grouped[grouped["pre_type"] != grouped["post_type"]].copy()
    print(f"  удалено self-loops: {before - len(grouped)}", flush=True)

    return grouped


# ============================================================
# Фильтры
# ============================================================
def apply_base_filters(grouped, min_count, threshold):
    print(f"Базовые фильтры: min_count={min_count}, threshold={threshold}", flush=True)

    if min_count > 1:
        before = len(grouped)
        grouped = grouped[grouped["count"] >= min_count].copy()
        print(f"  после min_count: {len(grouped)} (было {before})", flush=True)

    w = grouped["weight"].values
    w_max = w.max()
    grouped["weight_norm"] = np.log1p(w) / np.log1p(w_max)
    print(f"  max weight: {w_max:.0f}", flush=True)

    if threshold > 0:
        before = len(grouped)
        grouped = grouped[grouped["weight_norm"] >= threshold].copy()
        print(f"  после threshold: {len(grouped)} (было {before})", flush=True)

    return grouped.sort_values("weight_norm", ascending=False).reset_index(drop=True)


# ============================================================
# Top-N: обычный и smart
# ============================================================
def top_by_n_neurons(vc, n):
    """Топ-N по количеству нейронов."""
    return set(vc.head(n).index)


def smart_top_n(grouped, vc, per_slice):
    """
    Топ-N по каждому из 4 срезов:
      - n_neurons (из vc)
      - degree_in
      - degree_out
      - weight_total
    Возвращает объединение.
    """
    print(f"Smart top-{per_slice} по 4 срезам ...", flush=True)

    all_types = set(grouped["pre_type"]) | set(grouped["post_type"])
    print(f"  всего типов в графе: {len(all_types)}", flush=True)

    # Метрики
    metrics = {}
    grouped_by_pre = grouped.groupby("pre_type")
    grouped_by_post = grouped.groupby("post_type")

    # degree_out, weight_out
    degree_out = grouped_by_pre.size().to_dict()
    weight_out = grouped_by_pre["weight"].sum().to_dict()

    # degree_in, weight_in
    degree_in = grouped_by_post.size().to_dict()
    weight_in = grouped_by_post["weight"].sum().to_dict()

    for t in all_types:
        metrics[t] = {
            "n_neurons": int(vc.get(t, 0)),
            "degree_in": int(degree_in.get(t, 0)),
            "degree_out": int(degree_out.get(t, 0)),
            "weight_total": float(weight_out.get(t, 0) + weight_in.get(t, 0)),
        }

    # Топ-N по каждому срезу
    def top_by_key(key, n):
        return set(sorted(metrics.keys(), key=lambda t: -metrics[t][key])[:n])

    top_neurons = top_by_key("n_neurons", per_slice)
    top_in = top_by_key("degree_in", per_slice)
    top_out = top_by_key("degree_out", per_slice)
    top_weight = top_by_key("weight_total", per_slice)

    print(f"  top-{per_slice} по n_neurons: {len(top_neurons)}", flush=True)
    print(f"  top-{per_slice} по degree_in: {len(top_in)}", flush=True)
    print(f"  top-{per_slice} по degree_out: {len(top_out)}", flush=True)
    print(f"  top-{per_slice} по weight_total: {len(top_weight)}", flush=True)

    selected = top_neurons | top_in | top_out | top_weight
    print(f"  объединено: {len(selected)} уникальных типов", flush=True)

    # Покажем что попало в топ по разным срезам
    print(f"\n  Топ-10 по n_neurons:", flush=True)
    for t in sorted(metrics, key=lambda x: -metrics[x]["n_neurons"])[:10]:
        print(f"    {t:>30} : n={metrics[t]['n_neurons']}", flush=True)

    print(f"\n  Топ-10 по degree_in:", flush=True)
    for t in sorted(metrics, key=lambda x: -metrics[x]["degree_in"])[:10]:
        print(f"    {t:>30} : in={metrics[t]['degree_in']} out={metrics[t]['degree_out']}", flush=True)

    print(f"\n  Топ-10 по degree_out:", flush=True)
    for t in sorted(metrics, key=lambda x: -metrics[x]["degree_out"])[:10]:
        print(f"    {t:>30} : out={metrics[t]['degree_out']} in={metrics[t]['degree_in']}", flush=True)

    print(f"\n  Топ-10 по weight_total:", flush=True)
    for t in sorted(metrics, key=lambda x: -metrics[x]["weight_total"])[:10]:
        print(f"    {t:>30} : w={metrics[t]['weight_total']:.0f}", flush=True)

    return selected


def apply_type_selection(grouped, vc, top_types, smart_top):
    """Отбор типов: обычный или smart."""
    if smart_top > 0:
        selected = smart_top_n(grouped, vc, smart_top)
    elif top_types > 0:
        selected = top_by_n_neurons(vc, top_types)
        print(f"Обычный top-{top_types} по n_neurons: {len(selected)} типов", flush=True)
    else:
        return grouped  # без отбора

    before = len(grouped)
    grouped = grouped[grouped["pre_type"].isin(selected) &
                      grouped["post_type"].isin(selected)].copy()
    print(f"  после отбора типов: {len(grouped)} рёбер (было {before})", flush=True)
    return grouped


# ============================================================
# Сохранение
# ============================================================
def save_json(grouped, output, n_neurons, n_synapses, n_typed, vc, args):
    print(f"Сохраняю {output} ...", flush=True)

    types = sorted(set(grouped["pre_type"]) | set(grouped["post_type"]))
    t2i = {t: i for i, t in enumerate(types)}

    edges = [{
        "from": r["pre_type"],
        "to": r["post_type"],
        "from_idx": t2i[r["pre_type"]],
        "to_idx": t2i[r["post_type"]],
        "weight": round(float(r["weight_norm"]), 6),
        "count": int(r["count"]),
    } for _, r in grouped.iterrows()]

    payload = {
        "version": 5,
        "dataset": "hemibrain:v1.2.1",
        "filters": {
            "min_count": args.min_count,
            "threshold": args.threshold,
            "top_types": args.top_types,
            "smart_top": args.smart_top,
        },
        "types": types,
        "type_index": t2i,
        "type_counts": {str(k): int(v) for k, v in vc.items()},
        "edges": edges,
        "stats": {
            "n_neurons_input": int(n_neurons),
            "n_neurons_typed": int(n_typed),
            "n_synapses_input": int(n_synapses),
            "n_types": len(types),
            "n_edges": len(edges),
        },
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"  DONE: {os.path.getsize(output)/1e6:.2f} MB", flush=True)


def print_top(grouped, n=20):
    print(f"\nТоп-{n} рёбер:", flush=True)
    for _, r in grouped.head(n).iterrows():
        print(f"  {r['pre_type']:>25} -> {r['post_type']:<25} "
              f"w={r['weight']:>10.0f}  n={r['count']:>7}  norm={r['weight_norm']:.4f}",
              flush=True)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    if not Path(args.input).exists():
        print(f"ERROR: {args.input} не найден", file=sys.stderr)
        sys.exit(1)

    print("=" * 60, flush=True)
    print(f"COARSE-GRAIN v5: min_count={args.min_count}, "
          f"threshold={args.threshold}, "
          f"top_types={args.top_types}, smart_top={args.smart_top}", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()

    # 1. Загрузка
    neurons, synapses = load_data(args.input)

    # 2. Маппинг
    type_map, vc = build_type_map(neurons)

    # 3. Агрегация (с удалением self-loops)
    grouped = aggregate(synapses, type_map)

    # 4. Базовые фильтры
    grouped = apply_base_filters(grouped, args.min_count, args.threshold)

    # 5. Отбор типов
    grouped = apply_type_selection(grouped, vc, args.top_types, args.smart_top)

    # 6. Печать топ-20
    print_top(grouped, 20)

    # 7. Сохранение
    save_json(grouped, args.output, len(neurons), len(synapses),
              len(type_map), vc, args)

    n_types = len(set(grouped['pre_type']) | set(grouped['post_type']))
    print(f"\n{'=' * 60}", flush=True)
    print(f"ИТОГО: {n_types} типов, {len(grouped)} рёбер", flush=True)
    print(f"Время: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
