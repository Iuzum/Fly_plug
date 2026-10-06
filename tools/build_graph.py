"""
build_graph.py — строит coarse_graph.json из выбранного списка типов.

Читает:  data/hemibrain.pkl
         data/selected_types.txt  (список типов, по одному на строку)
Пишет:   data/coarse_graph.json

Логика:
  - Маппит bodyId -> type.
  - Оставляет только типы из списка.
  - Агрегирует синапсы.
  - Убирает self-loops.
  - Log-нормализация.
  - Опциональные фильтры min-count, threshold.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/hemibrain.pkl")
    p.add_argument("--types", default="data/selected_types.txt")
    p.add_argument("--output", default="data/coarse_graph.json")
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.0)
    return p.parse_args()


def load_data(path):
    print(f"Загружаю {path} ...", flush=True)
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"  neurons: {d['neurons'].shape}, synapses: {d['synapses'].shape}", flush=True)
    return d["neurons"], d["synapses"]


def load_selected_types(path):
    print(f"Загружаю {path} ...", flush=True)
    with open(path) as f:
        types = set(line.strip() for line in f if line.strip())
    print(f"  типов в списке: {len(types)}", flush=True)
    return types


def clean_str(v):
    if v is None: return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "unknown"): return None
    return s


def extract_type(row):
    t = clean_str(row.get("type"))
    if t: return t
    inst = clean_str(row.get("instance"))
    if inst:
        parts = inst.split("_")
        if parts and parts[0]: return parts[0]
    return None


def build_type_map(neurons):
    types = neurons.apply(extract_type, axis=1)
    typed = pd.DataFrame({"bodyId": neurons["bodyId"].astype(np.int64),
                          "type": types}).dropna(subset=["type"])
    return dict(zip(typed["bodyId"], typed["type"]))


def build_graph(synapses, type_map, selected_types, min_count, threshold):
    print("Агрегация ...", flush=True)
    pre = synapses["pre"].map(type_map)
    post = synapses["post"].map(type_map)
    mask = pre.notna() & post.notna() & pre.isin(selected_types) & post.isin(selected_types)

    print(f"  синапсов: {len(synapses)}, использовано: {int(mask.sum())}", flush=True)

    df = pd.DataFrame({"pre_type": pre[mask].astype(str),
                       "post_type": post[mask].astype(str),
                       "weight": synapses.loc[mask, "weight"].values})

    # Self-loops
    before = len(df)
    df = df[df["pre_type"] != df["post_type"]]
    print(f"  удалено self-loops: {before - len(df)}", flush=True)

    grouped = df.groupby(["pre_type", "post_type"], sort=False).agg(
        weight=("weight", "sum"), count=("weight", "size")).reset_index()
    print(f"  групп: {len(grouped)}", flush=True)

    # min-count
    if min_count > 1:
        before = len(grouped)
        grouped = grouped[grouped["count"] >= min_count].copy()
        print(f"  после min_count: {len(grouped)} (было {before})", flush=True)

    # log-норм
    w = grouped["weight"].values
    w_max = w.max()
    grouped["weight_norm"] = np.log1p(w) / np.log1p(w_max)
    print(f"  max weight: {w_max:.0f}", flush=True)

    # threshold
    if threshold > 0:
        before = len(grouped)
        grouped = grouped[grouped["weight_norm"] >= threshold].copy()
        print(f"  после threshold: {len(grouped)} (было {before})", flush=True)

    return grouped.sort_values("weight_norm", ascending=False).reset_index(drop=True)


def save_json(grouped, output, n_neurons, n_synapses, n_typed, args, n_selected):
    print(f"Сохраняю {output} ...", flush=True)
    types = sorted(set(grouped["pre_type"]) | set(grouped["post_type"]))
    t2i = {t: i for i, t in enumerate(types)}
    edges = [{"from": r["pre_type"], "to": r["post_type"],
              "from_idx": t2i[r["pre_type"]], "to_idx": t2i[r["post_type"]],
              "weight": round(float(r["weight_norm"]), 6), "count": int(r["count"])}
             for _, r in grouped.iterrows()]
    payload = {
        "version": 1, "dataset": "hemibrain:v1.2.1",
        "source": {"types_file": args.types, "n_selected_types": n_selected},
        "filters": {"min_count": args.min_count, "threshold": args.threshold},
        "types": types, "type_index": t2i,
        "edges": edges,
        "stats": {
            "n_neurons_input": int(n_neurons),
            "n_neurons_typed": int(n_typed),
            "n_synapses_input": int(n_synapses),
            "n_types": len(types), "n_edges": len(edges),
        },
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  DONE: {os.path.getsize(output)/1e6:.2f} MB", flush=True)


def print_top(grouped, n=25):
    print(f"\nТоп-{n} рёбер:", flush=True)
    for _, r in grouped.head(n).iterrows():
        print(f"  {r['pre_type']:>25} -> {r['post_type']:<25} "
              f"w={r['weight']:>10.0f}  n={r['count']:>7}  norm={r['weight_norm']:.4f}",
              flush=True)


def main():
    args = parse_args()
    if not Path(args.input).exists():
        print(f"ERROR: {args.input} не найден", file=sys.stderr); sys.exit(1)
    if not Path(args.types).exists():
        print(f"ERROR: {args.types} не найден", file=sys.stderr)
        print("Сначала: python3 tools/select_types.py", file=sys.stderr)
        sys.exit(1)

    print("="*60, flush=True)
    print(f"BUILD GRAPH", flush=True)
    print("="*60, flush=True)
    t0 = time.time()

    neurons, synapses = load_data(args.input)
    selected_types = load_selected_types(args.types)
    type_map = build_type_map(neurons)
    print(f"  typed neurons: {len(type_map)}", flush=True)

    grouped = build_graph(synapses, type_map, selected_types, args.min_count, args.threshold)
    print_top(grouped, 25)
    save_json(grouped, args.output, len(neurons), len(synapses),
              len(type_map), args, len(selected_types))

    n_types_final = len(set(grouped['pre_type']) | set(grouped['post_type']))
    print(f"\nИТОГО: {n_types_final} типов, {len(grouped)} рёбер", flush=True)
    print(f"Время: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
