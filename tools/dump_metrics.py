"""
dump_metrics.py — выгрузить метрики всех типов нейронов в CSV.

Метрики:
  - n_neurons    : число нейронов типа
  - degree_in    : число входящих рёбер (тип → другие)
  - degree_out   : число исходящих рёбер
  - weight_in    : суммарный вес входящих
  - weight_out   : суммарный вес исходящих
  - weight_total : weight_in + weight_out
  - count_in     : суммарное число синапсов входящих
  - count_out    : суммарное число синапсов исходящих

Также печатает отсортированные списки в консоль.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/hemibrain.pkl"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "data/type_metrics.csv"

    print(f"Загружаю {input_path} ...", flush=True)
    with open(input_path, "rb") as f:
        d = pickle.load(f)
    neurons = d["neurons"]
    synapses = d["synapses"]

    # Маппинг
    print("Маппинг ...", flush=True)
    types = neurons.apply(extract_type, axis=1)
    typed = pd.DataFrame({
        "bodyId": neurons["bodyId"].astype(np.int64),
        "type": types,
    }).dropna(subset=["type"])
    type_map = dict(zip(typed["bodyId"], typed["type"]))
    vc = typed["type"].value_counts()
    print(f"  типов: {len(vc)}", flush=True)

    # Агрегация (без self-loops — они не влияют на degree)
    print("Агрегация ...", flush=True)
    pre = synapses["pre"].map(type_map)
    post = synapses["post"].map(type_map)
    mask = pre.notna() & post.notna()

    df = pd.DataFrame({
        "pre_type": pre[mask].astype(str),
        "post_type": post[mask].astype(str),
        "weight": synapses.loc[mask, "weight"].values,
    })
    df = df[df["pre_type"] != df["post_type"]]

    grouped = df.groupby(["pre_type", "post_type"], sort=False).agg(
        weight=("weight", "sum"),
        count=("weight", "size"),
    ).reset_index()

    # Метрики
    print("Метрики ...", flush=True)
    all_types = sorted(set(grouped["pre_type"]) | set(grouped["post_type"]))

    by_pre = grouped.groupby("pre_type").agg(
        degree_out=("count", "size"),
        weight_out=("weight", "sum"),
        count_out=("count", "sum"),
    )
    by_post = grouped.groupby("post_type").agg(
        degree_in=("count", "size"),
        weight_in=("weight", "sum"),
        count_in=("count", "sum"),
    )

    rows = []
    for t in all_types:
        n = int(vc.get(t, 0))
        do = int(by_pre.loc[t, "degree_out"]) if t in by_pre.index else 0
        wo = float(by_pre.loc[t, "weight_out"]) if t in by_pre.index else 0.0
        co = int(by_pre.loc[t, "count_out"]) if t in by_pre.index else 0
        di = int(by_post.loc[t, "degree_in"]) if t in by_post.index else 0
        wi = float(by_post.loc[t, "weight_in"]) if t in by_post.index else 0.0
        ci = int(by_post.loc[t, "count_in"]) if t in by_post.index else 0
        rows.append({
            "type": t,
            "n_neurons": n,
            "degree_in": di,
            "degree_out": do,
            "degree_total": di + do,
            "weight_in": wi,
            "weight_out": wo,
            "weight_total": wi + wo,
            "count_in": ci,
            "count_out": co,
            "count_total": ci + co,
        })

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values("n_neurons", ascending=False).reset_index(drop=True)

    # Сохраняем CSV
    metrics.to_csv(output_csv, index=False)
    print(f"\nСохранено: {output_csv}", flush=True)
    print(f"  строк: {len(metrics)}", flush=True)

    # === Вывод по срезам ===
    def print_slice(name, sort_key, top=50, thresholds=None):
        print(f"\n{'=' * 80}", flush=True)
        print(f"ТОП по {name} (всего строк: {len(metrics)})", flush=True)
        print(f"{'=' * 80}", flush=True)
        sorted_df = metrics.sort_values(sort_key, ascending=False)
        print(f"\n  Топ-{top}:", flush=True)
        print(f"  {'type':>30} | {sort_key:>12} | n_neurons | deg_in | deg_out | weight_tot",
              flush=True)
        print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}", flush=True)
        for _, r in sorted_df.head(top).iterrows():
            print(f"  {r['type']:>30} | {r[sort_key]:>12.0f} | {r['n_neurons']:>9} | "
                  f"{r['degree_in']:>6} | {r['degree_out']:>7} | {r['weight_total']:>10.0f}",
                  flush=True)

        if thresholds:
            print(f"\n  Распределение по порогам ({sort_key}):", flush=True)
            for th in thresholds:
                n = (metrics[sort_key] >= th).sum()
                print(f"    ≥ {th:>10}: {n:>5} типов", flush=True)

    # Срезы с порогами
    print_slice("n_neurons", "n_neurons", top=30,
                thresholds=[1, 5, 10, 20, 30, 50, 100, 200, 300, 500])
    print_slice("degree_in", "degree_in", top=30,
                thresholds=[100, 200, 300, 400, 500, 800, 1000, 1500, 2000, 3000])
    print_slice("degree_out", "degree_out", top=30,
                thresholds=[100, 200, 300, 400, 500, 800, 1000, 1500, 2000, 3000])
    print_slice("degree_total", "degree_total", top=30,
                thresholds=[200, 300, 400, 500, 800, 1000, 1500, 2000, 3000, 5000])
    print_slice("weight_total", "weight_total", top=30,
                thresholds=[1000, 5000, 10000, 20000, 50000, 100000, 200000])
    print_slice("count_total", "count_total", top=30,
                thresholds=[500, 1000, 2000, 5000, 10000, 20000, 50000])



if __name__ == "__main__":
    main()
