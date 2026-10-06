"""
select_types.py — отбор типов по порогам из data/type_metrics.csv.

Читает:  data/type_metrics.csv  (все типы с метриками)
Пишет:   data/selected_types.txt  (список выбранных типов, по одному на строку)
         data/selected_types.json (полные метрики выбранных)

Логика: тип остаётся, если проходит ХОТЯ БЫ ОДИН порог.
Дубликаты убираются автоматически (set).

Использование:
  python3 tools/select_types.py
  python3 tools/select_types.py --min-neurons 5 --min-weight 10000 ...
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="data/type_metrics.csv")
    p.add_argument("--output-txt", default="data/selected_types.txt")
    p.add_argument("--output-json", default="data/selected_types.json")
    # пороги
    p.add_argument("--min-neurons", type=int, default=0)
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--min-count-total", type=int, default=0)
    p.add_argument("--min-degree-in", type=int, default=0)
    p.add_argument("--min-degree-out", type=int, default=0)
    p.add_argument("--min-degree-total", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.metrics).exists():
        print(f"ERROR: {args.metrics} не найден", file=sys.stderr)
        print("Сначала: python3 tools/dump_metrics.py", file=sys.stderr)
        sys.exit(1)

    print(f"Загружаю {args.metrics} ...", flush=True)
    df = pd.read_csv(args.metrics)
    print(f"  всего типов: {len(df)}", flush=True)

    # Маски по порогам
    def mask(key, th):
        if th <= 0:
            return pd.Series(False, index=df.index)
        return df[key] >= th

    masks = {
        "n_neurons":    mask("n_neurons",    args.min_neurons),
        "weight_total": mask("weight_total", args.min_weight),
        "count_total":  mask("count_total",  args.min_count_total),
        "degree_in":    mask("degree_in",    args.min_degree_in),
        "degree_out":   mask("degree_out",   args.min_degree_out),
        "degree_total": mask("degree_total", args.min_degree_total),
    }

    print(f"\nПороги:", flush=True)
    print(f"  n_neurons >= {args.min_neurons}", flush=True)
    print(f"  weight_total >= {args.min_weight}", flush=True)
    print(f"  count_total >= {args.min_count_total}", flush=True)
    print(f"  degree_in >= {args.min_degree_in}", flush=True)
    print(f"  degree_out >= {args.min_degree_out}", flush=True)
    print(f"  degree_total >= {args.min_degree_total}", flush=True)

    print(f"\nСколько типов проходит по каждому порогу:", flush=True)
    for name, m in masks.items():
        print(f"  {name}: {int(m.sum())}", flush=True)

    # Объединение
    combined = masks["n_neurons"]
    for m in masks.values():
        combined = combined | m

    if not combined.any():
        print("\n  (ни один порог не сработал — берём все типы)", flush=True)
        combined = pd.Series(True, index=df.index)

    selected = df[combined].copy()

    # Сколько "уникальных" (после дедупликации по type)
    n_unique = selected["type"].nunique()
    print(f"\nВЫБРАНО:", flush=True)
    print(f"  строк: {len(selected)}", flush=True)
    print(f"  уникальных типов: {n_unique}", flush=True)

    # Убираем дубликаты — по одному представителю на тип
    selected_unique = selected.drop_duplicates(subset=["type"], keep="first")
    print(f"  после дедупликации: {len(selected_unique)} типов", flush=True)

    # Список типов — по одному на строку
    type_list = sorted(selected_unique["type"].tolist())
    with open(args.output_txt, "w") as f:
        for t in type_list:
            f.write(f"{t}\n")
    print(f"\n  -> {args.output_txt} ({len(type_list)} типов)", flush=True)

    # Полные метрики — в JSON
    out_json = {
        "filters": {
            "min_neurons": args.min_neurons,
            "min_weight": args.min_weight,
            "min_count_total": args.min_count_total,
            "min_degree_in": args.min_degree_in,
            "min_degree_out": args.min_degree_out,
            "min_degree_total": args.min_degree_total,
        },
        "n_selected": len(type_list),
        "types": type_list,
        "metrics": selected_unique.to_dict(orient="records"),
    }
    with open(args.output_json, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"  -> {args.output_json}", flush=True)

    # Показать топ-30 по n_neurons среди выбранных
    print(f"\nТоп-30 выбранных по n_neurons:", flush=True)
    top = selected_unique.sort_values("n_neurons", ascending=False).head(30)
    for _, r in top.iterrows():
        print(f"  {r['type']:>25} | n={r['n_neurons']:>5} "
              f"| w={r['weight_total']:>10.0f} | c={r['count_total']:>7} "
              f"| di={r['degree_in']:>5} do={r['degree_out']:>5} dt={r['degree_total']:>5}",
              flush=True)


if __name__ == "__main__":
    main()
