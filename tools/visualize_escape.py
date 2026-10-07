"""
visualize_escape.py — детальный граф escape-пути LC4/LC16 -> Giant Fiber -> DN.

Что делает:
  1. Загружает coarse_graph.json.
  2. Строит подграф из LC4, LC16, Giant Fiber, всех DN + промежуточных.
  3. Печатает таблицу связей:
       LC4 -> GF, LC16 -> GF, GF -> DN (для каждого)
       Прямые LC4 -> DN, LC16 -> DN
       Двухшаговые LC4 -> X -> DN
  4. Рисует граф с левым столбцом (LC), центром (GF), правым (DN).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/coarse_graph.json")
    p.add_argument("--output-dir", default="data/graphs")
    return p.parse_args()


def load_graph(path):
    print(f"Загружаю {path} ...", flush=True)
    with open(path) as f:
        d = json.load(f)
    G = nx.DiGraph()
    types = d["types"]
    for t in types:
        G.add_node(t)
    for e in d["edges"]:
        G.add_edge(types[e["from_idx"]], types[e["to_idx"]],
                   weight=e["weight"], count=e["count"])
    print(f"  узлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()}", flush=True)
    return G, d


def find_escape_set(G):
    """Возвращает узлы escape-пути."""
    lc = ["LC4", "LC16"]
    gf = ["Giant Fiber"] if "Giant Fiber" in G else []
    dn = sorted([n for n in G.nodes() if n.startswith("DN")])
    return lc, gf, dn


def diagnose(G, lc, gf, dn):
    """Печатает диагностику связей."""
    print("\n" + "=" * 70, flush=True)
    print("ДИАГНОСТИКА ESCAPE-ПУТИ", flush=True)
    print("=" * 70, flush=True)

    # 1. LC -> GF
    print("\n--- LC4/LC16 -> Giant Fiber ---", flush=True)
    for l in lc:
        if l not in G:
            print(f"  {l}: НЕ в графе", flush=True)
            continue
        for g in gf:
            if G.has_edge(l, g):
                e = G.edges[l, g]
                print(f"  {l} -> {g}: w={e['weight']:.3f}, n={e['count']}", flush=True)
            else:
                print(f"  {l} -> {g}: НЕТ", flush=True)

    # 2. GF -> DN
    print("\n--- Giant Fiber -> DN ---", flush=True)
    if not gf:
        print("  Giant Fiber: НЕ в графе", flush=True)
    else:
        g = gf[0]
        hits = 0
        for d in dn:
            if G.has_edge(g, d):
                e = G.edges[g, d]
                print(f"  GF -> {d}: w={e['weight']:.3f}, n={e['count']}", flush=True)
                hits += 1
        if hits == 0:
            print("  GF НЕ идёт ни в один DN напрямую", flush=True)
        else:
            print(f"  Итого прямых GF -> DN: {hits}", flush=True)

    # 3. LC -> DN напрямую
    print("\n--- LC4/LC16 -> DN напрямую ---", flush=True)
    for l in lc:
        hits = 0
        for d in dn:
            if G.has_edge(l, d):
                e = G.edges[l, d]
                print(f"  {l} -> {d}: w={e['weight']:.3f}, n={e['count']}", flush=True)
                hits += 1
        if hits == 0:
            print(f"  {l}: НЕТ прямых -> DN", flush=True)

    # 4. LC -> X -> DN (двухшаговые)
    print("\n--- LC4/LC16 -> X -> DN (двухшаговые) ---", flush=True)
    for l in lc:
        if l not in G:
            continue
        # Найдём X, через которые LC->X->DN
        lc_succ = set(G.successors(l))
        two_step = {}
        for x in lc_succ:
            for d in dn:
                if G.has_edge(x, d):
                    two_step.setdefault(x, []).append(d)
        if two_step:
            print(f"  {l}: {len(two_step)} промежуточных", flush=True)
            for x, ds in sorted(two_step.items(),
                                key=lambda kv: -len(kv[1]))[:10]:
                w_lx = G.edges[l, x]["weight"]
                print(f"    {l} -> {x} (w={w_lx:.3f}) -> {len(ds)} DN: "
                      f"{', '.join(ds[:3])}{'...' if len(ds) > 3 else ''}",
                      flush=True)
        else:
            print(f"  {l}: НЕТ двухшаговых путей -> DN", flush=True)

    # 5. Что идёт в GF (входящие)
    print("\n--- Кто идёт в Giant Fiber ---", flush=True)
    if gf:
        g = gf[0]
        preds = list(G.predecessors(g))
        print(f"  Входящих в GF: {len(preds)}", flush=True)
        # Топ-10 по весу
        top = sorted(preds, key=lambda p: -G.edges[p, g]["weight"])[:10]
        for p in top:
            e = G.edges[p, g]
            print(f"    {p} -> GF: w={e['weight']:.3f}", flush=True)

    # 6. Что идёт из GF (исходящие)
    print("\n--- Куда идёт Giant Fiber ---", flush=True)
    if gf:
        g = gf[0]
        succs = list(G.successors(g))
        print(f"  Исходящих из GF: {len(succs)}", flush=True)
        top = sorted(succs, key=lambda s: -G.edges[g, s]["weight"])[:15]
        for s in top:
            e = G.edges[g, s]
            marker = " [DN!]" if s.startswith("DN") else ""
            print(f"    GF -> {s}: w={e['weight']:.3f}{marker}", flush=True)


def draw_escape(G, lc, gf, dn, out_path, intermediate=None):
    """Рисует граф escape-пути с колонками LC | GF | DN."""
    print(f"\nРисую escape-граф -> {out_path} ...", flush=True)

    # Собираем узлы: LC + GF + DN + промежуточные
    nodes = set(lc) | set(gf) | set(dn)
    if intermediate:
        nodes |= set(intermediate)
    nodes = {n for n in nodes if n in G}

    if not nodes:
        print("  SKIP: пусто", flush=True)
        return

    subG = G.subgraph(nodes).copy()

    fig, ax = plt.subplots(figsize=(28, 20))

    # === Раскладка по колонкам вручную ===
    pos = {}
    x_lc = 0.0
    x_int = 1.0
    x_gf = 2.0
    x_dn = 3.5

    # LC: левая колонка
    for i, n in enumerate(lc):
        if n in subG:
            pos[n] = (x_lc, -i * 0.8)

    # GF: центр
    for i, n in enumerate(gf):
        if n in subG:
            pos[n] = (x_gf, 0.0)

    # DN: правая колонка
    for i, n in enumerate(dn):
        if n in subG:
            pos[n] = (x_dn, -i * 0.4)

    # Промежуточные: между LC и GF, между GF и DN
    if intermediate:
        for i, n in enumerate(intermediate):
            if n in subG and n not in pos:
                pos[n] = (x_int, -i * 0.3)

    # Авто для тех, кто не попал
    missing = [n for n in subG.nodes() if n not in pos]
    if missing:
        auto = nx.spring_layout(subG.subgraph(missing), seed=42)
        for n, (x, y) in auto.items():
            pos[n] = (x * 0.5 + 1.5, y * 0.5)

    # === Рисуем узлы ===
    def color_for(name):
        if name in lc: return "#1f77b4"        # синий LC
        if name in gf: return "#000000"         # чёрный GF
        if name.startswith("DN"): return "#2ca02c"  # зелёный DN
        return "#cccccc"

    sizes = []
    colors = []
    for n in subG.nodes():
        if n in lc or n in gf or n.startswith("DN"):
            sizes.append(3000)
        else:
            sizes.append(800)
        colors.append(color_for(n))

    nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=sizes,
                           node_color=colors, alpha=0.9,
                           edgecolors="black", linewidths=1.0)

    # === Рисуем рёбра с толщиной по весу ===
    weights = [subG.edges[e]["weight"] for e in subG.edges()]
    wmax = max(weights) if weights else 1.0
    widths = [0.5 + 4.0 * (w / wmax) for w in weights]

    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.5, width=widths,
                           edge_color="gray", arrows=True,
                           arrowsize=12, connectionstyle="arc3,rad=0.05",
                           node_size=3000)

    # === Подписи ===
    labels = {n: n for n in subG.nodes()}
    nx.draw_networkx_labels(subG, pos, labels, ax=ax,
                            font_size=10, font_weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.8, pad=0.4))

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="LC4 / LC16 (looming)"),
        Patch(facecolor="#000000", label="Giant Fiber (escape)"),
        Patch(facecolor="#2ca02c", label="DN (моторные)"),
        Patch(facecolor="#cccccc", label="Промежуточные"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)

    ax.set_title("Escape-путь: LC4/LC16 → Giant Fiber → DN", fontsize=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print("  DONE", flush=True)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G, d = load_graph(args.input)
    lc, gf, dn = find_escape_set(G)
    print(f"\nLC: {lc}", flush=True)
    print(f"GF: {gf}", flush=True)
    print(f"DN ({len(dn)}): {dn}", flush=True)

    # Диагностика
    diagnose(G, lc, gf, dn)

    # Промежуточные — топ-10 по входящим от LC
    intermediate = set()
    for l in lc:
        if l in G:
            for x in G.successors(l):
                if x not in lc and x not in gf and not x.startswith("DN"):
                    intermediate.add(x)
    intermediate = list(intermediate)[:15]

    # Рисуем
    draw_escape(G, lc, gf, dn, out_dir / "graph_escape_detailed.png",
                intermediate=intermediate)

    print(f"\nСохранено: {out_dir}/graph_escape_detailed.png", flush=True)


if __name__ == "__main__":
    main()
