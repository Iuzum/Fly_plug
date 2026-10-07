"""visualize_graph.py — визуализация coarse_graph.json. v2 — больше места, меньше наложений."""

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
    p.add_argument("--top-n", type=int, default=80)
    p.add_argument("--radius", type=int, default=1)
    return p.parse_args()


def categorize(name):
    if name.startswith("KC"): return "KC"
    if name.startswith("MBON"): return "MBON"
    if name.startswith("APL"): return "APL"
    if name.startswith("LC"): return "LC"
    if name.startswith("DN"): return "DN"
    if name.startswith("ER"): return "ER"
    if name.startswith(("EPG", "Delta", "PEN", "PFN", "hDelta", "EL", "FC")): return "CX"
    if name.startswith("ORN"): return "ORN"
    if name.startswith("GF") or name == "Giant Fiber": return "GF"
    return "other"


COLORS = {
    "KC": "#ff7f0e", "MBON": "#d62728", "APL": "#9467bd",
    "LC": "#1f77b4", "DN": "#2ca02c", "ER": "#8c564b",
    "CX": "#17becf", "ORN": "#e377c2", "GF": "#000000",
    "other": "#cccccc",
}


def load_graph(path):
    print(f"Загружаю {path} ...", flush=True)
    with open(path) as f:
        d = json.load(f)
    G = nx.DiGraph()
    types = d["types"]
    for t in types:
        G.add_node(t, category=categorize(t))
    for e in d["edges"]:
        G.add_edge(types[e["from_idx"]], types[e["to_idx"]],
                   weight=e["weight"], count=e["count"])
    print(f"  узлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()}", flush=True)
    return G, d


def draw_subgraph(G, nodes, out_path, title, radius=1,
                  figsize=(32, 32), font_size=10, k=2.0):
    """Рисует подграф вокруг указанных узлов."""
    if not nodes:
        print(f"  SKIP: {title} — нет узлов", flush=True)
        return

    # Расширяем на radius шагов
    selected = set(nodes)
    for _ in range(radius):
        new = set()
        for n in selected:
            new.update(G.successors(n))
            new.update(G.predecessors(n))
        selected.update(new)
    selected = {n for n in selected if n in G}

    subG = G.subgraph(selected).copy()
    if subG.number_of_nodes() == 0:
        print(f"  SKIP: {title} — пустой подграф", flush=True)
        return

    print(f"Рисую {title} ({subG.number_of_nodes()} узлов) -> {out_path} ...", flush=True)

    fig, ax = plt.subplots(figsize=figsize)

    sizes = [200 + 200 * np.log1p(G.nodes[n].get("n_neurons", 1)) for n in subG.nodes()]
    colors = [COLORS.get(subG.nodes[n].get("category", "other"), "#cccccc")
              for n in subG.nodes()]

    # spring_layout с большим k — узлы разлетаются
    pos = nx.spring_layout(subG, k=k, iterations=200, seed=42)

    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.35, width=1.0,
                           edge_color="gray", arrows=True,
                           arrowsize=8, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=sizes,
                           node_color=colors, alpha=0.9,
                           edgecolors="black", linewidths=0.5)

    # Подписи с белым фоном — не сливаются
    labels = {n: n for n in subG.nodes()}
    nx.draw_networkx_labels(subG, pos, labels, ax=ax,
                            font_size=font_size, font_weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.7, pad=0.5))

    ax.set_title(title, fontsize=20)
    ax.axis("off")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[c], label=c)
                       for c in ["KC", "MBON", "APL", "LC", "DN", "ER",
                                 "CX", "ORN", "GF", "other"]]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  DONE", flush=True)


def draw_top(G, counts, out_path, top_n, figsize=(36, 36)):
    print(f"Рисую топ-{top_n} -> {out_path} ...", flush=True)

    top_nodes = sorted(G.nodes(),
                       key=lambda n: counts.get(n, 0),
                       reverse=True)[:top_n]
    subG = G.subgraph(top_nodes).copy()

    fig, ax = plt.subplots(figsize=figsize)

    sizes = [300 + 300 * np.log1p(counts.get(n, 1)) for n in subG.nodes()]
    colors = [COLORS.get(subG.nodes[n].get("category", "other"), "#cccccc")
              for n in subG.nodes()]

    pos = nx.spring_layout(subG, k=3.0, iterations=300, seed=42)

    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.2, width=0.8,
                           edge_color="gray", arrows=False)
    nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=sizes,
                           node_color=colors, alpha=0.9,
                           edgecolors="black", linewidths=0.5)

    labels = {n: n for n in subG.nodes()}
    nx.draw_networkx_labels(subG, pos, labels, ax=ax,
                            font_size=9, font_weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.7, pad=0.4))

    ax.set_title(f"Топ-{top_n} типов по n_neurons", fontsize=22)
    ax.axis("off")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[c], label=c)
                       for c in ["KC", "MBON", "APL", "LC", "DN", "ER",
                                 "CX", "ORN", "GF", "other"]]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  DONE", flush=True)


def draw_paths(G, out_dir):
    """Ключевые пути: escape (LC4 → GF → DN), обучение (KC → MBON), CX."""
    print("\n=== Ключевые пути ===\n", flush=True)

    # 1. Escape: LC4, LC16, Giant Fiber, DN*
    escape_nodes = ["LC4", "LC16", "Giant Fiber"] + \
                   [n for n in G.nodes() if n.startswith("DN")]
    draw_subgraph(G, escape_nodes, out_dir / "graph_escape.png",
                  "Escape path: LC4/LC16 -> Giant Fiber -> DN",
                  radius=1, font_size=9, k=3.0)

    # 2. Mushroom body: KC, MBON, APL
    mb_nodes = [n for n in G.nodes()
                if n.startswith("KC") or n.startswith("MBON")
                or n.startswith("APL")][:40]
    draw_subgraph(G, mb_nodes, out_dir / "graph_mb.png",
                  "Mushroom body: KC -> MBON (learning)",
                  radius=0, font_size=7, k=2.0)

    # 3. Central complex: EPG, Delta7, PEN, ER, PFN
    cx_nodes = [n for n in G.nodes()
                if n.startswith(("EPG", "Delta", "PEN", "ER", "PFN",
                                 "hDelta", "EL", "FC", "FB"))][:50]
    draw_subgraph(G, cx_nodes, out_dir / "graph_cx.png",
                  "Central complex: EPG, Delta7, PEN, ER (heading)",
                  radius=0, font_size=7, k=2.0)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G, d = load_graph(args.input)
    counts = d.get("type_counts", {})

    # Основные
    draw_top(G, counts, out_dir / "graph_top.png", args.top_n)
    draw_paths(G, out_dir)

    print(f"\nВсё сохранено в {out_dir}/", flush=True)
    print("Файлы:", flush=True)
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}", flush=True)


if __name__ == "__main__":
    main()
