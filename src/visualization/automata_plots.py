"""
Otomata modeline ozel gorsellestirmeler.

- Transition probability heatmap
- State diagram (networkx)
- Parametre duyarlilik grafikleri (window_size x alphabet_size)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.automata.automata import ProbabilisticAutomata
from src.config import Config


# ==================================================================
# 1. Transition Probability Heatmap
# ==================================================================
def plot_transition_heatmap(
    model: ProbabilisticAutomata,
    dataset: str,
    config: Config,
    max_states: int = 25,
) -> Path:
    """
    Gecis olasilik matrisini heatmap olarak cizer.
    State sayisi fazlaysa en sik kullanilan max_states tanesi secilir.
    """
    matrix, states = model.get_transition_matrix()

    # Cok fazla state varsa en aktif olanlari sec
    if len(states) > max_states:
        # Her state'in toplam gecis sayisina gore sirala
        activity = matrix.sum(axis=1) + matrix.sum(axis=0)
        top_idx = np.argsort(activity)[-max_states:]
        top_idx = np.sort(top_idx)
        matrix = matrix[np.ix_(top_idx, top_idx)]
        states = [states[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, len(states) * 0.5), max(6, len(states) * 0.4)))

    mask = matrix == 0
    sns.heatmap(
        matrix,
        xticklabels=states,
        yticklabels=states,
        annot=len(states) <= 15,
        fmt=".2f" if len(states) <= 15 else "",
        cmap="YlOrRd",
        mask=mask,
        ax=ax,
        cbar_kws={"label": "Gecis Olasiligi"},
        linewidths=0.5,
    )
    ax.set_xlabel("Hedef State")
    ax.set_ylabel("Kaynak State")
    ax.set_title(f"Transition Probability Heatmap — {dataset}\n"
                 f"(window={model.window_size}, alphabet={model.alphabet_size})")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)

    path = Path(config.results.output_dir) / f"transition_heatmap_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==================================================================
# 2. State Diagram (Networkx)
# ==================================================================
def plot_state_diagram(
    model: ProbabilisticAutomata,
    dataset: str,
    config: Config,
    max_states: int = 20,
    min_prob: float = 0.05,
) -> Path:
    """
    Otomata state diagram'ini cizer.
    Dugumler state'ler, kenarlar gecis olasiliklari.
    Dugum rengi anomali oranina gore kirmizi-yesil.
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx yuklenmemis, state diagram atilanıyor")
        return Path("")

    G = nx.DiGraph()

    # En aktif state'leri sec
    states = sorted(model._all_states)
    if len(states) > max_states:
        activity = {}
        for s in states:
            out_count = sum(model.transition_counts.get(s, {}).values())
            in_count = sum(
                dests.get(s, 0) for dests in model.transition_counts.values()
            )
            activity[s] = out_count + in_count
        states = sorted(activity, key=activity.get, reverse=True)[:max_states]

    # Dugum ekle
    for s in states:
        anomaly_rate = model.state_anomaly_rate.get(s, 0.5)
        G.add_node(s, anomaly_rate=anomaly_rate)

    # Kenar ekle (min_prob altindakiler filtrelenir)
    for src in states:
        if src in model.transition_probs:
            for dst, prob in model.transition_probs[src].items():
                if dst in states and prob >= min_prob:
                    G.add_edge(src, dst, weight=prob)

    if len(G.nodes) == 0:
        return Path("")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Layout
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Dugum renkleri: yesil (normal) -> kirmizi (anomali)
    node_colors = [
        G.nodes[n].get("anomaly_rate", 0.5) for n in G.nodes
    ]

    # Dugum boyutlari: gecis sayisina gore
    node_sizes = []
    for n in G.nodes:
        out_count = sum(model.transition_counts.get(n, {}).values())
        node_sizes.append(max(300, min(2000, out_count * 10)))

    # Kenar kalinliklari
    edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges]

    # Ciz
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.RdYlGn_r,
        vmin=0, vmax=1,
        edgecolors="black",
        linewidths=1,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_weights,
        alpha=0.6,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    # Kenar etiketleri (sadece yuksek olasiklikli)
    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.2f}"
        for u, v in G.edges
        if G[u][v]["weight"] >= 0.15
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax, font_size=6
    )

    # Renk cubugu
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Anomali Orani")

    ax.set_title(
        f"Automata State Diagram — {dataset}\n"
        f"(window={model.window_size}, alphabet={model.alphabet_size}, "
        f"states={len(G.nodes)}, edges={len(G.edges)})"
    )
    ax.axis("off")

    path = Path(config.results.output_dir) / f"state_diagram_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==================================================================
# 3. Parametre Duyarlilik Grafikleri
# ==================================================================
def plot_param_sensitivity_grid(
    param_results: Dict,
    dataset: str,
    config: Config,
) -> List[Path]:
    """
    Parametre analizi sonuclarindan 3 grafik uretir:
    1. F1 heatmap (window x alphabet)
    2. State sayisi heatmap
    3. F1 cizgi grafigi
    """
    paths = []

    # Verileri parse et
    window_sizes = sorted(set(v["window_size"] for v in param_results.values() if "error" not in v))
    alphabet_sizes = sorted(set(v["alphabet_size"] for v in param_results.values() if "error" not in v))

    # --- F1 Heatmap ---
    f1_matrix = np.zeros((len(window_sizes), len(alphabet_sizes)))
    for i, ws in enumerate(window_sizes):
        for j, als in enumerate(alphabet_sizes):
            key = f"ws{ws}_as{als}"
            if key in param_results and "error" not in param_results[key]:
                f1_matrix[i][j] = param_results[key]["f1"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        f1_matrix,
        xticklabels=alphabet_sizes,
        yticklabels=window_sizes,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "F1 Score"},
    )
    ax.set_xlabel("Alphabet Size")
    ax.set_ylabel("Window Size")
    ax.set_title(f"F1 Score: Window Size x Alphabet Size — {dataset}")
    path = Path(config.results.output_dir) / f"param_f1_heatmap_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # --- State Sayisi Heatmap ---
    state_matrix = np.zeros((len(window_sizes), len(alphabet_sizes)))
    for i, ws in enumerate(window_sizes):
        for j, als in enumerate(alphabet_sizes):
            key = f"ws{ws}_as{als}"
            if key in param_results and "error" not in param_results[key]:
                state_matrix[i][j] = param_results[key]["state_count"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        state_matrix,
        xticklabels=alphabet_sizes,
        yticklabels=window_sizes,
        annot=True,
        fmt=".0f",
        cmap="Oranges",
        ax=ax,
        cbar_kws={"label": "State Sayisi"},
    )
    ax.set_xlabel("Alphabet Size")
    ax.set_ylabel("Window Size")
    ax.set_title(f"State Sayisi: Window Size x Alphabet Size — {dataset}")
    path = Path(config.results.output_dir) / f"param_states_heatmap_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # --- F1 Cizgi Grafigi (alphabet_size bazinda) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sol: Window size sabit, alphabet degisir
    for ws in window_sizes:
        f1_vals = []
        for als in alphabet_sizes:
            key = f"ws{ws}_as{als}"
            if key in param_results and "error" not in param_results[key]:
                f1_vals.append(param_results[key]["f1"])
            else:
                f1_vals.append(0)
        axes[0].plot(alphabet_sizes, f1_vals, marker="o", label=f"ws={ws}")
    axes[0].set_xlabel("Alphabet Size")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Alphabet Size Etkisi")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sag: Alphabet sabit, window degisir
    for als in alphabet_sizes:
        f1_vals = []
        for ws in window_sizes:
            key = f"ws{ws}_as{als}"
            if key in param_results and "error" not in param_results[key]:
                f1_vals.append(param_results[key]["f1"])
            else:
                f1_vals.append(0)
        axes[1].plot(window_sizes, f1_vals, marker="s", label=f"as={als}")
    axes[1].set_xlabel("Window Size")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Window Size Etkisi")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Parametre Duyarlilik Analizi — {dataset}", fontsize=13)
    plt.tight_layout()
    path = Path(config.results.output_dir) / f"param_sensitivity_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


# ==================================================================
# 4. Model Karsilastirma (DL vs Automata)
# ==================================================================
def plot_dl_vs_automata(
    all_results: Dict,
    dataset: str,
    config: Config,
) -> Path:
    """DL modelleri ile otomata modelinin F1 karsilastirmasi."""
    models = []
    f1_means = []
    f1_stds = []
    colors = []

    for key, val in all_results.items():
        if dataset.upper() not in key.upper():
            continue
        if "original" not in key.lower():
            continue

        model_name = key.split("_")[1]
        f1 = val.get("f1_mean", val.get("f1_fold_mean", 0))
        std = val.get("f1_std", val.get("f1_fold_std", 0))

        models.append(model_name)
        f1_means.append(f1)
        f1_stds.append(std)
        colors.append("#e74c3c" if model_name == "Automata" else "#3498db")

    if not models:
        return Path("")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=5, color=colors, alpha=0.85)

    # Deger etiketleri
    for bar, mean in zip(bars, f1_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{mean:.3f}", ha="center", fontsize=10, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("F1 Score (mean +/- std)")
    ax.set_title(f"DL vs Automata — {dataset} (Original Senaryo)")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    path = Path(config.results.output_dir) / f"dl_vs_automata_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
