"""Recreate the manuscript's Scenario 3 plot from the saved exact scores."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase6/conditional_comparison/sweep.csv"
OUT = ROOT / "outputs/phase6/conditional_comparison/figure2_revised"


def main():
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["rho"]) for row in rows] == [
        0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95
    ]
    rho = [float(row["rho"]) for row in rows]
    # Design at the final half-column size so text stays legible in the report.
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    ax.plot(rho, [float(row["interventional_causal_fraction"]) for row in rows],
            "s--", color="tab:orange", linewidth=1.6, markersize=4.5,
            label="Interventional")
    ax.plot(rho, [float(row["conditional_causal_fraction"]) for row in rows],
            "o-", color="tab:blue", linewidth=1.6, markersize=4.5,
            label="Conditional")
    ax.set_xlabel(r"Correlation parameter, $\rho$", fontsize=10)
    ax.set_ylabel("Causal-group score fraction", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.95],
                  ["0", "0.2", "0.4", "0.6", "0.8", "0.95"])
    ax.set_ylim(0.48, 1.045)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"ld_proxy_attribution_fraction.{extension}", dpi=300)
    plt.close(fig)
    print(OUT / "ld_proxy_attribution_fraction.png")


if __name__ == "__main__":
    main()
