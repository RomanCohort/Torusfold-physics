"""Plot REMD energy convergence across 10 rounds."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 10-round REMD data from pipeline log
rounds = np.arange(1, 11)
energy = [309551, 215542, 215191, 214970, 214950, 214525, 214137, 213714, 214867, 214802]
rl_rmsd = [0.73, 0.23, 0.00, 0.05, 0.12, 0.00, 0.00, 0.15, 0.02, 0.37]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})

# Top: energy convergence
ax1.plot(rounds, np.array(energy) / 1000, "o-", color="#2563eb", linewidth=2,
         markersize=6, markerfacecolor="white", markeredgewidth=2, label="REMD energy")
ax1.set_ylabel("Energy (kJ/mol / 1000)", fontsize=11)
ax1.set_title("REMD Energy Convergence (10 Rounds)", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 10.5)

# Annotate round 1 → 2 drop
ax1.annotate(f"-30.4%", xy=(1.5, (309551 + 215542) / 2000),
             fontsize=9, color="#dc2626", fontweight="bold", ha="center")

# Bottom: RL RMSD
ax2.bar(rounds, rl_rmsd, color="#f59e0b", width=0.6, edgecolor="#d97706", linewidth=0.8)
ax2.set_xlabel("REMD Round", fontsize=11)
ax2.set_ylabel("RL Update RMSD (A)", fontsize=11)
ax2.set_xticks(rounds)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = "C:/Users/颜子壹/TorusFold-scheme2-rl/output_2013nt/fig_energy_convergence.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
