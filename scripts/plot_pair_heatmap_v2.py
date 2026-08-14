"""
plot_pair_heatmap_v2.py — 351对碱基配对热力图（从checkpoint数据）

不需要ViennaRNA，直接从checkpoint.json的pairs数据生成热力图
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'figure.dpi': 150,
})


def main():
    ROOT = Path(__file__).resolve().parent.parent
    output_dir = ROOT / "output_2013nt"

    # ── 加载 checkpoint ──
    ckpt_path = output_dir / "_checkpoint.json"
    with open(ckpt_path, 'r', encoding='utf-8') as f:
        ckpt = json.load(f)

    all_pairs = ckpt.get('pairs', [])  # [[i, j, bpp], ...]
    far_pairs = ckpt.get('far_pairs', [])
    L = ckpt.get('length', 2013)

    print(f"Sequence length: {L}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Far pairs: {len(far_pairs)}")

    # ── 构建 bpp 矩阵 ──
    bpp_matrix = np.zeros((L, L), dtype=np.float32)
    for (i, j, bpp_val) in all_pairs:
        bpp_matrix[i, j] = bpp_val
        bpp_matrix[j, i] = bpp_val

    # 分类
    near_pairs = [(i, j, b) for (i, j, b) in all_pairs if abs(i - j) < 100]
    far_pairs_data = [(i, j, b) for (i, j, b) in all_pairs if abs(i - j) >= 100]

    print(f"Near pairs (<100nt): {len(near_pairs)}")
    print(f"Far pairs (>=100nt): {len(far_pairs_data)}")

    # ── 画图 ──
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.35,
                           left=0.06, right=0.96, top=0.93, bottom=0.06)

    # ════════════════════════════════════════
    # Panel A: 全局 bpp 矩阵热力图
    # ════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, :2])
    im = ax_a.imshow(bpp_matrix, cmap='hot', vmin=0, vmax=1,
                     aspect='equal', interpolation='nearest')
    ax_a.set_title(f'A. Base Pair Probability Matrix ({L} nt)',
                   fontweight='bold', pad=12, fontsize=13)
    ax_a.set_xlabel('Nucleotide Position')
    ax_a.set_ylabel('Nucleotide Position')
    cbar = plt.colorbar(im, ax=ax_a, shrink=0.8, label='Base Pair Probability (bpp)')

    # 标记 top 配对
    top_n = min(20, len(all_pairs))
    top_pairs = sorted(all_pairs, key=lambda x: -x[2])[:top_n]
    for (i, j, bpp_val) in top_pairs:
        if i < L and j < L:
            ax_a.plot(j, i, 'c*', markersize=5, alpha=0.7)
            ax_a.plot(i, j, 'c*', markersize=5, alpha=0.7)

    # ════════════════════════════════════════
    # Panel B: 351对配对区域放大
    # ════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 2])
    pair_positions = set()
    for (i, j, _) in all_pairs:
        pair_positions.add(i)
        pair_positions.add(j)
    sorted_pos = sorted(pair_positions)

    if sorted_pos:
        sub_bpp = bpp_matrix[np.ix_(sorted_pos, sorted_pos)]
        im_b = ax_b.imshow(sub_bpp, cmap='hot', vmin=0, vmax=1,
                           aspect='auto', interpolation='nearest')
        ax_b.set_title(f'B. Paired Positions Only\n({len(sorted_pos)} positions)',
                       fontweight='bold', pad=12, fontsize=11)
        ax_b.set_xlabel('Paired Position Index')
        ax_b.set_ylabel('Paired Position Index')
        plt.colorbar(im_b, ax=ax_b, shrink=0.8, label='bpp')

    # ════════════════════════════════════════
    # Panel C: BPP 分布
    # ════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])
    bpp_vals = [b for (_, _, b) in all_pairs]
    ax_c.hist(bpp_vals, bins=50, color='#ff9f43', alpha=0.8, edgecolor='#30363d')
    ax_c.set_xlabel('Base Pair Probability')
    ax_c.set_ylabel('Count')
    ax_c.set_title('C. BPP Distribution\n(all predicted pairs)', fontweight='bold', pad=12)
    ax_c.axvline(x=np.median(bpp_vals), color='#58a6ff', linestyle='--', alpha=0.7,
                 label=f'Median: {np.median(bpp_vals):.3f}')
    ax_c.legend(fontsize=9)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # ════════════════════════════════════════
    # Panel D: 配对距离分布
    # ════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])
    near_dists = [abs(j - i) for (i, j, _) in near_pairs]
    far_dists = [abs(j - i) for (i, j, _) in far_pairs_data]

    ax_d.hist(near_dists, bins=30, color='#34d399', alpha=0.7,
              label=f'Near pairs ({len(near_pairs)})', edgecolor='#30363d')
    if far_dists:
        ax_d.hist(far_dists, bins=30, color='#f87171', alpha=0.7,
                  label=f'Far pairs ({len(far_pairs_data)})', edgecolor='#30363d')
    ax_d.set_xlabel('Sequence Distance (|j-i|)')
    ax_d.set_ylabel('Count')
    ax_d.set_title('D. Pair Distance Distribution\n(near vs far-end)', fontweight='bold', pad=12)
    ax_d.legend(fontsize=9)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # ════════════════════════════════════════
    # Panel E: 统计信息
    # ════════════════════════════════════════
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis('off')

    stats = [
        f"Total pairs: {len(all_pairs)}",
        f"  - Near (<100nt): {len(near_pairs)}",
        f"  - Far (>=100nt): {len(far_pairs_data)}",
        "",
        f"Sequence length: {L} nt",
        f"BPP > 0.5: {sum(1 for b in bpp_vals if b > 0.5)}",
        f"BPP > 0.8: {sum(1 for b in bpp_vals if b > 0.8)}",
        f"BPP > 0.95: {sum(1 for b in bpp_vals if b > 0.95)}",
        "",
        f"Median bpp: {np.median(bpp_vals):.4f}",
        f"Mean bpp: {np.mean(bpp_vals):.4f}",
        "",
        "Cyan stars: top 20 bpp pairs",
    ]
    ax_e.text(0.05, 0.95, '\n'.join(stats), transform=ax_e.transAxes,
              fontsize=10, fontfamily='monospace', verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#30363d'))
    ax_e.set_title('E. Statistics', fontweight='bold', pad=12)

    # ── 总标题 ──
    fig.suptitle(f'CircRNA 2013-nt Pairing Heatmap\n'
                 f'{L} nucleotides | {len(all_pairs)} pairs | '
                 f'{len(far_pairs_data)} far-end pairs',
                 fontsize=15, fontweight='bold', color='#58a6ff', y=0.97)

    out_path = output_dir / "pair_heatmap_final.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
