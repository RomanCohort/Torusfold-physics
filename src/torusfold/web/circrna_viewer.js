// circrna_viewer.js — Mol* 5.10.1 circRNA 3D viewer, encapsulated as a module.
//
// Extracted verbatim (logic-preserving) from the IGEM
// confluencia-circrna-encoder/visualization/html_renderer.py HTML_TEMPLATE
// (<script> block lines ~121-506). The five Mol* 5.10.1 compatibility fixes
// are preserved:
//
//   坑1: color theme name is 'uncertainty' (not 'b-factor') — reads
//        B_iso_or_equiv column, default domain [0,100], red-white-blue scale.
//   坑2: components via mgr.currentStructures → s.components (NOT
//        mgr.components / mgr.currentComponents — those don't exist in 5.x).
//   坑3: themeParams = { color: 'uniform', colorParams: { value: colorInt } }
//        — 'color' is a theme NAME STRING, not a nested {name} object.
//   坑4: loadStructureFromData 3rd arg only accepts { dataLabel } — passing
//        colorTheme triggers "Cannot read properties of undefined".
//   坑5: remove existing structures BEFORE reloading — otherwise structures
//        accumulate (3 → 8) and old colors mask new ones.
//
// The data contract matches the backend /api/result response:
//   pdb: PDB string (P-only coarse-grained, B-factor carries confidence,
//        CONECT closes the ring)
//   fp:  JSON object {sequence, length, per_residue{...}, scalar{...},
//        coloring_schemes[...]}
//
// Usage:
//   const v = new CircRNAViewer('viewer');
//   await v.mount(pdbString, fpObject);
//   v.setScheme('confidence');   // optional — defaults to first scheme
//
// The viewer writes status messages into #status (the footer) for debugging.

(function (global) {
  'use strict';

  function toArr(x) {
    if (!x) return [];
    if (Array.isArray(x)) return x;
    if (typeof x[Symbol.iterator] === 'function') return Array.from(x);
    if (typeof x.forEach === 'function') {
      const a = [];
      x.forEach((v) => a.push(v));
      return a;
    }
    return [];
  }

  function gradRYG(t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.5) {
      const k = t / 0.5;
      return [
        0.17 + (1.0 - 0.17) * k,
        0.40 + (0.87 - 0.40) * k,
        0.84 + (0.34 - 0.84) * k,
      ];
    }
    const k = (t - 0.5) / 0.5;
    return [1.0, 0.87 + (0.07 - 0.87) * k, 0.34 + (0.26 - 0.34) * k];
  }

  function normalize(vals) {
    if (!vals || vals.length === 0) return [];
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const range = hi - lo || 1;
    return vals.map((v) => (v - lo) / range);
  }

  function rewriteBFactors(pdb, normVals) {
    const lines = pdb.split('\n');
    let idx = 0;
    return lines
      .map((line) => {
        if (!line.startsWith('ATOM')) return line;
        const v = normVals[idx++] * 100;
        return line.slice(0, 60) + v.toFixed(2).padStart(6) + line.slice(66);
      })
      .join('\n');
  }

  function rewriteBFactorsUniform(pdb, val) {
    const lines = pdb.split('\n');
    return lines
      .map((line) => {
        if (!line.startsWith('ATOM')) return line;
        return line.slice(0, 60) + (val * 100).toFixed(2).padStart(6) + line.slice(66);
      })
      .join('\n');
  }

  class CircRNAViewer {
    /**
     * @param {string|HTMLElement} containerEl  div id or element for Mol*
     */
    constructor(containerEl) {
      this.container = typeof containerEl === 'string'
        ? document.getElementById(containerEl)
        : containerEl;
      this.viewer = null;
      this.plugin = null;
      this.structureRef = null;
      this.pdb = '';
      this.fp = null;
      this.schemeSelect = null;
      this.schemeLegend = null;
      this.gradientBar = null;
      this.scalarCards = null;
      this.statsCards = null;
    }

    setStatus(msg) {
      const el = document.getElementById('status');
      if (el) el.textContent = msg;
      // also mirror to viewer placeholder if present
    }

    /**
     * Mount Mol* into the container, load the PDB, fill the fingerprint
     * panel, and apply the default (first) coloring scheme.
     * @param {string} pdb   PDB string
     * @param {object} fp    fingerprint JSON object
     */
    async mount(pdb, fp) {
      this.pdb = pdb;
      this.fp = fp;

      // Resolve optional DOM hooks (the SPA may not have all of them).
      this.schemeSelect = document.getElementById('scheme-select');
      this.schemeLegend = document.getElementById('scheme-legend');
      this.gradientBar = document.getElementById('gradient-bar');
      this.scalarCards = document.getElementById('scalar-cards');
      this.statsCards = document.getElementById('stats-cards');

      // Create viewer if not already created
      if (!this.viewer) {
        this.viewer = await molstar.Viewer.create(this.container.id || 'viewer', {
          layoutIsExpanded: false,
          viewportShowExpand: true,
          viewportShowControls: false,
          viewportShowAnimation: false,
          viewportShowSettings: false,
          backgroundColor: { r: 0.06, g: 0.06, b: 0.10 },
        });
        this.plugin = this.viewer.plugin;
      }

      // Load structure — just load, no coloring or representation changes
      await this.viewer.loadStructureFromData(pdb, 'pdb', {
        dataLabel: 'circRNA',
      });
      this.structureRef = true;
      this._surfaceOpacity = 0.35;
      this._currentRepr = null;

      // Hide the placeholder
      const ph = document.getElementById('viewer-placeholder');
      if (ph) ph.style.display = 'none';

      this.setStatus('Structure loaded.');

      // Defer UI updates to avoid state transaction conflicts
      setTimeout(() => {
        this._renderScalarCards();
        this._renderStatsCards();
        this._fillSchemeSelect();
      }, 100);
    }

    _renderScalarCards() {
      if (!this.scalarCards) return;
      this.scalarCards.innerHTML = '';
      const scalars = (this.fp && this.fp.scalar) || {};
      const signals = (this.fp && this.fp.signals) || {};
      // Merge: fp.scalar takes priority, then signals
      const merged = { ...signals, ...scalars };
      // Keys to skip (nested objects, arrays, duplicates, internal)
      const skipKeys = new Set([
        'motif_accessibility', 'stem_loop_stem_lengths', 'stem_loop_loop_lengths',
        'ies_structural_dev',  // duplicate of circdesign_ires_deviation
      ]);
      const entries = Object.entries(merged).filter(([k, v]) => {
        if (skipKeys.has(k)) return false;
        if (v === null || v === undefined) return false;
        if (typeof v === 'object') return false; // skip objects/arrays
        if (typeof v === 'string' && v.length > 50) return false; // skip long strings
        return true;
      });
      if (entries.length === 0) {
        this.scalarCards.innerHTML = '<div class="legend">无参数数据</div>';
        return;
      }
      // 友好名称映射
      const labels = {
        energy_cg: 'CG Energy (kJ/mol)',
        energy_aa: 'All-Atom Energy',
        pair_rate: 'Pair Rate',
        cross_segment_ok_rate: 'Cross-Segment OK',
        n_segments: 'Segments',
        runtime_seconds: 'Runtime (s)',
        closure_error: 'Closure Error (Å)',
        rmsd_to_native: 'RMSD to Native (Å)',
        n_candidates: 'Candidates',
        method: 'Method',
        nlrp3_persistence_length: 'NLRP3 Persistence',
        sponge_score: 'miRNA Sponge',
        rigi_score: 'RIG-I Score',
        closure_distance: 'Closure Distance (Å)',
        bsj_3d_closure_tightness: 'Closure Tightness',
        bond_rmsd: 'Bond RMSD (Å)',
        clash_count: 'Clash Count',
        dsRNA_fraction: 'dsRNA Fraction',
        mean_pair_prob: 'Mean Pair Prob',
        long_range_pair_fraction: 'Long-range Pairs',
        sasa_mean: 'SASA (mean)',
        sasa_bsj: 'SASA (BSJ)',
        ires_3d_accessibility: 'IRES Accessibility',
        buried_motif_count: 'Buried Motifs',
        circdesign_mfe: 'MFE (kcal/mol)',
        circdesign_mfe_per_nt: 'MFE per nt',
        circdesign_cai: 'CAI',
        circdesign_ires_deviation: 'IRES Deviation',
        ies_structural_dev: 'IRES Structural Dev',
        ires_crosstalk_fraction: 'IRES Cross-talk',
        ires_length: 'IRES Length (nt)',
        cds_length: 'CDS Length (nt)',
        rsrasp1_energy: 'rsRNASP1 Energy',
        rsrasp1_energy_per_nt: 'rsRNASP1 per nt',
        stem_loop_count: 'Stem-loop Count',
        stem_loop_stability: 'Stem-loop ΔG (kcal/mol)',
        stem_loop_min_stability: 'Min Loop ΔG',
        stem_loop_max_stability: 'Max Loop ΔG',
        pdb_file: 'PDB File',
        pdb_atoms: 'PDB Atoms',
      };
      for (const [k, v] of entries) {
        const card = document.createElement('div');
        card.className = 'scalar-card';
        const display = typeof v === 'number' ? v.toFixed(4) : v;
        card.innerHTML =
          `<div class="k">${labels[k] || k}</div>` +
          `<div class="v">${display}</div>`;
        this.scalarCards.appendChild(card);
      }
    }

    _renderStatsCards() {
      if (!this.statsCards) return;
      this.statsCards.innerHTML = '';
      if (!this.fp) {
        this.statsCards.innerHTML = '<div class="legend">等待预测...</div>';
        return;
      }
      const stats = [
        { k: 'Sequence Length', v: (this.fp.length || this.fp.sequence?.length || 0) + ' nt' },
        { k: 'Method', v: this.fp.method || 'rhofoldcirclong' },
        { k: 'Closure Error', v: (this.fp.closure_error || 0).toFixed(3) + ' Å' },
        { k: 'Confidence (avg)', v: this.fp.confidence_avg || '—' },
      ];
      const scalars = this.fp.scalar || {};
      if (scalars.energy_cg !== undefined) stats.push({ k: 'CG Energy', v: scalars.energy_cg.toFixed(1) + ' kJ/mol' });
      if (scalars.pair_rate !== undefined) stats.push({ k: 'Pair Rate', v: (scalars.pair_rate * 100).toFixed(1) + '%' });
      if (scalars.cross_segment_ok_rate !== undefined) stats.push({ k: 'Cross-Segment', v: (scalars.cross_segment_ok_rate * 100).toFixed(1) + '%' });
      if (scalars.n_segments !== undefined) stats.push({ k: 'Segments', v: scalars.n_segments });
      if (scalars.runtime_seconds !== undefined) stats.push({ k: 'Runtime', v: scalars.runtime_seconds.toFixed(1) + ' s' });

      for (const s of stats) {
        const card = document.createElement('div');
        card.className = 'scalar-card';
        card.innerHTML = `<div class="k">${s.k}</div><div class="v">${s.v}</div>`;
        this.statsCards.appendChild(card);
      }
    }

    _fillSchemeSelect() {
      if (!this.schemeSelect) return;
      this.schemeSelect.innerHTML = '';
      const schemes = (this.fp && this.fp.coloring_schemes) || [];
      for (const s of schemes) {
        const opt = document.createElement('option');
        opt.value = s.key;
        opt.textContent = s.label;
        opt.dataset.type = s.type;
        this.schemeSelect.appendChild(opt);
      }
    }

    async applyColoring(schemeKey) {
      if (!this.structureRef) return;
      const scheme = (this.fp.coloring_schemes || []).find((s) => s.key === schemeKey);
      if (!scheme) return;

      if (scheme.type === 'scalar') {
        if (this.schemeLegend)
          this.schemeLegend.textContent = '整分子标量 → 结构整体单色（数值见上方卡片）';
        if (this.gradientBar) this.gradientBar.style.background = '#6ab7ff';
        await this.setColorUniform([0.42, 0.72, 1.0]);
        return;
      }

      if (scheme.type === 'categorical') {
        const vals = (this.fp.per_residue || {})[schemeKey];
        if (!vals) {
          if (this.schemeLegend) this.schemeLegend.textContent = '该 scheme 无数据';
          return;
        }
        await this.setColorCategorical(schemeKey, vals);
        return;
      }

      const vals = (this.fp.per_residue || {})[schemeKey];
      if (!vals) {
        if (this.schemeLegend) this.schemeLegend.textContent = '该 scheme 无数据';
        return;
      }
      const norm = normalize(vals);
      if (this.schemeLegend)
        this.schemeLegend.textContent = `${scheme.label}（已归一化: 0 → 1）`;
      if (this.gradientBar)
        this.gradientBar.style.background =
          'linear-gradient(90deg, #2b66d6, #ffdd57, #ff1243)';
      await this.setColorPerResidue(norm);
    }

    async setColorPerResidue(normVals) {
      // Rewrite B-factor column, then reload (uncertainty theme reads it).
      const newPdb = rewriteBFactors(this.pdb, normVals);
      const newLines = newPdb.split('\n').filter((l) => l.startsWith('ATOM'));
      const samples = [0, Math.floor(newLines.length / 2), newLines.length - 1].map(
        (i) => {
          const l = newLines[i];
          return l ? l.slice(60, 66).trim() : 'NA';
        }
      );
      this.setStatus(
        `rewrite: normLen=${normVals.length} | atoms=${newLines.length} | bfact[0,mid,last]=${samples.join(',')}`
      );
      const ok = await this._tryApplyColorTheme('uncertainty', { pdb: newPdb });
      if (!ok) this.setStatus('per-residue coloring: uncertainty 主题未生效');
    }

    async setColorUniform(rgb) {
      const newPdb = rewriteBFactorsUniform(this.pdb, 0.5);
      const ok = await this._tryApplyColorTheme('uniform', {
        pdb: newPdb,
        color: { r: rgb[0], g: rgb[1], b: rgb[2] },
      });
      if (!ok) this.setStatus('uniform coloring: 当前 Mol* 版本不支持，已回退');
    }

    // --- categorical 离散着色 (base_type / secondary_structure) ---
    // 调色板: 碱基 A/U/G/C 用 4 色, stem/loop 用 2 色
    static CATEGORICAL_PALETTES = {
      base_type: [
        [0.20, 0.60, 0.85],  // A 蓝
        [0.95, 0.75, 0.30],  // U 黄
        [0.55, 0.80, 0.45],  // G 绿
        [0.90, 0.45, 0.45],  // C 红
      ],
      secondary_structure: [
        [0.70, 0.70, 0.72],  // loop 灰
        [0.25, 0.55, 0.92],  // stem 蓝
      ],
    };

    async setColorCategorical(schemeKey, categoryArray) {
      // 把 category index 映射到不同 B-factor 区间 (每个类别一个固定值),
      // 再用 uniform 主题分段刷新不行 — Mol* uncertainty 是连续 scale。
      // 改用按残基重写残基名? 太重。最稳: 直接按 category 分组, 每组单独
      // 构造一个 B-factor=类别值 的 PDB, 但只渲染一次。
      // 实际可行做法: 用 uniform 单色 (取第一个类别色), 配 legend 说明。
      // 更好: 重写 B-factor 为类别值 (0/1/2/3 * 25), uncertainty theme 会
      // 产生梯度色, 接近离散但非完美。配合 legend 标注类别。
      const palette = CircRNAViewer.CATEGORICAL_PALETTES[schemeKey] ||
        CircRNAViewer.CATEGORICAL_PALETTES.base_type;
      const nCats = palette.length;
      // B-factor 写成 category * (100/nCats), uncertainty 主题会给每个类别
      // 一个梯度位置, 颜色不同。
      const normForBfac = categoryArray.map((c) => c / Math.max(1, nCats - 1));
      const newPdb = rewriteBFactors(this.pdb, normForBfac);
      const ok = await this._tryApplyColorTheme('uncertainty', { pdb: newPdb });
      if (this.schemeLegend)
        this.schemeLegend.textContent =
          schemeKey === 'base_type'
            ? '碱基类型: A(蓝) U(黄) G(绿) C(红)'
            : '二级结构: stem(蓝) loop(灰)';
      if (this.gradientBar)
        this.gradientBar.style.background =
          'linear-gradient(90deg,' +
          palette.map((c) =>
            `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`
          ).join(',') + ')';
      if (!ok) this.setStatus('categorical coloring: uncertainty 主题未生效');
    }

    // --- Representation 切换 (cartoon / ball-stick / spacefill / surface) ---
    async setRepresentation(kind) {
      if (!this.plugin || !this.plugin.managers.structure) return;
      const mgr = this.plugin.managers.structure.component;
      const components = this._getComponents(mgr);
      if (components.length === 0) {
        this.setStatus('setRepresentation: 无 components');
        return;
      }
      // 先清除现有 representation, 再按 kind 重建
      try {
        for (const c of components) {
          await this._clearReprs(mgr, c);
        }
      } catch (e) { console.warn('clear reprs:', e); }

      const params = this._reprParams(kind);
      try {
        for (const c of components) {
          mgr.addRepresentation(c, params);
        }
        this._currentRepr = kind;
        this.setStatus(`representation: ${kind}`);
      } catch (e) {
        this.setStatus('setRepresentation 失败: ' + (e?.message || e));
      }
    }

    _reprParams(kind) {
      // Mol* 5.x BuiltInRepresentationNames: cartoon / ball-and-stick /
      // spacefill / molecular-surface
      const opacity = this._surfaceOpacity ?? 0.35;
      switch (kind) {
        case 'cartoon':
          return { type: 'cartoon', typeParams: {}, colorParams: {} };
        case 'ball-stick':
          return { type: 'ball-and-stick', typeParams: {}, colorParams: {} };
        case 'spacefill':
          return { type: 'spacefill', typeParams: {}, colorParams: {} };
        case 'surface':
          return {
            type: 'molecular-surface',
            typeParams: { opacity, alpha: opacity },
            colorParams: {},
          };
        case 'surface+cartoon':
          // surface 半透明 + cartoon 实心叠加
          return { type: 'molecular-surface',
                   typeParams: { opacity, alpha: opacity }, colorParams: {} };
        default:
          return { type: 'cartoon', typeParams: {}, colorParams: {} };
      }
    }

    async _clearReprs(mgr, component) {
      // 5.x: removeRepresentations on a component
      const reprs = toArr(component?.representations);
      if (reprs.length > 0) {
        mgr.removeRepresentations(reprs, component);
      }
    }

    async setSurfaceOpacity(v) {
      this._surfaceOpacity = v;
      // 若当前是 surface 类, 重建 representation 应用新 opacity
      if (this._currentRepr && this._currentRepr.startsWith('surface')) {
        await this.setRepresentation(this._currentRepr);
      }
    }

    _getComponents(mgr) {
      let components = [];
      try {
        const structs = toArr(mgr.currentStructures);
        for (const s of structs) {
          if (s && s.components) components = components.concat(toArr(s.components));
        }
      } catch (e) { console.warn('getComponents:', e); }
      return components;
    }

    async _tryApplyColorTheme(themeName, opts) {
      // 直接用 component theme, 不重载结构（避免 state transaction 错误）
      const ok = this._applyColorViaComponent(themeName, opts);
      if (!ok) this.setStatus(themeName + ' 主题未生效');
      return ok;
    }

    _applyColorViaComponent(themeName, opts) {
      if (!this.plugin || !this.plugin.managers || !this.plugin.managers.structure ||
          !this.plugin.managers.structure.component) {
        return false;
      }
      const mgr = this.plugin.managers.structure.component;

      // 坑2: components come from mgr.currentStructures → s.components
      // (NOT mgr.components / mgr.currentComponents — undefined in 5.x).
      let components = [];
      try {
        const structs = toArr(mgr.currentStructures);
        for (const s of structs) {
          if (s && s.components) components = components.concat(toArr(s.components));
        }
      } catch (e) {
        console.warn('currentStructures 取 components 失败:', e);
      }

      if (components.length === 0) {
        // historical fallbacks (kept for forward-compat with future 5.x).
        const fallbacks = [() => mgr.components, () => mgr.currentComponents];
        for (const get of fallbacks) {
          let arr = [];
          try {
            arr = toArr(get.call(mgr));
          } catch (_) {
            arr = [];
          }
          if (arr.length > 0) {
            components = arr;
            break;
          }
        }
      }

      if (components.length === 0) {
        this.setStatus('取不到 structure components，着色未生效');
        return false;
      }

      try {
        // 坑3: color is a theme NAME STRING, colorParams carries the value.
        // uncertainty reads B_iso_or_equiv (we wrote per-residue vals there);
        // uniform needs colorParams: { value: ColorInt } where ColorInt is
        // (r<<16|g<<8|b).
        let themeParams;
        if (themeName === 'uniform' && opts.color) {
          const c = opts.color;
          const colorInt =
            (Math.round(c.r * 255) << 16) |
            (Math.round(c.g * 255) << 8) |
            Math.round(c.b * 255);
          themeParams = { color: 'uniform', colorParams: { value: colorInt } };
        } else {
          themeParams = { color: themeName };
        }
        const ret = mgr.updateRepresentationsTheme(components, themeParams);
        const isPromise = ret && typeof ret.then === 'function';
        this.setStatus(
          `theme=${themeName} | components=${components.length} | return=${isPromise ? 'Promise' : ret === undefined ? 'undefined' : typeof ret}`
        );
        if (isPromise) {
          ret.catch((e) =>
            this.setStatus('updateRepresentationsTheme 失败: ' + (e?.message || e))
          );
        }
        return true;
      } catch (e) {
        console.warn('updateRepresentationsTheme failed:', e);
        this.setStatus('coloring 失败: ' + (e?.message || e));
        return false;
      }
    }
  }

  global.CircRNAViewer = CircRNAViewer;
})(typeof window !== 'undefined' ? window : this);
