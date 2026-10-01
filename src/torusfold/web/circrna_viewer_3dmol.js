// circrna_viewer_3dmol.js — circRNA 3D viewer on 3Dmol.js.
//
// Drop-in replacement for CircRNAViewer (the Mol* implementation). Same method
// names, so the page's representation / opacity / colouring controls drive this
// one without changes: mount, updateStructure, setRepresentation,
// setSurfaceOpacity, applyColoring, setColorPerResidue, setColorUniform,
// setColorCategorical, setScheme.
//
// Why 3Dmol instead of Mol*: Mol* is a 4.8 MB toolkit whose UI is a molecular
// browser in its own right — the centre panel carried expand buttons, viewport
// controls and a settings pane, and five version-specific API pitfalls had to be
// worked around to do anything. This needs to draw a structure and recolor it.
// 3Dmol is 532 KB and does both directly.
//
// One thing 3Dmol does BETTER here: per-residue colouring. The Mol* path could
// only approximate discrete categories by writing a continuous B-factor scale and
// hoping the gradient landed near the right colours (see setColorCategorical in
// the Mol* file, which says as much). setColorByFunction takes the colour for each
// atom directly, so categories are exactly the palette colours.
//
// Data contract matches the backend, shared with the Mol* viewer:
//   pdb: PDB string. Coarse-grained output is P-only with B-factor = confidence;
//        all-atom output has a full backbone and CONECT closing the ring.
//   fp:  {sequence, length, per_residue{}, scalar{}, signals{}, coloring_schemes[]}

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

  // Residue numbers in a PDB are not guaranteed to start at 1 or to be
  // contiguous, so the residue index used to look up per-residue data is built
  // from the order residues actually appear rather than from the number itself.
  function residueIndexMap(atoms) {
    const seen = new Map();
    for (const a of atoms) {
      const key = (a.chain || '') + ':' + a.resi;
      if (!seen.has(key)) seen.set(key, seen.size);
    }
    return seen;
  }

  class CircRNAViewer3Dmol {
    constructor(containerEl) {
      this.container = typeof containerEl === 'string'
        ? document.getElementById(containerEl)
        : containerEl;
      this.viewer = null;
      this.model = null;
      this.structureRef = null;
      this.pdb = '';
      this.fp = null;
      this.schemeSelect = null;
      this.schemeLegend = null;
      this.gradientBar = null;
      this.scalarCards = null;
      this.statsCards = null;
      this._currentRepr = 'cartoon';
      this._surfaceOpacity = 0.35;
      // Explicit per-atom colours, applied after every style rebuild because
      // setStyle replaces the style object that carried them.
      this._colorFn = null;
      this._surfaceId = null;
    }

    setStatus(msg) {
      const el = document.getElementById('status');
      if (el) el.textContent = msg;
    }

    _ensureViewer() {
      if (this.viewer) return this.viewer;
      if (typeof $3Dmol === 'undefined') {
        throw new Error('3Dmol.js did not load');
      }
      const el = this.container || document.getElementById('viewer');
      if (!el) throw new Error('no viewer container');
      this.viewer = $3Dmol.createViewer(el, {
        // Match the panel: a near-black well, and no ambient occlusion pass to
        // keep the first frame fast on large all-atom structures.
        backgroundColor: 0x0c100f,
        antialias: true,
        // Required for screenshot export: a WebGL drawing buffer is cleared
        // after compositing unless it is preserved, so reading the canvas back
        // would yield an empty image.
        preserveDrawingBuffer: true,
      });
      return this.viewer;
    }

    /* Build the style object for a representation kind.
     *
     * 'cartoon' is 3Dmol's own cartoon with `ribbon: true`, which draws a
     * continuous band following the chain — the representation that makes a fold
     * readable. This was measured rather than assumed, because 3Dmol accepts
     * style names it cannot draw: on a real 24-residue RNA, `tube` and `trace`
     * render *nothing at all* (0% of the canvas lit) while `cartoon` and
     * `cartoon+ribbon` both draw (1.92% and 2.49%). 3Dmol's cartoon is written
     * for protein secondary structure, but it branches on the atom name being
     * 'P' for nucleic acids, and this pipeline's structures have a P on every
     * residue, so the ribbon path is reachable.
     *
     * Everything else stays as it was. Atom-level styles are kept because they
     * are the honest view of what the pipeline actually produced — but they are
     * not the default, because on a 42,831-atom structure they are unreadable and
     * slower (stick 114 ms, ribbon 210 ms, and stick lights 8% of the canvas
     * against the ribbon's 15%). */
    _styleFor(kind) {
      const op = this._surfaceOpacity;
      switch (kind) {
        case 'surface':
          return { cartoon: { ribbon: true }, stick: { radius: 0.1, opacity: Math.max(0.35, op) } };
        case 'ball-stick':
          return { stick: { radius: 0.16 }, sphere: { scale: 0.22 } };
        case 'spacefill':
          return { sphere: { scale: 1.0 } };
        case 'atoms':
          // The previous default: every atom drawn.
          return { stick: { radius: 0.12 }, sphere: { scale: 0.14 } };
        case 'cartoon':
        default:
          return { cartoon: { ribbon: true, thickness: 1.4 } };
      }
    }

    _applyStyle(kind) {
      const v = this.viewer;
      if (!v) return;
      // Only 'surface' adds a surface; the ribbon is drawn by the base style in
      // every representation, so this is no longer about the cartoon kind.
      const wantSurface = kind === 'surface';
      v.setStyle({}, this._styleFor(kind));
      if (this._colorFn) {
        this.model.setColorByFunction({}, this._colorFn);
      }
      if (wantSurface) {
        this._addSurface();
      } else {
        this._removeSurface();
      }
      this._currentRepr = kind;
      v.render();
    }

    _addSurface() {
      const v = this.viewer;
      if (!v || !this.model) return;
      this._removeSurface();
      // VDW is the cheap surface. MS/SAS are smoother but take seconds on an
      // all-atom structure, and this rebuilds on every opacity change.
      const id = v.addSurface($3Dmol.SurfaceType.VDW, {
        opacity: this._surfaceOpacity,
        color: '#7fb2ff',
      }, { model: this.model });
      // addSurface hands back a SurfaceId wrapper, not a number.
      this._surfaceId = (id && typeof id === 'object' && 'surfid' in id) ? id.surfid : id;
    }

    _removeSurface() {
      if (this._surfaceId === null || this._surfaceId === undefined) return;
      try {
        this.viewer.removeSurface(this._surfaceId);
      } catch (e) {
        /* already gone */
      }
      this._surfaceId = null;
    }

    async mount(pdb, fp) {
      this.pdb = pdb;
      this.fp = fp || {};
      this.schemeSelect = document.getElementById('scheme-select');
      this.schemeLegend = document.getElementById('scheme-legend');
      this.gradientBar = document.getElementById('gradient-bar');
      this.scalarCards = document.getElementById('scalar-cards');
      this.statsCards = document.getElementById('stats-cards');

      const v = this._ensureViewer();
      // Mol* accumulated structures on reload; 3Dmol has the same failure mode,
      // so clear before adding.
      try { v.removeAllSurfaces(); } catch (e) { /* none */ }
      try { v.removeAllModels(); } catch (e) { /* none */ }
      this._surfaceId = null;
      this._colorFn = null;

      this.model = v.addModel(pdb, 'pdb');
      // Confidence lives in the B-factor column of the coarse-grained output.
      // Colour by it using the observed range rather than a fixed 0..100 scale,
      // so a structure whose confidences sit in a narrow band still shows
      // contrast. This is also the fallback when no scheme has been chosen.
      this._applyConfidenceColors();
      this.structureRef = true;

      this._applyStyle(this._currentRepr);
      v.zoomTo({ model: this.model });
      v.render();

      const ph = document.getElementById('viewer-placeholder');
      if (ph) ph.style.display = 'none';
      this.setStatus('Structure loaded. Colour: ' + (this._colorMeaning || 'default'));

      setTimeout(() => {
        this._renderScalarCards();
        this._renderStatsCards();
        this._fillSchemeSelect();
      }, 0);
      return this;
    }

    /* Replace the displayed structure without touching the rest of the UI.
     *
     * Camera is preserved on purpose: re-framing on every checkpoint makes the
     * molecule appear to jump, and a steady camera is what shows that the same
     * structure is improving. */
    async updateStructure(pdb) {
      if (!this.viewer || !this.model) {
        return this.mount(pdb, this.fp || {});
      }
      const v = this.viewer;
      const view = v.getView();
      const keepColor = this._colorFn;
      const keepRepr = this._currentRepr;

      try { v.removeAllSurfaces(); } catch (e) { /* none */ }
      this._surfaceId = null;
      try { v.removeAllModels(); } catch (e) { /* none */ }

      this.pdb = pdb;
      this.model = v.addModel(pdb, 'pdb');
      // Rebuild the colour function against the new residue list: the same
      // scheme has to map onto whatever residues this file actually contains.
      this._colorFn = (keepColor && keepColor.__perResidue)
        ? this._buildColorFn(keepColor.__values, keepColor.__mode, keepColor.__palette)
        : null;
      if (this._colorFn) {
        this.model.setColorByFunction({}, this._colorFn);
      } else {
        this._applyConfidenceColors();
      }
      this._applyStyle(keepRepr);

      v.setView(view);
      v.render();
      const ph = document.getElementById('viewer-placeholder');
      if (ph) ph.style.display = 'none';
      return this;
    }

    async setRepresentation(kind) {
      if (!this.viewer) return;
      this._applyStyle(kind);
      this.setStatus('representation: ' + kind);
    }

    async setSurfaceOpacity(v) {
      this._surfaceOpacity = v;
      if (this._currentRepr === 'surface') {
        this._applyStyle(this._currentRepr);
      }
    }

    /* Colour by the B-factor column, when it carries data.
     *
     * The coarse-grained checkpoints are P-only with B-factor 0.00 throughout —
     * no confidence has been computed yet — and colouring by a constant makes the
     * structure a single flat blob. So when every value is identical there is
     * nothing to show, and the residue's position along the chain is used instead:
     * blue at the start through red at the end. That is real information (where
     * you are on the ring) rather than a colour with no meaning.
     *
     * Scales to the values actually present rather than a fixed 0..100, so a
     * structure whose confidences sit in a narrow band still shows contrast. */
    _applyConfidenceColors() {
      if (!this.model) return;
      const atoms = this.model.selectedAtoms({});
      const keys = atoms.map((a) => (a.chain || '') + ':' + a.resi);
      const bvals = atoms.map((a) => Number(a.b) || 0);
      let lo = Infinity, hi = -Infinity;
      for (const b of bvals) { if (b < lo) lo = b; if (b > hi) hi = b; }

      const colorByResidue = new Map();
      if (hi > lo) {
        // Real confidence data: spread it across the gradient.
        const range = hi - lo;
        atoms.forEach((a, i) => {
          const key = keys[i];
          if (!colorByResidue.has(key)) {
            colorByResidue.set(key, this._gradientColor((bvals[i] - lo) / range));
          }
        });
        this._colorMeaning = 'confidence';
      } else {
        // No usable B-factor: colour by order along the chain.
        let n = 0;
        const order = new Map();
        for (const key of keys) {
          if (!order.has(key)) order.set(key, n++);
        }
        const total = Math.max(1, order.size - 1);
        for (const [key, idx] of order) {
          colorByResidue.set(key, this._gradientColor(idx / total));
        }
        this._colorMeaning = 'chain position (no confidence in this checkpoint)';
      }

      const fn = (atom) => {
        const c = colorByResidue.get((atom.chain || '') + ':' + atom.resi);
        return c === undefined ? 0x6b7674 : c;
      };
      this._colorFn = fn;
      this.model.setColorByFunction({}, fn);
    }

    /* Build the colour function for one scheme.
     *
     * `values` is one entry per residue, `mode` is 'gradient' or 'categorical'.
     * Atoms are matched to residues by position in the file rather than by
     * residue number, which is neither guaranteed to start at 1 nor to be
     * contiguous. */
    _buildColorFn(values, mode, palette) {
      const atoms = this.model.selectedAtoms({});
      const index = residueIndexMap(atoms);
      const n = values.length;
      // Categorical values are category indices; keep them as integers and look
      // the colour up directly. Normalising them and multiplying back out rounds
      // neighbouring categories onto the same palette entry.
      const fn = (atom) => {
        const key = (atom.chain || '') + ':' + atom.resi;
        const i = index.get(key);
        if (i === undefined || i >= n) return 0x6b7674;  // unmapped residue: neutral
        const val = values[i];
        if (mode === 'categorical') {
          const c = palette[Math.max(0, Math.min(palette.length - 1, Math.round(val)))];
          return (Math.round(c[0] * 255) << 16) | (Math.round(c[1] * 255) << 8) | Math.round(c[2] * 255);
        }
        return this._gradientColor(val);
      };
      // Kept on the function so updateStructure() can rebuild it after a reload.
      fn.__perResidue = true;
      fn.__values = values;
      fn.__mode = mode;
      fn.__palette = palette || null;
      return fn;
    }

    /* Blue -> amber -> red, on the 0..1 normalised value. Matches the gradient
     * bar the page draws beside the legend (style.css .gradient-bar). */
    _gradientColor(t) {
      t = Math.max(0, Math.min(1, t));
      let r, g, b;
      if (t < 0.5) {
        const k = t / 0.5;
        r = 0.17 + (1.00 - 0.17) * k;
        g = 0.40 + (0.87 - 0.40) * k;
        b = 0.84 + (0.34 - 0.84) * k;
      } else {
        const k = (t - 0.5) / 0.5;
        r = 1.00;
        g = 0.87 + (0.07 - 0.87) * k;
        b = 0.34 + (0.26 - 0.34) * k;
      }
      return (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
    }

    async setColorPerResidue(normVals) {
      if (!this.model) return;
      this._colorFn = this._buildColorFn(toArr(normVals), 'gradient', null);
      this.model.setColorByFunction({}, this._colorFn);
      this.viewer.render();
    }

    async setColorUniform(rgb) {
      if (!this.model) return;
      // Captured as a plain number: `const` inside the closure would be in the
      // temporal dead zone when the function runs.
      const packed = (Math.round(rgb[0] * 255) << 16) |
                     (Math.round(rgb[1] * 255) << 8) |
                      Math.round(rgb[2] * 255);
      const fn = () => packed;
      this._colorFn = fn;
      this.model.setColorByFunction({}, fn);
      this.viewer.render();
    }

    /* Discrete colouring. Unlike the Mol* path this is exact: each category gets
     * its palette colour rather than a position on a continuous scale. */
    static CATEGORICAL_PALETTES = {
      base_type: [
        [0.20, 0.60, 0.85],  // A blue
        [0.95, 0.75, 0.30],  // U yellow
        [0.55, 0.80, 0.45],  // G green
        [0.90, 0.45, 0.45],  // C red
      ],
      secondary_structure: [
        [0.70, 0.70, 0.72],  // loop gray
        [0.25, 0.55, 0.92],  // stem blue
      ],
    };

    _paletteFor(key) {
      return CircRNAViewer3Dmol.CATEGORICAL_PALETTES[key] || null;
    }

    async setColorCategorical(schemeKey, categoryArray) {
      if (!this.model) return;
      const palette = this._paletteFor(schemeKey) ||
        CircRNAViewer3Dmol.CATEGORICAL_PALETTES.base_type;
      // Values are category indices, passed through as-is.
      const cats = toArr(categoryArray).map((c) => Number(c) || 0);
      this._colorFn = this._buildColorFn(cats, 'categorical', palette);
      this.model.setColorByFunction({}, this._colorFn);
      this.viewer.render();
      if (this.schemeLegend) {
        this.schemeLegend.textContent = schemeKey === 'base_type'
          ? 'Base type: A(blue) U(yellow) G(green) C(red)'
          : 'Secondary structure: stem(blue) loop(gray)';
      }
    }

    _fillSchemeSelect() {
      if (!this.schemeSelect) return;
      const schemes = (this.fp && this.fp.coloring_schemes) || [];
      this.schemeSelect.innerHTML = '';
      for (const s of schemes) {
        const opt = document.createElement('option');
        opt.value = s.key;
        opt.textContent = s.label;
        opt.dataset.type = s.type;
        this.schemeSelect.appendChild(opt);
      }
      // A single scheme is not a choice. The control is left disabled and the
      // legend says what is on screen instead, rather than offering a dropdown
      // whose one option could never change anything.
      this.schemeSelect.disabled = schemes.length < 2;
      if (schemes.length === 0 && this.schemeLegend) {
        this.schemeLegend.textContent = this._colorMeaning
          ? 'Colouring: ' + this._colorMeaning
          : '';
      }
    }

    async applyColoring(schemeKey) {
      if (!this.structureRef) return;
      const scheme = (this.fp.coloring_schemes || []).find((s) => s.key === schemeKey);
      if (!scheme) return;

      if (scheme.type === 'scalar') {
        if (this.schemeLegend) {
          this.schemeLegend.textContent =
            'Whole-molecule scalar \u2192 uniform colour (values in the cards above)';
        }
        if (this.gradientBar) this.gradientBar.style.background = '#6ab7ff';
        await this.setColorUniform([0.42, 0.72, 1.0]);
        return;
      }

      const vals = (this.fp.per_residue || {})[schemeKey];
      if (!vals) {
        if (this.schemeLegend) this.schemeLegend.textContent = 'No data for this scheme';
        return;
      }

      if (scheme.type === 'categorical') {
        await this.setColorCategorical(schemeKey, vals);
        return;
      }

      // Continuous: normalise across the observed range.
      const arr = toArr(vals);
      let lo = Infinity, hi = -Infinity;
      for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
      const range = (hi - lo) || 1;
      const norm = arr.map((v) => (v - lo) / range);
      if (this.schemeLegend) {
        this.schemeLegend.textContent = scheme.label + ' (normalized: 0 \u2192 1)';
      }
      if (this.gradientBar) {
        this.gradientBar.style.background =
          'linear-gradient(90deg, #2b66d6, #ffdd57, #ff1243)';
      }
      await this.setColorPerResidue(norm);
    }

    setScheme(key) {
      return this.applyColoring(key);
    }

    _renderScalarCards() {
      if (!this.scalarCards) return;
      const scalars = (this.fp && this.fp.scalar) || {};
      const signals = (this.fp && this.fp.signals) || {};
      const merged = Object.assign({}, signals, scalars);
      const skipKeys = new Set([
        'motif_accessibility', 'stem_loop_stem_lengths', 'stem_loop_loop_lengths',
        'ies_structural_dev',
      ]);
      const entries = Object.entries(merged).filter(([k, v]) => {
        if (skipKeys.has(k)) return false;
        if (v === null || v === undefined) return false;
        if (typeof v === 'object') return false;
        if (typeof v === 'string' && v.length > 50) return false;
        return true;
      });
      if (entries.length === 0) {
        this.scalarCards.innerHTML = '<div class="legend">No scalar data</div>';
        return;
      }
      this.scalarCards.innerHTML = entries.map(([k, v]) => {
        const shown = typeof v === 'number' ? (Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(2)) : String(v);
        return '<div class="scalar-card"><div class="k">' +
          k.replace(/_/g, ' ') + '</div><div class="v">' + shown + '</div></div>';
      }).join('');
    }

    _renderStatsCards() {
      if (!this.statsCards) return;
      const per = (this.fp && this.fp.per_residue) || {};
      const keys = Object.keys(per).filter((k) => toArr(per[k]).length > 0);
      if (keys.length === 0) {
        this.statsCards.innerHTML = '<div class="legend">No per-residue data</div>';
        return;
      }
      this.statsCards.innerHTML = keys.map((k) => {
        const arr = toArr(per[k]).map(Number).filter((x) => !isNaN(x));
        if (arr.length === 0) return '';
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        return '<div class="scalar-card"><div class="k">' + k.replace(/_/g, ' ') +
          '</div><div class="v">' + mean.toFixed(3) + '</div></div>';
      }).join('');
    }

    /* Replace the fingerprint the readout cards are built from, and redraw them.

       Needed because a checkpoint load carries no fingerprint: the result payload
       only exists when a run finishes, so on a live or delivered structure the
       "Structure statistics" card had nothing and said "No per-residue data" while
       per-residue measurements were available. */
    setFingerprint(fp) {
      this.fp = Object.assign({}, this.fp || {}, fp || {});
      this._renderStatsCards();
      this._renderScalarCards();
      return this;
    }

    // 3Dmol sizes its canvas from the container, so a panel resize needs a nudge.
    resize() {
      if (this.viewer && this.viewer.resize) {
        this.viewer.resize();
        this.viewer.render();
      }
    }

    /* Spin the scene about the vertical axis by `rad` radians.
     *
     * The camera itself is not exposed the way it was in the Mol* viewer, so this
     * rotates the rendered scene and redraws. Used by the auto-rotate control. */
    rotateBy(rad) {
      if (!this.viewer || !this.viewer.rotate) return;
      this.viewer.rotate(rad, 'y');
      this.viewer.render();
    }

    /* Static PNG of the current view, for the export path. Mol* drawings came
     * from its own screenshot API; 3Dmol renders to a canvas, so read it. */
    toPNGURI() {
      if (!this.viewer || !this.viewer.pngURI) return '';
      return this.viewer.pngURI();
    }
  }

  global.CircRNAViewer = CircRNAViewer3Dmol;
  global.CircRNAViewer3Dmol = CircRNAViewer3Dmol;
})(typeof window !== 'undefined' ? window : this);
