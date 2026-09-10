/* panels.js — Data visualization panel renderers for TorusFold */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Panels = TF.Panels = {};
  const Charts = TF.Charts;

  const $ = function (id) { return document.getElementById(id); };

  /* ── Helpers ── */
  function fmt(v, d) { return typeof v === 'number' ? v.toFixed(d || 2) : '--'; }
  function fmtPct(v) { return typeof v === 'number' ? (v * 100).toFixed(1) + '%' : '--'; }

  function makeCard(label, value, color) {
    return '<div class="scalar-card"><div class="k">' + label + '</div><div class="v" style="color:' + (color || 'var(--accent)') + '">' + value + '</div></div>';
  }

  /* ═══════════════ OVERVIEW TAB ═══════════════ */

  Panels.renderScoring = function (result) {
    if (!result) return;
    const rn = result.rnadvisor || {};
    const items = [
      // NOTE: the -2000 pass threshold below has no provenance (no calibration against a
      // reference set was recorded). Do not treat a PASS/FAIL here as meaningful until it
      // is calibrated; an absent value renders as N/A and skips the threshold entirely.
      { id: 'score-rsrnasp', valId: 'score-rsrnasp-val', key: 'rsRNASP_docker', passFn: function (v) { return v < -2000; } },
      { id: 'score-dfire', valId: 'score-dfire-val', key: 'DFIRE', passFn: function (v) { return v < 0; } },
      { id: 'score-3drnascore', valId: 'score-3drnascore-val', key: '3drnascore', passFn: function (v) { return v > 0; } },
    ];
    for (const it of items) {
      var el = $(it.id), valEl = $(it.valId);
      if (!el || !valEl) continue;
      var badge = el.querySelector('.score-badge');
      var v = rn[it.key];
      if (v == null) { valEl.textContent = '--'; if (badge) { badge.className = 'score-badge pending-badge'; badge.textContent = 'N/A'; } }
      else {
        valEl.textContent = typeof v === 'number' ? v.toFixed(1) : v;
        var pass = it.passFn(v);
        valEl.style.color = pass ? 'var(--ok)' : 'var(--err)';
        if (badge) { badge.className = 'score-badge ' + (pass ? 'pass-badge' : 'fail-badge'); badge.textContent = pass ? 'PASS' : 'FAIL'; }
      }
    }
  };

  Panels.renderPhysical = function (result) {
    if (!result) return;
    var phys = result.physical || {};
    var panel = $('physical-panel');
    var row = $('gauges-row');
    var extra = $('physical-extra');
    if (!panel || !row) return;
    row.innerHTML = '';
    if (extra) extra.innerHTML = '';

    var gauges = [
      ['closure_distance_Ang', 'Closure', 'A', 30, { good: 5, warn: 12 }, false],
      ['bond_rmsd_Ang', 'Bond RMSD', 'A', 5, { good: 1, warn: 2 }, false],
      ['sasa_mean', 'SASA', '', 1, { good: 0.4, warn: 0.7 }, false],
      ['bsj_closure_tightness', 'Tightness', '', 1, { good: 0.6, warn: 0.3 }, true],
    ];
    for (var i = 0; i < gauges.length; i++) {
      var g = gauges[i], v = phys[g[0]];
      if (v == null) continue;
      var c = document.createElement('div');
      c.className = 'gauge-cell';
      c.appendChild(Charts.createGauge(v, g[3], g[1], g[2], g[4], g[5]));
      row.appendChild(c);
    }

    if (extra) {
      var extras = [
        ['sasa_bsj', 'SASA (BSJ)'],
        ['dsRNA_fraction', 'dsRNA fraction'],
        ['mean_pair_prob', 'Mean pair prob'],
        ['long_range_pair_fraction', 'Long-range pairs'],
      ];
      for (var j = 0; j < extras.length; j++) {
        var kv = extras[j], val = phys[kv[0]];
        if (val == null) continue;
        extra.innerHTML += makeCard(kv[1], typeof val === 'number' ? val.toFixed(3) : val);
      }
    }
  };

  Panels.renderCircDesign = function (result) {
    if (!result) return;
    var cd = result.circdesign || {};
    var panel = $('circdesign-panel');
    var el = $('circdesign-cards');
    if (!panel || !el) return;
    if (cd.mfe_kcal_mol == null && cd.cai_human == null) { panel.style.display = 'none'; return; }
    panel.style.display = '';
    el.innerHTML = '';

    var items = [
      { key: 'mfe_kcal_mol', label: 'MFE', fmt: function (v) { return v.toFixed(1) + ' kcal/mol'; } },
      { key: 'mfe_per_nt', label: 'MFE per nt', fmt: function (v) { return v.toFixed(3) + ' kcal/nt'; } },
      { key: 'cai_human', label: 'CAI', fmt: function (v) { return v.toFixed(3); } },
      { key: 'ires_deviation_L2_clamped', label: 'IRES deviation', fmt: function (v) { return (v * 100).toFixed(1) + '%'; } },
      { key: 'ires_length', label: 'IRES length', fmt: function (v) { return v + ' nt'; } },
      { key: 'cds_length', label: 'CDS length', fmt: function (v) { return v + ' nt'; } },
    ];
    for (var i = 0; i < items.length; i++) {
      var it = items[i], v = cd[it.key];
      if (v == null) continue;
      el.innerHTML += makeCard(it.label, it.fmt(v));
    }
  };

  /* ═══════════════ STRUCTURE TAB ═══════════════ */

  Panels.renderQualityMetrics = function (result) {
    if (!result) return;
    var phys = result.physical || {};
    var s3d = result.structural_3d || {};
    var metrics = [
      { barId: 'qm-closure-bar', valId: 'qm-closure-val', key: 'closure_distance_Ang', src: phys, unit: 'A',
        pctFn: function (v) { return Math.min(100, (1 - v / 30) * 100); },
        colorFn: function (v) { return v < 5 ? 'var(--ok)' : v < 12 ? 'var(--warn)' : 'var(--err)'; } },
      { barId: 'qm-bond-bar', valId: 'qm-bond-val', key: 'bond_rmsd_Ang', src: phys, unit: 'A',
        pctFn: function (v) { return Math.min(100, (1 - v / 5) * 100); },
        colorFn: function (v) { return v < 1 ? 'var(--ok)' : v < 2 ? 'var(--warn)' : 'var(--err)'; } },
      { barId: 'qm-pair-bar', valId: 'qm-pair-val', key: 'pair_satisfaction_rate', src: s3d, unit: '',
        pctFn: function (v) { return v * 100; },
        colorFn: function (v) { return v > 0.7 ? 'var(--ok)' : v > 0.4 ? 'var(--warn)' : 'var(--err)'; } },
      { barId: 'qm-clash-bar', valId: 'qm-clash-val', key: 'clash_count', src: result.structural_3d || {}, unit: '',
        pctFn: function (v) { return Math.max(0, 100 - v * 10); },
        colorFn: function (v) { return v < 5 ? 'var(--ok)' : v < 20 ? 'var(--warn)' : 'var(--err)'; } },
    ];
    for (var i = 0; i < metrics.length; i++) {
      var m = metrics[i];
      var bar = $(m.barId), val = $(m.valId);
      if (!bar || !val) continue;
      var v = m.src ? m.src[m.key] : null;
      if (v == null) { bar.style.width = '0%'; val.textContent = '--'; val.style.color = 'var(--t3)'; }
      else {
        var pct = Math.max(2, m.pctFn(v));
        bar.style.width = pct + '%';
        bar.style.background = m.colorFn(v);
        val.textContent = (typeof v === 'number' ? v.toFixed(2) : v) + (m.unit ? ' ' + m.unit : '');
        val.style.color = m.colorFn(v);
      }
    }
  };

  Panels.renderPairDist = function (result) {
    if (!result) return;
    var s3d = result.structural_3d || {};
    var dist = s3d.pair_distance_distribution || {};
    var canvas = $('pair-dist-chart');
    var legend = $('pair-dist-legend');
    if (!canvas) return;

    var segments = [];
    if (dist.satisfied_lt15A) segments.push({ value: dist.satisfied_lt15A.count || 0, color: 'var(--ok)', label: '<15A' });
    if (dist.partial_15_30A) segments.push({ value: dist.partial_15_30A.count || 0, color: 'var(--warn)', label: '15-30A' });
    if (dist.unsatisfied_gt30A) segments.push({ value: dist.unsatisfied_gt30A.count || 0, color: 'var(--err)', label: '>30A' });

    if (segments.length > 0) Charts.drawStackedBar(canvas, segments);

    if (legend) {
      legend.innerHTML = segments.map(function (s) {
        return '<span><span class="dot" style="background:' + s.color + '"></span>' + s.label + '</span>';
      }).join('');
    }
  };

  Panels.renderPairQuality = function (result) {
    if (!result) return;
    var pq = result.pairing_quality || {};
    var pr = result.pair_range || {};
    var el = $('pair-quality-cards');
    if (!el) return;

    var html = '<div class="pair-quality-row">';
    html += '<div class="pq-card"><div class="pq-val">' + (pq.wc_pairs || 0) + '</div><div class="pq-label">WC Pairs</div><div class="pq-pct">' + fmt(pq.wc_pct, 1) + '%</div></div>';
    html += '<div class="pq-card"><div class="pq-val">' + (pq.wobble_pairs || 0) + '</div><div class="pq-label">Wobble</div><div class="pq-pct">' + fmt(pq.wobble_pct, 1) + '%</div></div>';
    html += '</div>';

    if (pr.local_lt50 != null) {
      html += '<div class="pair-quality-row">';
      html += '<div class="pq-card"><div class="pq-val">' + (pr.local_lt50 || 0) + '</div><div class="pq-label">Local (<50nt)</div><div class="pq-pct">' + fmt(pr.local_pct, 1) + '%</div></div>';
      html += '<div class="pq-card"><div class="pq-val">' + (pr.medium_50_500 || 0) + '</div><div class="pq-label">Medium</div><div class="pq-pct">' + fmt(pr.medium_pct, 1) + '%</div></div>';
      html += '<div class="pq-card"><div class="pq-val">' + (pr.long_gt500 || 0) + '</div><div class="pq-label">Long (>500nt)</div><div class="pq-pct">' + fmt(pr.long_pct, 1) + '%</div></div>';
      html += '</div>';
    }
    el.innerHTML = html;
  };

  Panels.renderStats = function (result) {
    if (!result) return;
    var el = $('stats-cards');
    if (!el) return;
    el.innerHTML = '';
    var stats = [
      { k: 'Sequence Length', v: (result.length || 0) + ' nt' },
      { k: 'Method', v: result.method || '--' },
      { k: 'Closure Error', v: result.physical ? fmt(result.physical.closure_distance_Ang, 3) + ' A' : '--' },
      { k: 'Radius of Gyration', v: result.shape_3d ? fmt(result.shape_3d.rog_A, 1) + ' A' : '--' },
      { k: 'Pair Rate', v: result.structural_3d ? fmtPct(result.structural_3d.pair_satisfaction_rate) : '--' },
      { k: 'GC Content', v: result.sequence_composition ? fmt(result.sequence_composition.gc_pct, 1) + '%' : '--' },
    ];
    for (var i = 0; i < stats.length; i++) {
      el.innerHTML += makeCard(stats[i].k, stats[i].v);
    }
  };

  /* ═══════════════ IMMUNE TAB ═══════════════ */

  Panels.renderImmune = function (result) {
    if (!result) return;
    var ia = result.immune_activation || {};
    var ds = ia.dsRNA_ssRNA || {};

    // dsRNA/ssRNA donut
    var balanceEl = $('immune-balance');
    if (balanceEl && (ds.dsRNA_pct != null || ds.ssRNA_pct != null)) {
      var canvas = document.createElement('canvas');
      canvas.width = 100; canvas.height = 100;
      canvas.style.cssText = 'width:80px;height:80px';
      Charts.drawDonut(canvas, [
        { value: ds.dsRNA_pct || 0, color: '#4d8dff' },
        { value: ds.ssRNA_pct || 0, color: '#fbbf24' },
      ], { centerText: (ds.dsRNA_pct || 0).toFixed(0) + '%' });

      balanceEl.innerHTML = '';
      balanceEl.appendChild(canvas);
      var info = document.createElement('div');
      info.className = 'ib-info';
      info.innerHTML = '<div class="ib-row"><span class="ib-label">dsRNA</span><span class="ib-val" style="color:#4d8dff">' + (ds.dsRNA_nt || 0) + ' nt (' + fmt(ds.dsRNA_pct, 1) + '%)</span></div>' +
        '<div class="ib-row"><span class="ib-label">ssRNA</span><span class="ib-val" style="color:#fbbf24">' + (ds.ssRNA_nt || 0) + ' nt (' + fmt(ds.ssRNA_pct, 1) + '%)</span></div>';
      if (ds.note) info.innerHTML += '<div class="legend" style="margin-top:6px">' + ds.note + '</div>';
      balanceEl.appendChild(info);
    }

    // RIG-I motifs
    var rigi = ia.rigi_motif_to_bsj || {};
    var rigiEl = $('rigi-table');
    if (rigiEl && rigi.motifs && rigi.motifs.length > 0) {
      var html = '<table class="rigi-table"><thead><tr><th>Pos</th><th>Seq Dist</th><th>3D Dist</th><th>SASA</th></tr></thead><tbody>';
      for (var i = 0; i < rigi.motifs.length; i++) {
        var m = rigi.motifs[i];
        html += '<tr><td class="mono">' + m.pos + '</td><td>' + m.seq_dist + '</td><td class="mono">' + (m.dist_3d_A || '--') + ' A</td><td class="mono">' + fmt(m.sasa, 3) + '</td></tr>';
      }
      html += '</tbody></table>';
      rigiEl.innerHTML = html;
    }

    // Flexibility
    var flex = ia.local_flexibility || {};
    var flexEl = $('flexibility-cards');
    if (flexEl && flex.most_flexible) {
      var html2 = makeCard('Mean Flexibility', fmt(flex.mean_flexibility_A, 4) + ' A');
      for (var j = 0; j < Math.min(5, flex.most_flexible.length); j++) {
        var f = flex.most_flexible[j];
        var color = f.flexibility > 0.04 ? 'var(--err)' : f.flexibility > 0.02 ? 'var(--warn)' : 'var(--ok)';
        html2 += '<div class="flex-card"><span class="flex-pos">' + f.pos + '</span><span class="flex-region">' + f.region + '</span><span class="flex-val" style="color:' + color + '">' + f.flexibility.toFixed(4) + '</span></div>';
      }
      flexEl.innerHTML = html2;
    }

    // Receptor binding
    var rb = ia.receptor_binding || {};
    var rbEl = $('receptor-panel');
    if (rbEl) {
      var html3 = '<div class="receptor-row">';
      html3 += '<div class="receptor-card"><div class="rc-val">' + (rb.dsRNA_duplex_ends || 0) + '</div><div class="rc-label">dsRNA Ends</div></div>';
      html3 += '<div class="receptor-card"><div class="rc-val">' + (rb.ssRNA_GU_TLR7_8 || 0) + '</div><div class="rc-label">GU for TLR7/8</div></div>';
      html3 += '<div class="receptor-card"><div class="rc-val">' + fmt(rb.ssRNA_GU_sasa, 3) + '</div><div class="rc-label">GU SASA</div></div>';
      html3 += '</div>';
      if (rb.note) html3 += '<div class="legend">' + rb.note + '</div>';
      rbEl.querySelector('h2').insertAdjacentHTML('afterend', html3);
    }
  };

  /* ═══════════════ SEQUENCE TAB ═══════════════ */

  Panels.renderSequence = function (result) {
    if (!result) return;
    var sc = result.sequence_composition || {};
    var el = $('seq-comp-cards');
    if (el) {
      el.innerHTML = '';
      el.innerHTML += makeCard('Length', (sc.length || 0) + ' nt');
      el.innerHTML += makeCard('GC Content', fmt(sc.gc_pct, 1) + '%');
    }

    // Nucleotide bar
    var nucBar = $('nucleotide-bar');
    if (nucBar && sc.A != null) {
      var total = (sc.A || 0) + (sc.U || 0) + (sc.G || 0) + (sc.C || 0);
      if (total > 0) {
        nucBar.innerHTML = '<div class="nuc-comp-bar">' +
          '<div class="nuc-seg nuc-seg-A" style="flex:' + sc.A + '">A ' + sc.A + '</div>' +
          '<div class="nuc-seg nuc-seg-U" style="flex:' + sc.U + '">U ' + sc.U + '</div>' +
          '<div class="nuc-seg nuc-seg-G" style="flex:' + sc.G + '">G ' + sc.G + '</div>' +
          '<div class="nuc-seg nuc-seg-C" style="flex:' + sc.C + '">C ' + sc.C + '</div>' +
          '</div>';
      }
    }

    // GC by region
    var gcEl = $('gc-region-cards');
    if (gcEl) {
      gcEl.innerHTML = '';
      var regions = [
        { key: 'gc_5UTR', label: "5'UTR" },
        { key: 'gc_IRES', label: 'IRES' },
        { key: 'gc_CDS', label: 'CDS' },
      ];
      for (var i = 0; i < regions.length; i++) {
        var r = regions[i], v = sc[r.key];
        if (v == null) continue;
        gcEl.innerHTML += '<div class="gc-card"><div class="gc-region">' + r.label + '</div><div class="gc-val">' + fmt(v, 1) + '%</div></div>';
      }
    }

    // Per-residue energy heatmap
    var pr = result.per_residue || {};
    var heatmapCanvas = $('energy-heatmap');
    if (heatmapCanvas && pr.top_penalized && pr.top_penalized.length > 0) {
      // Build energy array from top penalized (sparse → full)
      var energyVals = new Array(result.length || 0).fill(0);
      for (var j = 0; j < pr.top_penalized.length; j++) {
        var tp = pr.top_penalized[j];
        if (tp.pos < energyVals.length) energyVals[tp.pos] = tp.energy;
      }
      Charts.drawHeatmapStrip(heatmapCanvas, energyVals, result.ires_bounds ? { IRES: result.ires_bounds, CDS: result.cds_bounds } : null);
    }

    // Top penalized list
    var topEl = $('energy-top-list');
    if (topEl && pr.top_penalized) {
      topEl.innerHTML = '';
      var maxE = 0;
      for (var k = 0; k < pr.top_penalized.length; k++) maxE = Math.max(maxE, pr.top_penalized[k].energy || 0);
      for (var k2 = 0; k2 < Math.min(10, pr.top_penalized.length); k2++) {
        var tp2 = pr.top_penalized[k2];
        var pct = maxE > 0 ? (tp2.energy / maxE * 100) : 0;
        topEl.innerHTML += '<div class="energy-top-item">' +
          '<span class="etp-pos">' + tp2.pos + '</span>' +
          '<span class="etp-region" data-region="' + tp2.region + '">' + tp2.region + '</span>' +
          '<span class="etp-energy">' + tp2.energy.toFixed(1) + '</span>' +
          '<span class="etp-bar"><span class="etp-bar-fill" style="width:' + pct + '%"></span></span></div>';
      }
    }
  };

  /* ═══════════════ THERMO TAB ═══════════════ */

  Panels.renderThermo = function (result) {
    if (!result) return;
    var sl = result.stem_loops || {};
    var thermo = result.thermodynamics || {};

    // Stem-loop stats
    var statsEl = $('stemloop-stats');
    if (statsEl) {
      statsEl.innerHTML = '';
      statsEl.innerHTML += makeCard('Count', sl.count || 0);
      statsEl.innerHTML += makeCard('Mean Stem Length', fmt(sl.mean_stem_length, 1) + ' bp');
      statsEl.innerHTML += makeCard('Mean Loop Length', fmt(sl.mean_loop_length, 1) + ' nt');
    }

    // Stem histogram
    if (sl.stem_lengths && sl.stem_lengths.length > 0) {
      Charts.drawHistogram($('stem-histogram'), sl.stem_lengths, { binWidth: 1, barColor: '#4d8dff', label: 'Stem Length (bp)' });
    }

    // Loop histogram
    if (sl.loop_lengths && sl.loop_lengths.length > 0) {
      Charts.drawHistogram($('loop-histogram'), sl.loop_lengths, { binWidth: 20, barColor: '#a855f7', label: 'Loop Length (nt)' });
    }

    // MFE cards
    var mfeEl = $('mfe-cards');
    if (mfeEl) {
      mfeEl.innerHTML = '';
      mfeEl.innerHTML += makeCard('MFE', fmt(thermo.mfe_kcal, 1) + ' kcal/mol');
      mfeEl.innerHTML += makeCard('MFE/nt', fmt(thermo.mfe_per_nt, 3) + ' kcal/nt');
    }
  };

  /* ═══════════════ SHAPE TAB ═══════════════ */

  Panels.renderShape = function (result) {
    if (!result) return;
    var sh = result.shape_3d || {};
    var hull = result.convex_hull || {};

    // Shape classification
    var classEl = $('shape-class-label');
    if (classEl) {
      var asc = sh.asphericity || 0;
      var prol = sh.prolateness || 0;
      var shapeName = 'Unknown';
      var shapeDesc = '';
      if (asc < 0.1 && Math.abs(prol) < 0.1) { shapeName = 'Globular'; shapeDesc = 'Compact, roughly spherical shape'; }
      else if (prol > 0.1) { shapeName = 'Prolate'; shapeDesc = 'Elongated along one axis'; }
      else if (prol < -0.1) { shapeName = 'Oblate'; shapeDesc = 'Flattened along one axis'; }
      else { shapeName = 'Anisotropic'; shapeDesc = 'Irregular three-axis shape'; }
      classEl.innerHTML = '<div class="shape-name">' + shapeName + '</div><div class="shape-desc">' + shapeDesc + '</div>';
    }

    // Metric cards
    var metricEl = $('shape-metric-cards');
    if (metricEl) {
      metricEl.innerHTML = '';
      metricEl.innerHTML += makeCard('Asphericity', fmt(sh.asphericity, 4));
      metricEl.innerHTML += makeCard('Prolateness', fmt(sh.prolateness, 4));
      metricEl.innerHTML += makeCard('Radius of Gyration', fmt(sh.rog_A, 1) + ' A');
      metricEl.innerHTML += makeCard('Shape Moment', fmt(sh.shape_moment, 4));
    }

    // Eigenvalue chart
    if (sh.eigenvalues && sh.eigenvalues.length > 0) {
      var eigenItems = sh.eigenvalues.map(function (v, i) {
        return { label: 'v' + (i + 1), value: v };
      });
      Charts.drawBarChart($('eigen-chart'), eigenItems);
    }

    // Hull cards
    var hullEl = $('hull-cards');
    if (hullEl) {
      hullEl.innerHTML = '';
      if (hull.volume_A3) hullEl.innerHTML += makeCard('Volume', fmt(hull.volume_A3, 1) + ' A^3');
      if (hull.surface_A2) hullEl.innerHTML += makeCard('Surface', fmt(hull.surface_A2, 1) + ' A^2');
    }
  };

  /* ═══════════════ MASTER RENDER ═══════════════ */

  Panels.renderAll = function (result) {
    if (!result) return;
    Panels.renderScoring(result);
    Panels.renderPhysical(result);
    Panels.renderCircDesign(result);
    Panels.renderQualityMetrics(result);
    Panels.renderPairDist(result);
    Panels.renderPairQuality(result);
    Panels.renderStats(result);
    Panels.renderImmune(result);
    Panels.renderSequence(result);
    Panels.renderThermo(result);
    Panels.renderShape(result);
  };

})();
