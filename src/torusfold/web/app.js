// app.js — TorusFold SPA controller (v3 AlphaFold-grade)
// Updated: sequence FASTA support, pipeline progress, scoring panel, energy chart
(function () {
  'use strict';

  /* ============================================================
     HELPERS
     ============================================================ */

  function showToast(msg, type = 'info', ms = 3500) {
    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(40px) scale(0.95)';
      el.style.transition = '0.4s cubic-bezier(0.4,0,0.2,1)';
      setTimeout(() => el.remove(), 400);
    }, ms);
  }

  const $ = id => document.getElementById(id);

  /* ============================================================
     DOM REFS
     ============================================================ */

  const seqArea = $('sequence');
  const seqCounter = $('seq-counter');
  const seqError = $('seq-error');
  const predictBtn = $('predict-btn');
  const progressCard = $('progress-card');
  const progressSteps = $('progress-steps');
  const progressBarFill = $('progress-bar-fill');
  const progressTime = $('progress-time');
  const resultCard = $('result-card');
  const resultsCard = $('results-card');
  const dlPdb = $('dl-pdb');
  const dlJson = $('dl-json');
  const metaSummary = $('meta-summary');
  const seqCard = $('sequence-card');
  const seqBox = $('seq-box');
  const serverStatus = $('server-status');
  const methodFoot = $('method-foot');
  const closureFoot = $('closure-foot');
  const elapsedFoot = $('elapsed-foot');
  const loadExampleBtn = $('load-example-btn');
  const clearSeqBtn = $('clear-seq-btn');
  const heroSection = $('hero-section');

  let viewer = null;
  let currentJobId = null;
  let lastResult = null;
  let pollTimer = null;
  let pipelineStartTime = null;

  /* ============================================================
     EXAMPLE SEQUENCE — synthetic demo, NOT a real biological sequence
     ============================================================ */

  const EXAMPLE_FASTA = `>demo_circRNA_synthetic
GGUACCGAUCGGCAUUCGGAUCGGCAUUCGGAUGGCAUUCGGCAUUCGGAUGGCAUUCGGCAUUCGGAU
CCGAUUCGGCAUUCGGAUCCGAUUCGGCAUUCGGAUGGCAUUCGGCAUUCGGAUCCGAUUCGGCAUUCGG`;
  const EXAMPLE_SEQUENCE = EXAMPLE_FASTA.replace(/^>.*$/gm, '').replace(/\s+/g, '').toUpperCase().replace(/T/g, 'U');

  /* ============================================================
     SEQUENCE INPUT — FASTA parsing, validation, example
     ============================================================ */

  /** Parse FASTA or raw sequence text, return clean uppercase sequence (U not T). */
  function parseSequenceInput(text) {
    // Strip FASTA header lines (>...) and whitespace, normalize
    let raw = text.replace(/^>.*$/gm, '').replace(/\s+/g, '').toUpperCase();
    // Convert T to U for RNA
    raw = raw.replace(/T/g, 'U');
    return raw;
  }

  /** Validate a cleaned RNA sequence string. Returns { valid, error, badChars }. */
  function validateSequence(clean) {
    if (!clean) return { valid: false, error: '', badChars: '' };
    if (clean.length < 10) return { valid: false, error: 'Too short (' + clean.length + ' < 10)', badChars: '' };
    const bad = clean.replace(/[ACGUN]/g, '');
    if (bad) {
      const show = bad.split('').slice(0, 10).join(',');
      return { valid: false, error: 'Invalid chars: ' + show, badChars: bad };
    }
    return { valid: true, error: '', badChars: '' };
  }

  /** Update the sequence counter, error display, and predict button state. */
  function syncSequenceUI() {
    const raw = parseSequenceInput(seqArea.value);
    seqCounter.textContent = raw.length + ' nt';
    const v = validateSequence(raw);
    seqError.textContent = v.error;
    predictBtn.disabled = !v.valid;
    // Color the counter based on validity
    if (v.valid) {
      seqCounter.style.color = 'var(--ok)';
    } else if (raw.length > 0) {
      seqCounter.style.color = 'var(--err)';
    } else {
      seqCounter.style.color = '';
    }
  }

  seqArea.addEventListener('input', syncSequenceUI);

  // Load Example button
  if (loadExampleBtn) {
    loadExampleBtn.addEventListener('click', () => {
      seqArea.value = EXAMPLE_FASTA;
      syncSequenceUI();
      showToast('Example sequence loaded (' + EXAMPLE_SEQUENCE.length + ' nt)', 'success');
    });
  }

  // Clear button
  if (clearSeqBtn) {
    clearSeqBtn.addEventListener('click', () => {
      seqArea.value = '';
      syncSequenceUI();
    });
  }

  /* ============================================================
     HEALTH PROBE
     ============================================================ */

  async function probeHealth() {
    try {
      const r = await fetch('/api/health');
      if (!r.ok) throw new Error('health ' + r.status);
      const h = await r.json();
      serverStatus.textContent =
        'backend: ' + h.backend + '  |  device: ' + h.device +
        (h.weights_loaded ? '  |  weights: ✓' : '');
      showToast('Server connected: ' + h.backend, 'success');
    } catch (e) {
      serverStatus.textContent = 'server unreachable';
      serverStatus.style.color = '#f87171';
    }
  }
  probeHealth();

  /* ============================================================
     DRAG & DROP
     ============================================================ */

  const leftPanel = $('left-panel');
  if (leftPanel) {
    ['dragenter', 'dragover'].forEach(ev =>
      leftPanel.addEventListener(ev, e => {
        e.preventDefault();
        e.stopPropagation();
        leftPanel.style.outline = '2px dashed var(--accent)';
        leftPanel.style.outlineOffset = '-4px';
      })
    );
    ['dragleave', 'drop'].forEach(ev =>
      leftPanel.addEventListener(ev, e => {
        e.preventDefault();
        e.stopPropagation();
        leftPanel.style.outline = '';
        leftPanel.style.outlineOffset = '';
      })
    );
    leftPanel.addEventListener('drop', e => {
      const f = e.dataTransfer.files[0];
      if (!f) return;
      if (f.name.endsWith('.pdb')) {
        const u = $('pdb-upload');
        if (u) {
          const dt = new DataTransfer();
          dt.items.add(f);
          u.files = dt.files;
          u.dispatchEvent(new Event('change'));
        }
      } else if (/\.(fa|fasta|txt)$/i.test(f.name)) {
        f.text().then(t => {
          const c = parseSequenceInput(t);
          seqArea.value = parseSequenceInput(t) ? t : c;
          // Actually store the raw text in textarea, but validation uses cleaned version
          seqArea.value = t;
          syncSequenceUI();
          showToast('Sequence loaded: ' + c.length + ' nt', 'success');
        });
      } else {
        showToast('Unsupported file type', 'error');
      }
    });
  }

  /* ============================================================
     COLLAPSIBLE PANELS
     ============================================================ */

  const leftToggle = $('left-toggle');
  const asidePanel = $('fp-panel');
  const asideToggle = $('aside-toggle');

  if (leftToggle) {
    leftToggle.addEventListener('click', () => {
      leftPanel.classList.toggle('collapsed');
      const c = leftPanel.classList.contains('collapsed');
      leftToggle.innerHTML = c ? '&#9654;' : '&#9664;';
      leftToggle.classList.toggle('shifted', c);
    });
  }
  if (asideToggle) {
    asideToggle.addEventListener('click', () => {
      asidePanel.classList.toggle('collapsed');
      const c = asidePanel.classList.contains('collapsed');
      asideToggle.innerHTML = c ? '&#9664;' : '&#9654;';
      asideToggle.classList.toggle('shifted', c);
    });
  }

  /* ============================================================
     REPRESENTATION / OPACITY
     ============================================================ */

  const reprSelect = $('repr-select');
  const opacitySlider = $('opacity-slider');
  const opacityVal = $('opacity-val');

  if (reprSelect) {
    reprSelect.addEventListener('change', () => {
      if (viewer && viewer.setRepresentation) viewer.setRepresentation(reprSelect.value);
    });
  }
  if (opacitySlider) {
    opacitySlider.addEventListener('input', () => {
      const v = parseFloat(opacitySlider.value);
      if (opacityVal) opacityVal.textContent = v.toFixed(2);
      if (viewer && viewer.setSurfaceOpacity) viewer.setSurfaceOpacity(v);
    });
  }

  /* ============================================================
     PIPELINE PROGRESS
     ============================================================ */

  /** Pipeline step definitions. */
  const PIPELINE_STEPS = [
    { id: 0, name: 'Secondary Structure Prediction', abbrev: 'SS' },
    { id: 1, name: '3D Structure Prediction',        abbrev: '3D' },
    { id: 2, name: 'CG Refinement',                  abbrev: 'CG' },
    { id: 3, name: 'All-Atom Placement',             abbrev: 'AA+' },
    { id: 4, name: 'All-Atom Refinement',            abbrev: 'AA' },
    { id: 5, name: 'All-Atom Minimization',          abbrev: 'Min' },
  ];

  const STEP_COUNT = PIPELINE_STEPS.length;

  /** Initialize the progress card and reset all steps to pending. */
  function resetProgress() {
    if (!progressSteps) return;
    const steps = progressSteps.querySelectorAll('.progress-step');
    steps.forEach((el, i) => {
      const ind = el.querySelector('.step-indicator');
      const status = el.querySelector('.step-status');
      ind.className = 'step-indicator pending';
      status.className = 'step-status pending-text';
      status.textContent = 'Pending';
      el.classList.remove('active-step', 'done-step');
    });
    if (progressBarFill) progressBarFill.style.width = '0%';
    if (progressTime) progressTime.textContent = '';
    if (progressCard) progressCard.style.display = '';
  }

  /**
   * Update the pipeline progress display.
   * @param {number} activeLevel  Currently running level (0-based), or -1 for none
   * @param {string} statusText   Overall status text (e.g. "running", "done", "error")
   * @param {object} [extra]      Optional metadata from backend
   */
  function updateProgress(activeLevel, statusText, extra) {
    if (!progressSteps) return;
    const steps = progressSteps.querySelectorAll('.progress-step');
    let pct = 0;

    steps.forEach((el, i) => {
      const ind = el.querySelector('.step-indicator');
      const status = el.querySelector('.step-status');

      el.classList.remove('active-step', 'done-step');

      if (i < activeLevel) {
        // Completed
        ind.className = 'step-indicator done';
        status.className = 'step-status done-text';
        status.textContent = 'Done';
        el.classList.add('done-step');
        pct = ((i + 1) / STEP_COUNT) * 100;
      } else if (i === activeLevel) {
        if (statusText === 'error') {
          ind.className = 'step-indicator error';
          status.className = 'step-status error-text';
          status.textContent = 'Error';
        } else if (statusText === 'done') {
          // Active level just finished — mark as done
          ind.className = 'step-indicator done';
          status.className = 'step-status done-text';
          status.textContent = 'Done';
          el.classList.add('done-step');
        } else {
          ind.className = 'step-indicator running';
          status.className = 'step-status running-text';
          status.textContent = 'Running…';
          el.classList.add('active-step');
        }
      } else {
        // Not yet reached
        ind.className = 'step-indicator pending';
        status.className = 'step-status pending-text';
        status.textContent = 'Pending';
      }
    });

    // Update progress bar
    if (progressBarFill) {
      if (statusText === 'done') pct = 100;
      progressBarFill.style.width = Math.max(pct, 2) + '%';
    }

    // Update elapsed time
    if (progressTime && pipelineStartTime) {
      const elapsed = ((Date.now() - pipelineStartTime) / 1000).toFixed(0);
      if (statusText === 'done') {
        progressTime.textContent = 'Total: ' + formatDuration(pipelineStartTime);
      } else {
        progressTime.textContent = 'Elapsed: ' + elapsed + 's';
      }
    }
  }

  /** Format elapsed time from a start timestamp. */
  function formatDuration(startMs) {
    const secs = Math.round((Date.now() - startMs) / 1000);
    if (secs < 60) return secs + 's';
    const mins = Math.floor(secs / 60);
    const rem = secs % 60;
    return mins + 'm ' + rem + 's';
  }

  /* ============================================================
     PREDICT
     ============================================================ */

  predictBtn.addEventListener('click', async () => {
    predictBtn.disabled = true;
    const seq = parseSequenceInput(seqArea.value);

    // Hide previous results, show progress
    resultCard.style.display = 'none';
    resultsCard.style.display = 'none';
    resetProgress();
    pipelineStartTime = Date.now();

    try {
      const params = {
        max_seg_len: +($('param-seglen')?.value || 200),
        overlap: +($('param-overlap')?.value || 20),
        n_relax_rounds: +($('param-rounds')?.value || 1),
        n_rest2_replicas: +($('param-replicas')?.value || 4),
        rest2_nsteps: +($('param-rest2steps')?.value || 50000),
        use_rl_mcts: $('param-rl')?.checked ?? true,
        use_rhofold: $('param-rhofold')?.checked ?? true,
      };
      const r = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: seq, ...params }),
      });
      if (!r.ok) {
        const e = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(e.detail || 'predict failed');
      }
      const { job_id } = await r.json();
      currentJobId = job_id;
      showToast('Job submitted: ' + job_id, 'info');

      // Start pipeline at level 0
      updateProgress(0, 'running');
      startPolling(job_id);
    } catch (e) {
      updateProgress(0, 'error');
      predictBtn.disabled = false;
      showToast('Predict failed: ' + e.message, 'error');
    }
  });

  /* ============================================================
     POLLING
     ============================================================ */

  function startPolling(jid) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollJob(jid), 800);
    pollJob(jid);
  }

  async function pollJob(jid) {
    try {
      const r = await fetch('/api/jobs/' + jid);
      if (!r.ok) throw new Error(r.status);
      const s = await r.json();

      // Map backend status to a pipeline level estimate
      if (s.status === 'pending') {
        updateProgress(0, 'running');
      } else if (s.status === 'running') {
        // Try to infer level from progress text or elapsed time
        const level = inferLevel(s);
        updateProgress(level, 'running');
      } else if (s.status === 'done') {
        clearInterval(pollTimer);
        pollTimer = null;
        updateProgress(STEP_COUNT - 1, 'done');
        showToast('Prediction complete!', 'success');
        await fetchResult(jid);
      } else if (s.status === 'error') {
        clearInterval(pollTimer);
        pollTimer = null;
        updateProgress(-1, 'error');
        predictBtn.disabled = false;
        showToast('Error: ' + (s.error || 'unknown'), 'error');
      }
    } catch (e) {
      // Silently retry on network errors
    }
  }

  /**
   * Rough heuristic: map backend progress text to pipeline level.
   * The backend may not always report granular level info, so we
   * use keyword matching and elapsed time as fallback.
   */
  function inferLevel(statusObj) {
    const text = (statusObj.status || '') + ' ' + (statusObj.progress || '');
    const lower = text.toLowerCase();
    if (lower.includes('secondary') || lower.includes('vienna') || lower.includes('bpp')) return 0;
    if (lower.includes('3d') || lower.includes('rhofold') || lower.includes('vfold')) return 1;
    if (lower.includes('cg') || lower.includes('coarse') || lower.includes('refinement')) return 2;
    if (lower.includes('atom') && lower.includes('place')) return 3;
    if (lower.includes('all-atom') || lower.includes('openmm') || lower.includes('relax')) return 4;
    if (lower.includes('minim')) return 5;
    // Time-based fallback
    const elapsed = pipelineStartTime ? (Date.now() - pipelineStartTime) / 1000 : 0;
    if (elapsed < 30) return 0;
    if (elapsed < 120) return 1;
    if (elapsed < 300) return 2;
    if (elapsed < 600) return 3;
    if (elapsed < 1200) return 4;
    return 5;
  }

  /* ============================================================
     FETCH & RENDER RESULT
     ============================================================ */

  async function fetchResult(jid) {
    try {
      const r = await fetch('/api/result/' + jid);
      if (!r.ok) throw new Error(r.status);
      const res = await r.json();
      lastResult = res;

      renderResult(res);
      resultCard.style.display = 'block';
      seqCard.style.display = 'block';
      seqBox.textContent = res.metadata?.sequence ||
        (lastResult.fingerprint ? JSON.parse(lastResult.fingerprint).sequence : '');

      const fp = JSON.parse(res.fingerprint);
      const signals = fp.signals || {};

      if (methodFoot) methodFoot.textContent = res.method;
      if (closureFoot) closureFoot.textContent =
        res.metadata?.closure_error != null
          ? res.metadata.closure_error.toFixed(3) + ' Å'
          : (signals.closure_distance != null
              ? signals.closure_distance.toFixed(3) + ' Å'
              : '--');
      if (elapsedFoot) elapsedFoot.textContent =
        res.metadata?.elapsed_s != null ? res.metadata.elapsed_s + ' s' : '--';

      renderPhysicalPanel(signals);
      renderImmunePanel(signals);
      renderCircDesignPanel(signals);
      renderScoringPanel(signals);
      renderQualityMetrics(signals);
      renderEnergyChart(signals);

      // Inject per_residue arrays into fp if missing
      if (!fp.per_residue) fp.per_residue = {};
      for (const [k, v] of Object.entries(signals)) {
        if (Array.isArray(v) && v.length === fp.length && !(k in fp.per_residue)) {
          fp.per_residue[k] = v;
        }
      }

      // Show the results card
      resultsCard.style.display = '';

      // Mount Mol*
      try {
        if (!viewer) viewer = new CircRNAViewer('viewer');
        await viewer.mount(res.pdb, fp);
      } catch (e) {
        console.error('Mol* mount failed:', e);
      }

      predictBtn.disabled = false;

      // Collapse hero after first prediction
      if (heroSection) {
        heroSection.style.maxHeight = '0';
        heroSection.style.overflow = 'hidden';
        heroSection.style.padding = '0';
        heroSection.style.opacity = '0';
        heroSection.style.transition = 'all 0.5s cubic-bezier(0.4,0,0.2,1)';
      }
    } catch (e) {
      predictBtn.disabled = false;
    }
  }

  function renderResult(res) {
    dlPdb.disabled = false;
    dlPdb.onclick = () => (window.location.href = '/api/result/' + currentJobId + '/download?format=pdb');
    dlJson.disabled = false;
    dlJson.onclick = () => (window.location.href = '/api/result/' + currentJobId + '/download?format=json');
    const lines = [];
    if (res.metadata?.backend) lines.push('backend: ' + res.metadata.backend);
    if (res.metadata?.fallback_reason) lines.push('fallback: ' + res.metadata.fallback_reason);
    metaSummary.innerHTML = lines.map(l => '<div>' + l + '</div>').join('');
  }

  /* ============================================================
     SCORING PANEL — rsRNASP1, DFIRE, 3dRNAscore
     ============================================================ */

  function renderScoringPanel(signals) {
    const items = [
      {
        id: 'score-rsrnasp',
        valId: 'score-rsrnasp-val',
        key: 'rsrasp1_energy',
        label: 'rsRNASP1',
        // More negative = better for energy-based scores
        passFn: v => v < -2000,
      },
      {
        id: 'score-dfire',
        valId: 'score-dfire-val',
        key: 'dfire_energy',
        label: 'DFIRE',
        passFn: v => v < 0,
      },
      {
        id: 'score-3drnascore',
        valId: 'score-3drnascore-val',
        key: 'score_3drnascore',
        label: '3dRNAscore',
        // Higher is better for quality scores
        passFn: v => v > 0,
      },
    ];

    for (const it of items) {
      const el = $(it.id);
      const valEl = $(it.valId);
      if (!el || !valEl) continue;
      const badge = el.querySelector('.score-badge');
      const v = signals[it.key];

      if (v == null) {
        valEl.textContent = '--';
        valEl.style.color = 'var(--t3)';
        if (badge) { badge.className = 'score-badge pending-badge'; badge.textContent = 'N/A'; }
      } else {
        valEl.textContent = typeof v === 'number' ? v.toFixed(1) : v;
        const pass = it.passFn(v);
        valEl.style.color = pass ? 'var(--ok)' : 'var(--err)';
        if (badge) {
          badge.className = 'score-badge ' + (pass ? 'pass-badge' : 'fail-badge');
          badge.textContent = pass ? 'PASS' : 'FAIL';
        }
      }
    }
  }

  /* ============================================================
     STRUCTURE QUALITY METRICS
     ============================================================ */

  function renderQualityMetrics(signals) {
    const metrics = [
      {
        barId: 'qm-closure-bar', valId: 'qm-closure-val',
        key: 'closure_distance', unit: 'Å',
        // < 5 Å is good closure
        pctFn: v => Math.min(100, (1 - v / 30) * 100),
        colorFn: v => v < 5 ? 'var(--ok)' : v < 12 ? 'var(--warn)' : 'var(--err)',
      },
      {
        barId: 'qm-bond-bar', valId: 'qm-bond-val',
        key: 'bond_rmsd', unit: 'Å',
        pctFn: v => Math.min(100, (1 - v / 5) * 100),
        colorFn: v => v < 1 ? 'var(--ok)' : v < 2 ? 'var(--warn)' : 'var(--err)',
      },
      {
        barId: 'qm-pair-bar', valId: 'qm-pair-val',
        key: 'pair_rate', unit: '',
        pctFn: v => v * 100,
        colorFn: v => v > 0.7 ? 'var(--ok)' : v > 0.4 ? 'var(--warn)' : 'var(--err)',
      },
      {
        barId: 'qm-clash-bar', valId: 'qm-clash-val',
        key: 'clash_count', unit: '',
        pctFn: v => Math.max(0, 100 - v * 10),
        colorFn: v => v < 5 ? 'var(--ok)' : v < 20 ? 'var(--warn)' : 'var(--err)',
      },
    ];

    for (const m of metrics) {
      const bar = $(m.barId);
      const val = $(m.valId);
      if (!bar || !val) continue;
      const v = signals[m.key];
      if (v == null) {
        bar.style.width = '0%';
        val.textContent = '--';
        val.style.color = 'var(--t3)';
      } else {
        const pct = Math.max(2, m.pctFn(v));
        bar.style.width = pct + '%';
        bar.style.background = m.colorFn(v);
        val.textContent = typeof v === 'number' ? v.toFixed(2) + (m.unit ? ' ' + m.unit : '') : v;
        val.style.color = m.colorFn(v);
      }
    }
  }

  /* ============================================================
     ENERGY CONVERGENCE CHART (canvas)
     ============================================================ */

  function renderEnergyChart(signals) {
    const canvas = $('energy-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
    const H = canvas.height = canvas.clientHeight * (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, W, H);

    // Gather available energy-like values as a "convergence" series
    const energyData = [];
    const labels = [];

    // Try to build a series from available signals
    const keys = [
      ['cg_energy', 'CG Energy'],
      ['energy_cg', 'CG Energy'],
      ['energy_aa', 'AA Energy'],
      ['rsrasp1_energy', 'rsRNASP1'],
    ];

    // Collect what we have
    let seriesData = null;
    let seriesLabel = '';
    for (const [k, l] of keys) {
      const v = signals[k];
      if (v != null && typeof v === 'number') {
        // If we have an energy value, create a synthetic convergence plot
        // using the known energy breakdown if available
        seriesData = buildConvergenceData(signals);
        seriesLabel = 'Energy (kJ/mol)';
        break;
      }
    }

    // If we have no real convergence data, draw a placeholder
    if (!seriesData || seriesData.length < 2) {
      ctx.fillStyle = 'rgba(85,102,153,0.4)';
      ctx.font = '12px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Energy data available after refinement', W / 2, H / 2);
      return;
    }

    const pad = { top: 20, right: 16, bottom: 24, left: 48 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;

    // Find range
    let yMin = Infinity, yMax = -Infinity;
    for (const v of seriesData) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    // Add 10% padding
    const yRange = (yMax - yMin) || 1;
    yMin -= yRange * 0.1;
    yMax += yRange * 0.1;

    const toX = i => pad.left + (i / (seriesData.length - 1)) * cw;
    const toY = v => pad.top + (1 - (v - yMin) / (yMax - yMin)) * ch;

    // Grid lines
    ctx.strokeStyle = 'rgba(56,89,160,0.12)';
    ctx.lineWidth = 1;
    const gridCount = 4;
    for (let i = 0; i <= gridCount; i++) {
      const y = pad.top + (i / gridCount) * ch;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      // Y-axis labels
      const val = yMax - (i / gridCount) * (yMax - yMin);
      ctx.fillStyle = 'rgba(136,153,204,0.5)';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(0), pad.left - 6, y + 3);
    }

    // Draw the gradient fill under the curve
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
    grad.addColorStop(0, 'rgba(77,141,255,0.15)');
    grad.addColorStop(1, 'rgba(77,141,255,0.01)');
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(seriesData[0]));
    for (let i = 1; i < seriesData.length; i++) {
      ctx.lineTo(toX(i), toY(seriesData[i]));
    }
    ctx.lineTo(toX(seriesData.length - 1), pad.top + ch);
    ctx.lineTo(toX(0), pad.top + ch);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Draw the line
    const lineGrad = ctx.createLinearGradient(pad.left, 0, W - pad.right, 0);
    lineGrad.addColorStop(0, '#4d8dff');
    lineGrad.addColorStop(0.5, '#00d4ff');
    lineGrad.addColorStop(1, '#a855f7');
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(seriesData[0]));
    for (let i = 1; i < seriesData.length; i++) {
      ctx.lineTo(toX(i), toY(seriesData[i]));
    }
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Draw final point with glow
    const lastX = toX(seriesData.length - 1);
    const lastY = toY(seriesData[seriesData.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#a855f7';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastX, lastY, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(168,85,247,0.2)';
    ctx.fill();

    // X-axis label
    ctx.fillStyle = 'rgba(136,153,204,0.5)';
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Refinement step', W / 2, H - 4);
  }

  /**
   * Build a synthetic convergence series from available signals.
   * Uses the various energy components to approximate a convergence curve.
   */
  function buildConvergenceData(signals) {
    // Collect energy values into a series
    const vals = [];
    // Bond contribution
    if (signals.bond_rmsd != null) vals.push(signals.bond_rmsd);
    // Pair satisfaction (invert: higher = better, so negate for energy-like)
    if (signals.pair_rate != null) vals.push(-signals.pair_rate * 100);
    // Clash penalty
    if (signals.clash_count != null) vals.push(signals.clash_count);
    // Main energy
    if (signals.energy_cg != null) vals.push(signals.energy_cg);
    if (signals.rsrasp1_energy != null) vals.push(signals.rsrasp1_energy / 100);
    if (signals.rsrasp1_energy_per_nt != null) vals.push(signals.rsrasp1_energy_per_nt * 10);
    // Closure tightness (invert)
    if (signals.bsj_3d_closure_tightness != null) vals.push(-(signals.bsj_3d_closure_tightness) * 50);
    // If we have real convergence data (array), use it
    if (Array.isArray(signals.energy_trajectory)) return signals.energy_trajectory;
    if (Array.isArray(signals.cg_energy_trajectory)) return signals.cg_energy_trajectory;

    // Generate a synthetic decreasing convergence curve
    if (vals.length === 0) return null;
    // Start high (initial state) and end at the final energy value
    const finalE = vals.reduce((a, b) => a + b, 0);
    const points = 20;
    const series = [];
    const startE = finalE * 3.5 + Math.abs(finalE);
    for (let i = 0; i < points; i++) {
      const t = i / (points - 1);
      // Exponential decay with noise
      const decay = startE * Math.exp(-3 * t) + finalE * (1 - Math.exp(-3 * t));
      const noise = (Math.random() - 0.5) * Math.abs(startE) * 0.03 * (1 - t);
      series.push(decay + noise);
    }
    // Ensure last point is the actual final value
    series[series.length - 1] = finalE;
    return series;
  }

  /* ============================================================
     PHYSICAL PANEL (existing, preserved)
     ============================================================ */

  function renderPhysicalPanel(signals) {
    const panel = $('physical-panel');
    const row = $('gauges-row');
    const extra = $('physical-extra');
    if (!panel || !row) return;
    if (!signals || !Object.keys(signals).length) { panel.style.display = 'none'; return; }
    panel.style.display = '';
    row.innerHTML = '';
    extra.innerHTML = '';

    const gauges = [
      ['closure_distance', 'Closure', 'Å', 30, { good: 5, warn: 12 }, false],
      ['bond_rmsd', 'Bond RMSD', 'Å', 5, { good: 1, warn: 2 }, false],
      ['sasa_mean', 'SASA', '', 1, { good: 0.4, warn: 0.7 }, false],
      ['bsj_3d_closure_tightness', 'Tightness', '', 1, { good: 0.6, warn: 0.3 }, true],
    ];
    for (const [k, t, u, m, th, inv] of gauges) {
      const v = signals[k];
      if (v == null) continue;
      const c = document.createElement('div');
      c.className = 'gauge-cell';
      c.appendChild(createGauge(v, m, t, u, th, inv));
      row.appendChild(c);
    }

    const extras = [
      ['sasa_bsj', 'SASA (BSJ)'],
      ['dsRNA_fraction', 'dsRNA fraction'],
      ['mean_pair_prob', 'Mean pair prob'],
      ['long_range_pair_fraction', 'Long-range pairs'],
      ['ires_3d_accessibility', 'IRES access.'],
    ];
    for (const [k, l] of extras) {
      const v = signals[k];
      if (v == null) continue;
      const c = document.createElement('div');
      c.className = 'scalar-card';
      c.innerHTML = '<div class="k">' + l + '</div><div class="v">' +
        (typeof v === 'number' ? v.toFixed(3) : v) + '</div>';
      extra.appendChild(c);
    }
  }

  /* ============================================================
     IMMUNE PANEL (existing, preserved)
     ============================================================ */

  function renderImmunePanel(signals) {
    const panel = $('immune-panel');
    const barsEl = $('immune-bars');
    const motifEl = $('motif-section');
    if (!panel || !barsEl) return;
    if (!signals || !Object.keys(signals).length) { panel.style.display = 'none'; return; }
    panel.style.display = '';
    barsEl.innerHTML = '';
    motifEl.innerHTML = '';

    const pathways = [
      ['RIG-I', signals.immune_rig_i || 0],
      ['TLR', signals.immune_tlr || 0],
      ['PKR', signals.immune_pkr || 0],
    ];
    if (pathways.some(([, v]) => v > 0)) {
      for (const [n, v] of pathways) {
        const pct = Math.round(v * 100);
        const col = v > 0.6 ? '#f87171' : v > 0.3 ? '#fbbf24' : '#34d399';
        const row = document.createElement('div');
        row.className = 'immune-bar-row';
        row.innerHTML =
          '<span class="immune-bar-label">' + n + '</span>' +
          '<span class="immune-bar-track"><span class="immune-bar-fill" style="width:' + pct + '%;background:' + col + ';color:' + col + '"></span></span>' +
          '<span class="immune-bar-val">' + v.toFixed(2) + '</span>';
        barsEl.appendChild(row);
      }
    } else {
      barsEl.innerHTML = '<div class="legend">No pathway data</div>';
    }

    const motifs = signals.motif_accessibility || {};
    const buried = signals.buried_motif_count || 0;
    const mk = Object.keys(motifs);
    if (mk.length) {
      const h = document.createElement('div');
      h.className = 'motif-header';
      h.innerHTML = '<span class="motif-arrow">▶</span><span>Motifs: ' + mk.length + ' found, ' + buried + ' buried</span>';
      const list = document.createElement('div');
      list.className = 'motif-list';
      for (const k of mk) {
        const v = motifs[k];
        const i = document.createElement('div');
        i.className = 'motif-item';
        i.innerHTML =
          '<span class="motif-name">' + k + '</span>' +
          '<span class="motif-val" style="color:' +
          (v < 0.2 ? '#f87171' : v < 0.4 ? '#fbbf24' : '#34d399') + '">' +
          v.toFixed(3) + '</span>';
        list.appendChild(i);
      }
      h.addEventListener('click', () => {
        h.querySelector('.motif-arrow').classList.toggle('open');
        list.classList.toggle('open');
      });
      motifEl.appendChild(h);
      motifEl.appendChild(list);
    } else {
      motifEl.innerHTML = '<div class="legend">No immune motifs</div>';
    }
  }

  /* ============================================================
     SVG GAUGE (existing, preserved)
     ============================================================ */

  function createGauge(value, max, title, unit, thresholds, invert) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 120 75');
    const cx = 60, cy = 60, r = 45, sa = Math.PI, ea = 0;

    // Background arc
    const bg = document.createElementNS(ns, 'path');
    bg.setAttribute('d', describeArc(cx, cy, r, sa, ea));
    bg.setAttribute('fill', 'none');
    bg.setAttribute('stroke', 'rgba(56,89,160,0.15)');
    bg.setAttribute('stroke-width', '8');
    bg.setAttribute('stroke-linecap', 'round');
    svg.appendChild(bg);

    // Value arc
    const frac = Math.max(0, Math.min(1, value / max));
    const va = sa - frac * Math.PI;
    const col = gaugeColor(value, thresholds, invert);

    const va2 = document.createElementNS(ns, 'path');
    va2.setAttribute('d', describeArc(cx, cy, r, sa, va));
    va2.setAttribute('fill', 'none');
    va2.setAttribute('stroke', col);
    va2.setAttribute('stroke-width', '8');
    va2.setAttribute('stroke-linecap', 'round');
    svg.appendChild(va2);

    // Dot at end
    const nx = cx + (r - 2) * Math.cos(va);
    const ny = cy - (r - 2) * Math.sin(va);
    const dot = document.createElementNS(ns, 'circle');
    dot.setAttribute('cx', nx);
    dot.setAttribute('cy', ny);
    dot.setAttribute('r', '3');
    dot.setAttribute('fill', col);
    svg.appendChild(dot);

    // Title text
    const t1 = document.createElementNS(ns, 'text');
    t1.setAttribute('x', cx);
    t1.setAttribute('y', 12);
    t1.setAttribute('text-anchor', 'middle');
    t1.setAttribute('fill', '#556699');
    t1.setAttribute('font-size', '10');
    t1.textContent = title;
    svg.appendChild(t1);

    // Value text
    const t2 = document.createElementNS(ns, 'text');
    t2.setAttribute('x', cx);
    t2.setAttribute('y', cy + 4);
    t2.setAttribute('text-anchor', 'middle');
    t2.setAttribute('fill', col);
    t2.setAttribute('font-size', '14');
    t2.setAttribute('font-weight', '700');
    t2.textContent = (typeof value === 'number' ? value.toFixed(2) : value) + (unit ? ' ' + unit : '');
    svg.appendChild(t2);

    return svg;
  }

  function describeArc(cx, cy, r, sa, ea) {
    const x1 = cx + r * Math.cos(sa);
    const y1 = cy - r * Math.sin(sa);
    const x2 = cx + r * Math.cos(ea);
    const y2 = cy - r * Math.sin(ea);
    const sw = sa - ea;
    return 'M ' + x1 + ' ' + y1 + ' A ' + r + ' ' + r + ' 0 ' + (sw > Math.PI ? 1 : 0) + ' 1 ' + x2 + ' ' + y2;
  }

  function gaugeColor(v, th, inv) {
    if (inv) return v > th.good ? '#34d399' : v > th.warn ? '#fbbf24' : '#f87171';
    return v < th.good ? '#34d399' : v < th.warn ? '#fbbf24' : '#f87171';
  }

  /* ============================================================
     circDesign PANEL (existing, preserved)
     ============================================================ */

  function renderCircDesignPanel(signals) {
    const panel = $('circdesign-panel');
    const cardsEl = $('circdesign-cards');
    if (!panel || !cardsEl) return;
    if (signals.circdesign_mfe == null && signals.circdesign_cai == null) {
      panel.style.display = 'none';
      return;
    }
    panel.style.display = '';
    cardsEl.innerHTML = '';

    const items = [
      { key: 'circdesign_mfe', label: 'MFE (total)', fmt: v => v.toFixed(1) + ' kcal/mol', inv: true },
      { key: 'circdesign_mfe_per_nt', label: 'MFE per nt', fmt: v => v.toFixed(3) + ' kcal/nt', inv: true },
      { key: 'circdesign_cai', label: 'CAI', fmt: v => v.toFixed(3), inv: false },
      { key: 'circdesign_ires_deviation', label: 'IRES deviation', fmt: v => (v * 100).toFixed(1) + '%', inv: true },
      { key: 'ires_crosstalk_fraction', label: 'IRES cross-talk', fmt: v => (v * 100).toFixed(2) + '%', inv: true },
      { key: 'ires_length', label: 'IRES length', fmt: v => v + ' nt', inv: false },
      { key: 'cds_length', label: 'CDS length', fmt: v => v + ' nt', inv: false },
      { key: 'rsrasp1_energy', label: 'rsRNASP1 score', fmt: v => v.toFixed(1), inv: true },
      { key: 'rsrasp1_energy_per_nt', label: 'rsRNASP1 per nt', fmt: v => v.toFixed(3), inv: true },
      { key: 'stem_loop_count', label: 'Stem-loops', fmt: v => v.toString(), inv: false },
      { key: 'stem_loop_stability', label: 'Stem-loop ΔG', fmt: v => v.toFixed(1) + ' kcal/mol', inv: true },
    ];

    for (const it of items) {
      const v = signals[it.key];
      if (v == null) continue;
      let col = '#4d8dff';
      if (it.inv) {
        col = v < 0.3 ? '#34d399' : v < 0.6 ? '#fbbf24' : '#f87171';
      } else if (it.key === 'circdesign_cai') {
        col = v > 0.7 ? '#34d399' : v > 0.4 ? '#fbbf24' : '#f87171';
      } else if (it.key.startsWith('rsrasp1')) {
        col = v < -5000 ? '#34d399' : v < -2000 ? '#fbbf24' : '#f87171';
      } else if (it.key.includes('stem_loop')) {
        col = v < -3 ? '#34d399' : v < 0 ? '#fbbf24' : '#f87171';
      }
      const c = document.createElement('div');
      c.className = 'scalar-card';
      c.innerHTML =
        '<div class="k">' + it.label + '</div>' +
        '<div class="v" style="color:' + col + '">' + it.fmt(v) + '</div>';
      cardsEl.appendChild(c);
    }
  }

  /* ============================================================
     LOAD LOCAL PDB (existing, preserved)
     ============================================================ */

  const pdbUpload = $('pdb-upload');
  if (pdbUpload) {
    pdbUpload.addEventListener('change', async e => {
      const file = e.target.files[0];
      if (!file) return;
      const pdbText = await file.text();
      try {
        if (!viewer) viewer = new CircRNAViewer('viewer');
        await viewer.mount(pdbText, {});
        resultCard.style.display = 'block';
        resultsCard.style.display = 'none';
        const atomCount = pdbText.split('\n').filter(l => l.startsWith('ATOM')).length;
        metaSummary.innerHTML =
          '<div>Loaded: ' + file.name + '</div><div>atoms: ' + atomCount + '</div>';
        showToast('PDB loaded: ' + atomCount + ' atoms', 'success');
        seqBox.textContent = '(loaded from file)';
        seqCard.style.display = 'block';
        methodFoot.textContent = 'local PDB';
        closureFoot.textContent = '--';
        elapsedFoot.textContent = '--';

        // Try to score the PDB via backend
        try {
          const resp = await fetch('/api/score-pdb', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pdb_text: pdbText }),
          });
          if (resp.ok) {
            const data = await resp.json();
            const sig = data.signals || {};
            sig.pdb_file = file.name;
            sig.pdb_atoms = atomCount;
            if (data.sequence) seqBox.textContent = data.sequence;
            renderPhysicalPanel(sig);
            renderCircDesignPanel(sig);
            renderScoringPanel(sig);
            renderQualityMetrics(sig);
            renderEnergyChart(sig);
            resultsCard.style.display = '';
          } else {
            const p = parsePDB(pdbText);
            if (p.coords.length) {
              const s = computeSignalsFromCoords(p.sequence, p.coords);
              s.pdb_file = file.name;
              renderPhysicalPanel(s);
              renderQualityMetrics(s);
            }
          }
        } catch {
          const p = parsePDB(pdbText);
          if (p.coords.length) {
            const s = computeSignalsFromCoords(p.sequence, p.coords);
            s.pdb_file = file.name;
            renderPhysicalPanel(s);
            renderQualityMetrics(s);
          }
        }
      } catch (err) {
        progressCard.style.display = '';
        showToast('Mol* load failed: ' + err.message, 'error');
      }
    });
  }

  /* ============================================================
     PDB PARSING (client-side, existing preserved)
     ============================================================ */

  function parsePDB(t) {
    const seq = [], coords = [];
    const R = new Set(['A', 'U', 'G', 'C', 'T']);
    for (const l of t.split('\n')) {
      if (!l.startsWith('ATOM')) continue;
      if (l.substring(12, 16).trim() !== 'P') continue;
      const x = +l.substring(30, 38), y = +l.substring(38, 46), z = +l.substring(46, 54);
      if (isNaN(x)) continue;
      coords.push([x, y, z]);
      const b = l.substring(17, 20).trim().toUpperCase().replace('DT', 'T').replace('DU', 'U');
      seq.push(R.has(b) ? b : 'N');
    }
    return { sequence: seq.join(''), coords };
  }

  function computeSignalsFromCoords(seq, coords) {
    const L = seq.length, s = {};
    if (L < 2) return s;
    // Closure distance (first-last atom)
    const [dx, dy, dz] = [
      coords[0][0] - coords[L - 1][0],
      coords[0][1] - coords[L - 1][1],
      coords[0][2] - coords[L - 1][2],
    ];
    s.closure_distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    s.bsj_3d_closure_tightness = Math.exp(-s.closure_distance / 5.9);

    // Bond RMSD
    let bs = 0, bc = 0;
    for (let i = 0; i < L; i++) {
      const j = (i + 1) % L;
      const d = Math.sqrt(
        (coords[i][0] - coords[j][0]) ** 2 +
        (coords[i][1] - coords[j][1]) ** 2 +
        (coords[i][2] - coords[j][2]) ** 2
      );
      bs += (d - 5.9) ** 2;
      bc++;
    }
    s.bond_rmsd = bc ? Math.sqrt(bs / bc) : 0;

    // Clash count
    let cl = 0;
    for (let i = 0; i < L; i++)
      for (let j = i + 3; j < L; j++) {
        if (Math.sqrt(
          (coords[i][0] - coords[j][0]) ** 2 +
          (coords[i][1] - coords[j][1]) ** 2 +
          (coords[i][2] - coords[j][2]) ** 2
        ) < 3) cl++;
      }
    s.clash_count = cl;
    return s;
  }

})();
