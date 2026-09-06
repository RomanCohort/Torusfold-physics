/* app.js — TorusFold v4 main orchestrator */
(function () {
  'use strict';

  /* ── Namespace ── */
  var TF = window.TorusFold = window.TorusFold || {};
  TF.State = TF.State || {};
  TF.EventBus = TF.EventBus || {};

  var EventBus = TF.EventBus;
  var _listeners = {};
  EventBus.on = function (event, fn) { (_listeners[event] = _listeners[event] || []).push(fn); };
  EventBus.emit = function (event, data) { (_listeners[event] || []).forEach(function (fn) { fn(data); }); };

  /* ── Helpers ── */
  var $ = function (id) { return document.getElementById(id); };

  function showToast(msg, type, ms) {
    type = type || 'info'; ms = ms || 3500;
    var el = document.createElement('div');
    el.className = 'toast ' + type;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(function () {
      el.style.opacity = '0';
      el.style.transform = 'translateX(40px) scale(0.95)';
      el.style.transition = '0.4s cubic-bezier(0.4,0,0.2,1)';
      setTimeout(function () { el.remove(); }, 400);
    }, ms);
  }
  TF.App = TF.App || {};
  TF.App.showToast = showToast;

  /* ── DOM refs ── */
  var seqArea = $('sequence');
  var seqCounter = $('seq-counter');
  var seqError = $('seq-error');
  var predictBtn = $('predict-btn');
  var progressCard = $('progress-card');
  var progressSteps = $('progress-steps');
  var progressBarFill = $('progress-bar-fill');
  var progressTime = $('progress-time');
  var resultCard = $('result-card');
  var dlPdb = $('dl-pdb');
  var dlJson = $('dl-json');
  var metaSummary = $('meta-summary');
  var seqCard = $('sequence-card');
  var seqBox = $('seq-box');
  var serverStatus = $('server-status');
  var heroSection = $('hero-section');
  var exportGroup = $('export-group');
  var leftToggle = $('left-toggle');
  var asideToggle = $('aside-toggle');
  var leftPanel = $('left-panel');
  var asidePanel = $('fp-panel');

  var viewer = null;
  var currentJobId = null;
  var pollTimer = null;
  var pipelineStartTime = null;

  /* ── Example sequence ── */
  var EXAMPLE_FASTA = '>demo_circRNA_synthetic\nGGUACCGAUCGGCAUUCGGAUCGGCAUUCGGAUGGCAUUCGGCAUUCGGAUGGCAUUCGGCAUUCGGAU\nCCGAUUCGGCAUUCGGAUCCGAUUCGGCAUUCGGAUGGCAUUCGGCAUUCGGAUCCGAUUCGGCAUUCGG';

  /* ═══════════════ SEQUENCE INPUT ═══════════════ */

  function parseSequenceInput(text) {
    var raw = text.replace(/^>.*$/gm, '').replace(/\s+/g, '').toUpperCase();
    return raw.replace(/T/g, 'U');
  }

  function validateSequence(clean) {
    if (!clean) return { valid: false, error: '', badChars: '' };
    if (clean.length < 10) return { valid: false, error: 'Too short (' + clean.length + ' < 10)', badChars: '' };
    var bad = clean.replace(/[ACGUN]/g, '');
    if (bad) {
      var show = bad.split('').slice(0, 10).join(',');
      return { valid: false, error: 'Invalid chars: ' + show, badChars: bad };
    }
    return { valid: true, error: '', badChars: '' };
  }

  function syncSequenceUI() {
    var raw = parseSequenceInput(seqArea.value);
    seqCounter.textContent = raw.length + ' nt';
    var v = validateSequence(raw);
    seqError.textContent = v.error;
    predictBtn.disabled = !v.valid;
    seqCounter.style.color = v.valid ? 'var(--ok)' : raw.length > 0 ? 'var(--err)' : '';
  }

  if (seqArea) seqArea.addEventListener('input', syncSequenceUI);

  var loadExampleBtn = $('load-example-btn');
  if (loadExampleBtn) loadExampleBtn.addEventListener('click', function () {
    seqArea.value = EXAMPLE_FASTA;
    syncSequenceUI();
    showToast('Example loaded (' + parseSequenceInput(EXAMPLE_FASTA).length + ' nt)', 'success');
  });

  var clearSeqBtn = $('clear-seq-btn');
  if (clearSeqBtn) clearSeqBtn.addEventListener('click', function () {
    seqArea.value = '';
    syncSequenceUI();
  });

  /* ═══════════════ HEALTH PROBE ═══════════════ */

  function probeHealth() {
    fetch('/api/health').then(function (r) { return r.json(); }).then(function (h) {
      serverStatus.textContent = 'backend: ' + (h.backend || h.status || 'ok');
      showToast('Server connected', 'success');
    }).catch(function () {
      serverStatus.textContent = 'server unreachable';
      serverStatus.style.color = '#f87171';
    });
  }
  probeHealth();

  /* ═══════════════ DRAG & DROP ═══════════════ */

  if (leftPanel) {
    ['dragenter', 'dragover'].forEach(function (ev) {
      leftPanel.addEventListener(ev, function (e) { e.preventDefault(); e.stopPropagation(); leftPanel.style.outline = '2px dashed var(--accent)'; leftPanel.style.outlineOffset = '-4px'; });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      leftPanel.addEventListener(ev, function (e) { e.preventDefault(); e.stopPropagation(); leftPanel.style.outline = ''; leftPanel.style.outlineOffset = ''; });
    });
    leftPanel.addEventListener('drop', function (e) {
      var f = e.dataTransfer.files[0];
      if (!f) return;
      if (f.name.endsWith('.pdb')) {
        var u = $('pdb-upload');
        if (u) {
          var dt = new DataTransfer(); dt.items.add(f); u.files = dt.files;
          u.dispatchEvent(new Event('change'));
        }
      } else if (/\.(fa|fasta|txt)$/i.test(f.name)) {
        f.text().then(function (t) { seqArea.value = t; syncSequenceUI(); showToast('Sequence loaded', 'success'); });
      } else { showToast('Unsupported file type', 'error'); }
    });
  }

  /* ═══════════════ COLLAPSIBLE PANELS ═══════════════ */

  if (leftToggle) {
    leftToggle.addEventListener('click', function () {
      leftPanel.classList.toggle('collapsed');
      var c = leftPanel.classList.contains('collapsed');
      leftToggle.innerHTML = c ? '&#9654;' : '&#9664;';
      leftToggle.classList.toggle('shifted', c);
    });
  }
  if (asideToggle) {
    asideToggle.addEventListener('click', function () {
      asidePanel.classList.toggle('collapsed');
      var c = asidePanel.classList.contains('collapsed');
      asideToggle.innerHTML = c ? '&#9664;' : '&#9654;';
      asideToggle.classList.toggle('shifted', c);
    });
  }

  /* ═══════════════ REPRESENTATION / OPACITY ═══════════════ */

  var reprSelect = $('repr-select');
  var opacitySlider = $('opacity-slider');
  var opacityVal = $('opacity-val');

  if (reprSelect) reprSelect.addEventListener('change', function () {
    if (viewer && viewer.setRepresentation) viewer.setRepresentation(reprSelect.value);
  });
  if (opacitySlider) opacitySlider.addEventListener('input', function () {
    var v = parseFloat(opacitySlider.value);
    if (opacityVal) opacityVal.textContent = v.toFixed(2);
    if (viewer && viewer.setSurfaceOpacity) viewer.setSurfaceOpacity(v);
  });

  /* ═══════════════ PIPELINE PROGRESS ═══════════════ */

  var PIPELINE_STEPS = [
    { id: 0, name: 'Secondary Structure', abbrev: 'SS' },
    { id: 1, name: '3D Structure Prediction', abbrev: '3D' },
    { id: 2, name: 'CG Refinement', abbrev: 'CG' },
    { id: 3, name: 'All-Atom Placement', abbrev: 'AA+' },
    { id: 4, name: 'All-Atom Refinement', abbrev: 'AA' },
    { id: 5, name: 'All-Atom Minimization', abbrev: 'Min' },
  ];
  var STEP_COUNT = PIPELINE_STEPS.length;

  function resetProgress() {
    if (!progressSteps) return;
    var steps = progressSteps.querySelectorAll('.progress-step');
    steps.forEach(function (el) {
      var ind = el.querySelector('.step-indicator');
      var status = el.querySelector('.step-status');
      ind.className = 'step-indicator pending';
      status.className = 'step-status pending-text';
      status.textContent = 'Pending';
      el.classList.remove('active-step', 'done-step');
    });
    if (progressBarFill) progressBarFill.style.width = '0%';
    if (progressTime) progressTime.textContent = '';
    if (progressCard) progressCard.style.display = '';
  }

  function updateProgress(activeLevel, statusText) {
    if (!progressSteps) return;
    var steps = progressSteps.querySelectorAll('.progress-step');
    var pct = 0;
    steps.forEach(function (el, i) {
      var ind = el.querySelector('.step-indicator');
      var status = el.querySelector('.step-status');
      el.classList.remove('active-step', 'done-step');
      if (i < activeLevel) {
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
        ind.className = 'step-indicator pending';
        status.className = 'step-status pending-text';
        status.textContent = 'Pending';
      }
    });
    if (progressBarFill) {
      if (statusText === 'done') pct = 100;
      progressBarFill.style.width = Math.max(pct, 2) + '%';
    }
    if (progressTime && pipelineStartTime) {
      var elapsed = ((Date.now() - pipelineStartTime) / 1000).toFixed(0);
      progressTime.textContent = statusText === 'done' ? 'Total: ' + formatDuration(pipelineStartTime) : 'Elapsed: ' + elapsed + 's';
    }
  }

  function formatDuration(startMs) {
    var secs = Math.round((Date.now() - startMs) / 1000);
    if (secs < 60) return secs + 's';
    return Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
  }

  /* ═══════════════ SSE → PROGRESS INTEGRATION ═══════════════ */

  EventBus.on('sse:heartbeat', function (data) {
    if (data.current_level != null) updateProgress(data.current_level, 'running');
    if (data.progress != null && progressBarFill) progressBarFill.style.width = Math.max(data.progress, 2) + '%';
  });

  EventBus.on('sse:done', function () {
    updateProgress(STEP_COUNT - 1, 'done');
    fetchResult(currentJobId);
  });

  EventBus.on('sse:error', function (data) {
    updateProgress(-1, 'error');
    predictBtn.disabled = false;
    showToast('Error: ' + (data.message || 'unknown'), 'error');
  });

  /* ═══════════════ PREDICT ═══════════════ */

  predictBtn.addEventListener('click', function () {
    predictBtn.disabled = true;
    var seq = parseSequenceInput(seqArea.value);

    resultCard.style.display = 'none';
    if (seqCard) seqCard.style.display = 'none';
    resetProgress();
    pipelineStartTime = Date.now();

    var params = {
      sequence: seq,
      max_seg_len: +($('param-seglen') ? $('param-seglen').value : 200),
      overlap: +($('param-overlap') ? $('param-overlap').value : 20),
      rounds: +($('param-rounds') ? $('param-rounds').value : 1),
      replicas: +($('param-replicas') ? $('param-replicas').value : 4),
      rest2steps: +($('param-rest2steps') ? $('param-rest2steps').value : 50000),
      use_rl: $('param-rl') ? $('param-rl').checked : true,
      use_rhofold: $('param-rhofold') ? $('param-rhofold').checked : true,
    };

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (e) { throw new Error(e.detail || 'predict failed'); });
      return r.json();
    }).then(function (data) {
      currentJobId = data.job_id;
      TF.State.jobId = data.job_id;
      TF.State.length = data.length;
      showToast('Job submitted: ' + data.job_id, 'info');
      updateProgress(0, 'running');

      // Try SSE first, fallback to polling
      if (TF.Console && TF.Console.isAvailable()) {
        TF.Console.connect(data.job_id);
      } else {
        startPolling(data.job_id);
      }
    }).catch(function (e) {
      updateProgress(0, 'error');
      predictBtn.disabled = false;
      showToast('Predict failed: ' + e.message, 'error');
    });
  });

  /* ═══════════════ POLLING FALLBACK ═══════════════ */

  function startPolling(jid) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(function () { pollJob(jid); }, 800);
    pollJob(jid);
  }

  function pollJob(jid) {
    fetch('/api/jobs/' + jid).then(function (r) { return r.json(); }).then(function (s) {
      if (s.status === 'pending') { updateProgress(0, 'running'); }
      else if (s.status === 'running') {
        var level = inferLevel(s);
        updateProgress(level, 'running');
      } else if (s.status === 'done') {
        clearInterval(pollTimer); pollTimer = null;
        updateProgress(STEP_COUNT - 1, 'done');
        showToast('Prediction complete!', 'success');
        fetchResult(jid);
      } else if (s.status === 'error') {
        clearInterval(pollTimer); pollTimer = null;
        updateProgress(-1, 'error');
        predictBtn.disabled = false;
        showToast('Error: ' + (s.error || 'unknown'), 'error');
      }
    }).catch(function () { /* retry silently */ });
  }

  function inferLevel(s) {
    var text = (s.status || '') + ' ' + (s.message || '');
    var lower = text.toLowerCase();
    if (lower.includes('secondary') || lower.includes('vienna') || lower.includes('bpp')) return 0;
    if (lower.includes('3d') || lower.includes('rhofold') || lower.includes('vfold')) return 1;
    if (lower.includes('cg') || lower.includes('coarse') || lower.includes('refinement')) return 2;
    if (lower.includes('atom') && lower.includes('place')) return 3;
    if (lower.includes('all-atom') || lower.includes('openmm') || lower.includes('relax')) return 4;
    if (lower.includes('minim')) return 5;
    var elapsed = pipelineStartTime ? (Date.now() - pipelineStartTime) / 1000 : 0;
    if (elapsed < 30) return 0;
    if (elapsed < 120) return 1;
    if (elapsed < 300) return 2;
    if (elapsed < 600) return 3;
    if (elapsed < 1200) return 4;
    return 5;
  }

  /* ═══════════════ FETCH & RENDER RESULT ═══════════════ */

  function fetchResult(jid) {
    fetch('/api/result/' + jid).then(function (r) { return r.json(); }).then(function (res) {
      TF.State.result = res;
      TF.State.pdb = res.pdb || '';

      // Show export buttons
      if (exportGroup) exportGroup.style.display = '';
      if (resultCard) resultCard.style.display = 'block';
      if (seqCard) seqCard.style.display = 'block';
      if (seqBox) seqBox.textContent = res.ss || '';

      // Footer
      var mf = $('method-foot');
      var cf = $('closure-foot');
      var ef = $('elapsed-foot');
      if (mf) mf.textContent = res.method || '--';
      if (cf) cf.textContent = res.physical ? (res.physical.closure_distance_Ang || 0).toFixed(3) + ' A' : '--';
      if (ef) ef.textContent = res.runtime ? res.runtime.toFixed(0) + ' s' : '--';

      // Download buttons
      if (dlPdb) { dlPdb.disabled = false; dlPdb.onclick = function () { TF.Export && TF.Export.pdb(); }; }
      if (dlJson) { dlJson.disabled = false; dlJson.onclick = function () { TF.Export && TF.Export.json(); }; }
      if (metaSummary) metaSummary.innerHTML = '<div>method: ' + (res.method || '--') + '</div><div>length: ' + (res.length || 0) + ' nt</div>';

      // Render all panels
      if (TF.Panels) TF.Panels.renderAll(res);

      // Mount Mol* viewer
      try {
        if (!viewer) viewer = new CircRNAViewer('viewer');
        TF.Viewer = TF.Viewer || {};
        TF.Viewer.instance = viewer;
        var fp = {
          sequence: res.ss || '',
          length: res.length,
          method: res.method,
          closure_error: res.physical ? res.physical.closure_distance_Ang : 0,
          per_residue: {},
          scalar: {},
          signals: res.physical || {},
          coloring_schemes: [],
        };
        viewer.mount(res.pdb || '', fp);

        // Show minigame toolbar
        if (TF.MiniGame) TF.MiniGame.show();
      } catch (e) {
        console.error('Mol* mount failed:', e);
      }

      predictBtn.disabled = false;

      // Collapse hero
      if (heroSection) {
        heroSection.style.maxHeight = '0';
        heroSection.style.overflow = 'hidden';
        heroSection.style.padding = '0';
        heroSection.style.opacity = '0';
        heroSection.style.transition = 'all 0.5s cubic-bezier(0.4,0,0.2,1)';
      }
    }).catch(function (e) {
      predictBtn.disabled = false;
      showToast('Failed to load result: ' + e.message, 'error');
    });
  }

  /* ═══════════════ LOCAL PDB LOAD ═══════════════ */

  var pdbUpload = $('pdb-upload');
  if (pdbUpload) {
    pdbUpload.addEventListener('change', function (e) {
      var file = e.target.files[0];
      if (!file) return;
      file.text().then(function (pdbText) {
        try {
          if (!viewer) viewer = new CircRNAViewer('viewer');
          TF.Viewer = TF.Viewer || {};
          TF.Viewer.instance = viewer;
          viewer.mount(pdbText, {});
          if (exportGroup) exportGroup.style.display = 'none';
          if (resultCard) resultCard.style.display = 'block';
          if (seqCard) seqCard.style.display = 'block';
          if (seqBox) seqBox.textContent = '(loaded from file)';
          var atomCount = pdbText.split('\n').filter(function (l) { return l.startsWith('ATOM'); }).length;
          if (metaSummary) metaSummary.innerHTML = '<div>Loaded: ' + file.name + '</div><div>atoms: ' + atomCount + '</div>';
          showToast('PDB loaded: ' + atomCount + ' atoms', 'success');
          if (TF.MiniGame) TF.MiniGame.show();

          // Trigger real-time analysis via SSE
          startPdbAnalysis(pdbText, file.name);
        } catch (err) {
          showToast('Mol* load failed: ' + err.message, 'error');
        }
      });
    });
  }

  /* ═══════════════ PDB REAL-TIME ANALYSIS ═══════════════ */

  function startPdbAnalysis(pdbText, fileName) {
    // Show progress card
    if (progressCard) progressCard.style.display = 'block';
    resetProgress();
    pipelineStartTime = Date.now();
    updateProgress(0, 'running');
    if (TF.Console) TF.Console.appendLine({ timestamp: Date.now() / 1000, level: 'step', message: 'Uploading PDB for analysis...' });

    // Step 1: POST PDB to get session_id
    fetch('/api/score-pdb', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: pdbText,
    }).then(function (r) { return r.json(); }).then(function (res) {
      if (!res.ok || !res.session_id) throw new Error(res.error || 'No session_id');
      if (TF.Console) TF.Console.appendLine({ timestamp: Date.now() / 1000, level: 'info', message: 'Session: ' + res.session_id + ', starting analysis...' });

      // Step 2: GET SSE stream with session_id
      var sseUrl = '/api/score-pdb/sse/' + res.session_id;
      var evtSource;
      try {
        evtSource = new EventSource(sseUrl);
      } catch (err) {
        showToast('SSE not supported', 'error');
        updateProgress(-1, 'error');
        return;
      }

      var stepOrder = ['parse', 'clash', 'rog', 'bond', 'sasa', 'e2e', 'shape', 'backbone', 'pairs', 'b_factor'];
      var stepIdx = 0;

    evtSource.addEventListener('step', function (e) {
      var data = JSON.parse(e.data);
      if (TF.Console) TF.Console.appendLine({ timestamp: Date.now() / 1000, level: 'step', message: data.message });
      var idx = stepOrder.indexOf(data.step);
      if (idx >= 0) {
        stepIdx = idx;
        updateProgress(Math.floor((idx / stepOrder.length) * STEP_COUNT), 'running');
      }
      if (progressBarFill) progressBarFill.style.width = Math.floor((stepIdx / stepOrder.length) * 100) + '%';
    });

    evtSource.addEventListener('metric', function (e) {
      var data = JSON.parse(e.data);
      if (TF.Console) TF.Console.appendLine({ timestamp: Date.now() / 1000, level: 'info', message: data.message });
      // Update right panel with metric data
      updateAnalysisPanel(data.key, data.data);
    });

    evtSource.addEventListener('done', function (e) {
      var data = JSON.parse(e.data);
      evtSource.close();
      updateProgress(STEP_COUNT - 1, 'done');
      if (progressBarFill) progressBarFill.style.width = '100%';
      showToast('PDB analysis complete!', 'success');
      // Switch to Structure tab
      var structTab = document.getElementById('tab-structure');
      if (structTab) structTab.checked = true;
      if (metaSummary) metaSummary.innerHTML = '<div>Analysis: ' + (fileName || 'PDB') + '</div><div>atoms: ' + (data.n_atoms || '?') + '</div>';
    });

    evtSource.onerror = function () {
      evtSource.close();
      updateProgress(-1, 'error');
      showToast('Analysis connection lost', 'error');
    };

    }).catch(function (err) {
      updateProgress(-1, 'error');
      showToast('Upload failed: ' + err.message, 'error');
    });
  }

  function updateAnalysisPanel(key, data) {
    // Clash
    if (key === 'clash') {
      var barEl = document.getElementById('qm-clash-bar');
      var valEl = document.getElementById('qm-clash-val');
      if (barEl && valEl) {
        var pct = Math.max(2, Math.min(100, 100 - (data.clash_score || 0) * 10));
        barEl.style.width = pct + '%';
        barEl.style.background = data.clash_count === 0 ? 'var(--ok)' : data.clash_count < 10 ? 'var(--warn)' : 'var(--err)';
        valEl.textContent = data.clash_score.toFixed(1) + ' / ' + data.clash_count + ' clashes';
        valEl.style.color = barEl.style.background;
      }
    }
    // Bond RMSD
    if (key === 'bond') {
      var barEl = document.getElementById('qm-bond-bar');
      var valEl = document.getElementById('qm-bond-val');
      if (barEl && valEl) {
        var pct = Math.max(2, Math.min(100, 100 - data.bond_rmsd * 20));
        barEl.style.width = pct + '%';
        barEl.style.background = data.bond_rmsd < 0.5 ? 'var(--ok)' : data.bond_rmsd < 1.5 ? 'var(--warn)' : 'var(--err)';
        valEl.textContent = data.bond_rmsd.toFixed(3) + ' A';
        valEl.style.color = barEl.style.background;
      }
    }
    // Pair satisfaction
    if (key === 'pairs') {
      var barEl = document.getElementById('qm-pair-bar');
      var valEl = document.getElementById('qm-pair-val');
      if (barEl && valEl) {
        var pct = data.pair_satisfaction_rate * 100;
        barEl.style.width = Math.max(2, pct) + '%';
        barEl.style.background = data.pair_satisfaction_rate > 0.7 ? 'var(--ok)' : data.pair_satisfaction_rate > 0.4 ? 'var(--warn)' : 'var(--err)';
        valEl.textContent = (data.pair_satisfaction_rate * 100).toFixed(1) + '%';
        valEl.style.color = barEl.style.background;
      }
    }
    // RoG
    if (key === 'rog') {
      addScalarCard('stats-cards', 'Radius of Gyration', data.rog.toFixed(2) + ' A');
    }
    // End-to-end
    if (key === 'e2e') {
      addScalarCard('stats-cards', 'End-to-End Distance', data.e2e.toFixed(2) + ' A');
    }
    // Shape
    if (key === 'shape') {
      addScalarCard('stats-cards', 'Asphericity', data.asphericity.toFixed(4));
      addScalarCard('stats-cards', 'Prolateness', data.prolateness.toFixed(4));
    }
    // Backbone
    if (key === 'backbone') {
      addScalarCard('stats-cards', 'Mean Backbone Angle', data.mean_angle.toFixed(1) + ' deg');
    }
    // SASA
    if (key === 'sasa') {
      addScalarCard('stats-cards', 'Mean SASA', data.mean_sasa.toFixed(4));
    }
  }

  function addScalarCard(containerId, label, value) {
    var el = document.getElementById(containerId);
    if (!el) return;
    // Check if already exists
    var cards = el.querySelectorAll('.scalar-card');
    for (var i = 0; i < cards.length; i++) {
      if (cards[i].querySelector('.k') && cards[i].querySelector('.k').textContent === label) {
        cards[i].querySelector('.v').textContent = value;
        return;
      }
    }
    // Create new card
    var card = document.createElement('div');
    card.className = 'scalar-card';
    card.innerHTML = '<div class="k">' + label + '</div><div class="v">' + value + '</div>';
    el.appendChild(card);
  }

  /* ═══════════════ MODULE INIT ═══════════════ */

  // Init all modules after DOM ready
  if (TF.Export) TF.Export.init();
  if (TF.Feedback) TF.Feedback.init();
  if (TF.MiniGame) TF.MiniGame.init();

  // Console clear/copy buttons
  var consoleClear = $('console-clear');
  var consoleCopy = $('console-copy');
  if (consoleClear) consoleClear.addEventListener('click', function () { TF.Console && TF.Console.clear(); });
  if (consoleCopy) consoleCopy.addEventListener('click', function () { TF.Console && TF.Console.copyAll(); });

})();
