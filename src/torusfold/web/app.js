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

  var serverWasDown = false;
  function probeHealth(announce) {
    fetch('/api/health').then(function (r) { return r.json(); }).then(function (h) {
      /* Two different questions, and they used to share one answer: is the server
         answering, and is a prediction running. Reporting the job state as the
         backend's state made a healthy idle server read as "backend: idle", which
         looks like a failure. The dot is driven by the class now, so it can
         actually turn green — before, only the error path set a colour and the
         healthy path never added `.live`. */
      if (!serverStatus) return;
      var job = h.job_status || 'idle';
      serverStatus.classList.remove('live', 'down');
      if (h.ok) {
        serverStatus.classList.add('live');
        serverStatus.style.color = '';
        serverStatus.textContent = job === 'idle'
          ? 'backend: ready'
          : 'backend: ' + job;
        // Only announce the transition. A toast every poll would be noise.
        if (announce || serverWasDown) showToast('Server connected', 'success');
        serverWasDown = false;
      } else {
        serverStatus.classList.add('down');
        serverStatus.textContent = 'backend: not ready';
        serverWasDown = true;
      }
    }).catch(function () {
      if (!serverStatus) return;
      serverStatus.classList.remove('live');
      serverStatus.classList.add('down');
      serverStatus.textContent = 'server unreachable';
      serverStatus.style.color = 'var(--err)';
      serverWasDown = true;
    });
  }
  probeHealth(true);

  /* Keep the indicator honest while the page is open: a run started elsewhere
     should show up here, and a server that stops answering should stop claiming
     to be ready. */
  setInterval(function () { probeHealth(false); }, 15000);

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

  /* ═══════════════ THEME ═══════════════ */

  /* Cycle auto -> dark -> light -> auto. The inline script in index.html has
     already applied a saved value before first paint; this only handles the
     button and the persistence. */
  var themeToggle = $('theme-toggle');
  if (themeToggle) {
    var THEME_ORDER = ['auto', 'dark', 'light'];
    themeToggle.addEventListener('click', function () {
      var current = 'auto';
      try { current = localStorage.getItem('tf-theme') || 'auto'; } catch (e) {}
      var next = THEME_ORDER[(THEME_ORDER.indexOf(current) + 1) % THEME_ORDER.length];
      if (next === 'auto') {
        document.documentElement.removeAttribute('data-theme');
        try { localStorage.removeItem('tf-theme'); } catch (e) {}
      } else {
        document.documentElement.setAttribute('data-theme', next);
        try { localStorage.setItem('tf-theme', next); } catch (e) {}
      }
      themeToggle.title = 'Theme: ' + next;
      /* Canvas/SVG charts cache resolved token colours, so they must re-read
         after the palette changes or they keep the old theme's colours. */
      if (TF.Charts && TF.Charts.refreshTokens) TF.Charts.refreshTokens();
    });
    try {
      themeToggle.title = 'Theme: ' + (localStorage.getItem('tf-theme') || 'auto');
    } catch (e) {}
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

  /* ═══════════════ PARAMETERS (built from the server schema) ═══════════════ */

  /* The knob list is not written here. serve.py owns _PARAM_SPEC, derives each
     default from isrnaclong_pipeline's real signature, and serves the result at
     /api/schema; this reads it and builds the form. A knob added server-side
     shows up here with no frontend edit, which is the point: the previous
     hand-written list offered 8 of the pipeline's 34 options and three of its
     values contradicted the demo run. */
  var paramSchema = null;
  var paramInputs = {};

  function loadParamSchema() {
    var host = $('params-form');
    if (!host) return;
    fetch('/api/schema').then(function (r) { return r.json(); }).then(function (schema) {
      paramSchema = schema;
      buildParamForm(host, schema);
    }).catch(function (e) {
      host.innerHTML = '<div class="legend">Could not load /api/schema — ' +
        'the pipeline will run with its own defaults.</div>';
      if (window.console) console.warn('schema load failed:', e);
    });
  }

  function buildParamForm(host, schema) {
    host.innerHTML = '';
    paramInputs = {};
    var open = schema.open_by_default || [];

    (schema.groups || []).forEach(function (group) {
      var box = document.createElement('details');
      box.className = 'param-group';
      if (open.indexOf(group.name) !== -1) box.open = true;

      var sum = document.createElement('summary');
      sum.innerHTML = '<span class="pg-name">' + group.name + '</span>' +
        '<span class="pg-count">' + group.params.length + '</span>';
      box.appendChild(sum);

      var body = document.createElement('div');
      body.className = 'param-body';

      group.params.forEach(function (p) {
        var row = document.createElement('label');
        row.className = 'param-row';
        row.title = p.help || '';
        var id = 'param-' + p.name;

        var name = document.createElement('span');
        name.className = 'param-name';
        name.textContent = p.label + (p.unit ? ' (' + p.unit + ')' : '');
        row.appendChild(name);

        var input;
        if (p.kind === 'bool') {
          input = document.createElement('input');
          input.type = 'checkbox';
          input.checked = !!p.default;
          row.classList.add('param-row-bool');
        } else if (p.kind === 'text') {
          input = document.createElement('input');
          input.type = 'text';
          input.value = p.default == null ? '' : p.default;
          input.placeholder = p.placeholder || '';
        } else {
          input = document.createElement('input');
          input.type = 'number';
          input.value = p.default;
          if (p.min != null) input.min = p.min;
          if (p.max != null) input.max = p.max;
          input.step = p.step != null ? p.step : (p.kind === 'float' ? 0.01 : 1);
        }
        input.id = id;
        input.dataset.param = p.name;
        input.addEventListener('change', function () { markTouched(input, p); });
        input.addEventListener('input', function () { markTouched(input, p); });
        row.appendChild(input);

        if (p.help) {
          var help = document.createElement('span');
          help.className = 'param-help';
          help.textContent = p.help;
          row.appendChild(help);
        }

        paramInputs[p.name] = { el: input, spec: p };
        body.appendChild(row);
      });

      var reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'btn btn-quiet btn-xs param-reset';
      reset.textContent = 'Reset group';
      reset.addEventListener('click', function () {
        group.params.forEach(function (p) {
          var entry = paramInputs[p.name];
          if (!entry) return;
          if (p.kind === 'bool') entry.el.checked = !!p.default;
          else entry.el.value = p.default == null ? '' : p.default;
          markTouched(entry.el, p);
        });
      });
      body.appendChild(reset);

      box.appendChild(body);
      host.appendChild(box);
    });

    var foot = document.createElement('div');
    foot.className = 'param-foot';
    foot.textContent = 'defaults from ' + (schema.path || 'the pipeline signature') +
      ' — ' + (schema.knob_count || materialisedCount()) + ' of ' +
      (schema.pipeline_option_count || '?') + ' pipeline options';
    host.appendChild(foot);
  }

  /* Highlight a value that no longer matches the pipeline default, so a changed
     run is visible at a glance before submitting. */
  function markTouched(input, spec) {
    var isDefault;
    if (spec.kind === 'bool') isDefault = input.checked === !!spec.default;
    else if (spec.kind === 'text') isDefault = input.value === (spec.default == null ? '' : spec.default);
    else isDefault = parseFloat(input.value) === spec.default;
    input.classList.toggle('touched', !isDefault);
  }

  function materialisedCount() {
    return Object.keys(paramInputs).length;
  }

  /* Collect the form into the pipeline kwargs. The server clamps and coerces
     again, so a bad value cannot reach the pipeline from here either. */
  function collectParams() {
    var out = {};
    Object.keys(paramInputs).forEach(function (name) {
      var entry = paramInputs[name];
      var el = entry.el;
      if (entry.spec.kind === 'bool') out[name] = !!el.checked;
      else if (entry.spec.kind === 'text') out[name] = el.value;
      else {
        var v = parseFloat(el.value);
        if (!isNaN(v)) out[name] = v;          // leave it out rather than send NaN
      }
    });
    return out;
  }

  loadParamSchema();

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
    if (progressSteps) {
      var steps = progressSteps.querySelectorAll('.progress-step');
      steps.forEach(function (el) {
        var ind = el.querySelector('.step-indicator');
        var status = el.querySelector('.step-status');
        ind.className = 'step-indicator pending';
        status.className = 'step-status pending-text';
        status.textContent = 'Pending';
        el.classList.remove('active-step', 'done-step');
      });
    }
    if (progressBarFill) progressBarFill.style.width = '0%';
    if (progressCard) progressCard.style.display = '';
    applyRunState(null);
    syncHeaderStrip(-1, 'idle');
  }

  /* ═══════════════ RUN STATE (server-authoritative) ═══════════════ */

  /* Everything below renders what the server believes, rather than a second
     guess computed in the browser. The previous bar was fed by three hard-coded
     assignments in serve.py (5, 10, 20) and therefore sat at 20% for the whole
     of Levels 1-5 — most of a multi-hour run. serve.py now derives progress from
     the `[Level X.Y]` banners the pipeline prints, and this just draws it. */
  var runLevel = $('run-level');
  var runLabel = $('run-label');
  var runElapsed = $('run-elapsed');
  var runRemaining = $('run-remaining');
  var runConfidence = $('run-confidence');
  var runPct = $('run-pct');
  var runLadder = $('run-ladder');
  var runJob = $('run-job');
  var runReconnect = $('run-reconnect');
  var runLogPathWrap = $('run-logpath');
  var runLogPathValue = $('run-logpath-value');
  var ladderBuilt = false;

  /* The stage ladder is published by the server, so the bar and the list cannot
     disagree about how many stages there are or what they are called. */
  function buildLadder(levels, currentIdx) {
    if (!runLadder || !levels || !levels.length) return;
    if (!ladderBuilt) {
      runLadder.innerHTML = '';
      // Widths are computed from the weights here rather than left to flex-grow.
      // flex-grow only distributes space left over after the base sizes, and
      // each segment's base is its own (narrow) label, so grow ratios produced
      // near-equal segments. Percentages of the total make the widths exactly
      // proportional to the modelled cost.
      var totalCost = 0;
      levels.forEach(function (lv) { totalCost += Math.max(lv.cost, 0.004); });
      levels.forEach(function (lv) {
        var el = document.createElement('div');
        el.className = 'ladder-step';
        el.dataset.level = lv.level;
        el.title = 'Level ' + lv.level + ' — ' + lv.label +
          ' (modelled at ~' + Math.round(lv.cost * 100) + '% of the run)';
        var share = Math.max(lv.cost, 0.004) / (totalCost || 1);
        el.style.flexBasis = (share * 100).toFixed(2) + '%';
        var bar = document.createElement('span');
        bar.className = 'ladder-cost';
        var name = document.createElement('span');
        name.className = 'ladder-name';
        name.textContent = lv.level;
        el.appendChild(bar);
        el.appendChild(name);
        runLadder.appendChild(el);
      });
      ladderBuilt = true;
    }
    var nodes = runLadder.querySelectorAll('.ladder-step');
    nodes.forEach(function (el, i) {
      el.classList.remove('active', 'done', 'failed');
      if (currentIdx < 0) return;
      if (i < currentIdx) el.classList.add('done');
      else if (i === currentIdx) el.classList.add('active');
    });
  }

  function fmtClock(seconds) {
    if (seconds == null) return '—';
    var s = Math.max(0, Math.round(seconds));
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    if (h > 0) return h + 'h ' + m + 'm';
    if (m > 0) return m + 'm ' + (s % 60) + 's';
    return s + 's';
  }

  /* The step-by-step timeline is rendered from the server's plan, so it lists
     every stage the run will actually execute — including the sub-levels such
     as 2.5 and 5.5 that a six-entry hand-written list could not show — and it
     carries each finished stage's measured duration. */
  function renderPlan(plan) {
    if (!progressSteps || !plan || !plan.length) return;
    if (progressSteps.childElementCount !== plan.length) {
      progressSteps.innerHTML = '';
      plan.forEach(function (item) {
        var el = document.createElement('div');
        el.className = 'progress-step';
        el.dataset.step = item.level;
        el.innerHTML =
          '<div class="step-indicator pending"><span class="step-num"></span></div>' +
          '<div class="step-info">' +
            '<span class="step-name"></span>' +
            '<span class="step-status pending-text">Pending</span>' +
          '</div>';
        progressSteps.appendChild(el);
      });
    }
    plan.forEach(function (item, i) {
      var el = progressSteps.children[i];
      if (!el) return;
      var ind = el.querySelector('.step-indicator');
      var name = el.querySelector('.step-name');
      var status = el.querySelector('.step-status');
      ind.querySelector('.step-num').textContent = item.level;
      name.textContent = item.label;
      el.classList.remove('active-step', 'done-step');
      var cls = item.state === 'done' ? 'done' : item.state === 'running' ? 'running' : 'pending';
      ind.className = 'step-indicator ' + cls;
      status.className = 'step-status ' + cls + '-text';
      if (item.state === 'done') {
        el.classList.add('done-step');
        // The measured duration is the useful number for a finished stage.
        status.textContent = item.seconds != null ? fmtClock(item.seconds) : 'Done';
      } else if (item.state === 'running') {
        el.classList.add('active-step');
        status.textContent = 'Running…';
      } else {
        status.textContent = 'Pending · ~' + item.weight_pct + '%';
      }
    });
  }

  /* One function renders every field, so a refresh and a live SSE heartbeat
     produce an identical view from the same payload. */
  function applyRunState(state) {
    if (!state) {
      if (runLevel) runLevel.textContent = '—';
      if (runLabel) runLabel.textContent = 'Waiting for the first stage…';
      if (runElapsed) runElapsed.textContent = '0:00';
      if (runRemaining) runRemaining.textContent = 'measuring…';
      if (runConfidence) runConfidence.textContent = 'estimating from 0 stage boundaries';
      if (runPct) runPct.textContent = '0%';
      if (runJob) runJob.textContent = '';
      if (runLadder) buildLadder(null, -1);
      renderPlan(null);
      return;
    }

    var idx = state.current_level == null ? -1 : state.current_level;
    buildLadder(state.levels, idx);
    renderPlan(state.plan);

    if (runLevel) {
      runLevel.textContent = state.level_name != null ? 'Level ' + state.level_name : '—';
    }
    if (runLabel) {
      runLabel.textContent = state.stage_label || state.message || '—';
      runLabel.classList.toggle('is-error', state.status === 'error');
      runLabel.classList.toggle('is-done', state.status === 'done');
    }
    if (runElapsed) runElapsed.textContent = fmtClock(state.elapsed);
    if (runPct) runPct.textContent = (state.progress == null ? 0 : state.progress) + '%';
    if (progressBarFill && state.progress != null) {
      progressBarFill.style.width = Math.max(state.progress, state.status === 'running' ? 1 : 0) + '%';
    }

    var eta = state.eta || {};
    if (runRemaining) {
      if (eta.state === 'measuring') runRemaining.textContent = 'measuring…';
      else if (eta.state === 'done') runRemaining.textContent = 'done';
      else if (eta.state === 'stopped') runRemaining.textContent = 'stopped';
      else if (eta.remaining_low == null) runRemaining.textContent = '—';
      // A single number would claim precision the anchor points do not support,
      // so the range is shown as-is.
      else runRemaining.textContent = fmtClock(eta.remaining_low) + '–' + fmtClock(eta.remaining_high);
      runRemaining.title = eta.basis || '';
      runRemaining.dataset.confidence = eta.confidence || 'none';
    }
    if (runConfidence) {
      runConfidence.textContent = eta.basis || '';
    }
    if (runJob && state.job_id) {
      runJob.textContent = 'job ' + state.job_id +
        (state.sequence_length ? ' · ' + state.sequence_length + ' nt' : '');
    }
    // Publish the log file path so a second terminal can follow the run. The
    // server writes it line-buffered, so `Get-Content -Wait` on it is live.
    if (runLogPathWrap) {
      if (state.log_path) {
        runLogPathWrap.hidden = false;
        if (runLogPathValue) runLogPathValue.textContent = state.log_path;
      } else {
        runLogPathWrap.hidden = true;
      }
    }

    showStructure(state.structure);
  }

  /* Show the latest finished checkpoint in the 3D panel.
   *
   * A prediction leaves a chain of progressively better PDBs, so the panel can
   * show the structure being built instead of staying empty for the whole run.
   * Only a CHANGE is acted on: the heartbeat arrives every few seconds and
   * reloading the same structure would churn the 3D viewer for nothing.
   */
  var shownStructure = null;
  var shownDigest = null;
  function showStructure(stage) {
    var labelEl = $('viewer-stage');
    var levelEl = $('viewer-stage-level');
    var descEl = $('viewer-stage-desc');
    var sameEl = $('viewer-stage-same');

    if (!stage) return;
    // Keyed on the file, not its contents: the label must follow the stage even
    // when the coordinates did not change.
    var key = stage.name + '|' + stage.mtime;
    if (key === shownStructure) return;
    shownStructure = key;

    if (labelEl) {
      labelEl.hidden = false;
      if (levelEl) levelEl.textContent = 'Level ' + stage.level;
      if (descEl) descEl.textContent = stage.desc || stage.name;
    }

    // Stages pass the same coordinates forward, so a new stage can be the same
    // molecule under a new name. Say so rather than reloading it and letting the
    // view sit still, which would look like the viewer had frozen. Gated on the
    // viewer already holding something: after a refresh the digest cache is empty
    // and the restored checkpoint still has to be loaded.
    var unchanged = !!(shownDigest && stage.digest && stage.digest === shownDigest);
    if (sameEl) sameEl.hidden = !unchanged;
    if (unchanged) return;
    shownDigest = stage.digest || null;

    fetch('/api/structure/' + stage.name.split('/').map(encodeURIComponent).join('/'))
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.text();
      })
      .then(function (pdb) {
        if (!viewer) viewer = new CircRNAViewer('viewer');
        TF.Viewer = TF.Viewer || {};
        TF.Viewer.instance = viewer;
        // updateStructure, not mount: the scalar/stat cards belong to the finished
        // result and must not be re-rendered on every checkpoint, and the camera
        // must not move — the point is that this is the same molecule improving.
        return viewer.updateStructure(pdb);
      })
      .catch(function (e) {
        if (window.console) console.warn('checkpoint load failed:', e.message);
      });
  }

  /* Exposed so the checkpoint path can be driven directly: the stages that hand the
     same coordinates forward only occur deep inside a multi-hour run, and this is
     the branch that decides whether the panel reloads or says "unchanged". */
  TF.App.showStructure = showStructure;

  /* The idle path into the 3D panel: ask which structure is the newest one on
     disk and show it. Reached on load when the server has no job, so a finished
     prediction is still on screen after a refresh instead of an empty panel that
     looks like the run was lost. */
  function showLastStructure() {
    fetch('/api/structure').then(function (r) { return r.json(); }).then(function (s) {
      if (s && s.available) showStructure(s);
    }).catch(function () { /* panel stays as it was */ });
  }

  /* A page refresh loses the job id the browser was holding. Rather than
     offering to start a second run — which the server would reject as a
     conflict anyway — ask the server what it is doing and re-attach. */
  function reconnectToRunningJob() {
    fetch('/api/current').then(function (r) { return r.json(); }).then(function (s) {
      if (!s.has_job) {
        /* No run in progress, but a finished one may still be lying in the output
           directory. Show its last checkpoint instead of an empty 3D panel. */
        showLastStructure();
        return;
      }
      applyRunState(s);
      if (s.job_id) {
        currentJobId = s.job_id;
        TF.State.jobId = s.job_id;
      }
      if (s.status === 'running') {
        if (progressCard) progressCard.style.display = '';
        if (runReconnect) runReconnect.hidden = true;
        pipelineStartTime = s.started_at ? s.started_at * 1000 : Date.now();
        if (predictBtn) predictBtn.disabled = true;
        updateProgress(s.current_level == null ? 0 : s.current_level, 'running');
        showToast('Re-attached to running job ' + s.job_id, 'info');
        if (TF.Console && TF.Console.connect) TF.Console.connect(s.job_id);
        startPolling(s.job_id);
      } else if (s.status === 'done' && s.job_id) {
        /* Finished while the page was away. Pull the result rather than asking
           the user to run a multi-hour job again. */
        currentJobId = s.job_id;
        TF.State.jobId = s.job_id;
        if (progressCard) progressCard.style.display = '';
        updateProgress(STEP_COUNT - 1, 'done');
        fetchResult(currentJobId);
        showToast('Recovered the previous run (' + s.job_id + ')', 'success');
      } else if (s.status === 'error') {
        if (progressCard) progressCard.style.display = '';
        applyRunState(s);
        showToast('Last run failed: ' + (s.error || 'unknown'), 'error');
      }
    }).catch(function () { /* nothing to re-attach to */ });
  }

  /* Mirror the progress card onto the pipeline strip in the header, so the six
     levels are readable without opening the Parameters column. Class names
     differ on purpose: .pipe-step is the monospace strip, .progress-step is the
     card. Without this the strip would be fixed decoration that lies about
     which level is running. */
  function syncHeaderStrip(activeLevel, statusText) {
    var strip = $('hero-pipeline');
    if (!strip) return;
    strip.querySelectorAll('.pipe-step').forEach(function (el, i) {
      el.classList.remove('active', 'done');
      if (statusText === 'idle') return;
      if (i < activeLevel || (i === activeLevel && statusText === 'done')) {
        el.classList.add('done');
      } else if (i === activeLevel && statusText === 'error') {
        el.classList.add('failed');
      } else if (i === activeLevel) {
        el.classList.add('active');
      }
    });
  }

  /* Layer two of the progress UI: the per-step detail list from the local step
     machine. The server-driven bar and ladder above are authoritative for how
     far the run has got; this list is the fine-grained narration.
     progressSteps is empty until the server's ladder has been rendered, so this
     loops over whatever is present rather than assuming six entries. */
  function updateProgress(activeLevel, statusText) {
    if (progressSteps) {
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
          pct = ((i + 1) / (steps.length || 1)) * 100;
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
    }
    syncHeaderStrip(activeLevel, statusText);
  }

  /* ═══════════════ SSE → PROGRESS INTEGRATION ═══════════════ */

  /* The heartbeat carries the whole run state, so the card is refreshed from
     the server on every frame rather than from a local counter. That is what
     makes a re-attached page and a live session render identically. */
  EventBus.on('sse:heartbeat', function (data) {
    applyRunState(data);
    if (data.current_level != null) updateProgress(data.current_level, 'running');
  });

  EventBus.on('sse:done', function (data) {
    if (data) applyRunState(data);
    updateProgress(STEP_COUNT - 1, 'done');
    fetchResult(currentJobId);
  });

  EventBus.on('sse:error', function (data) {
    updateProgress(-1, 'error');
    if (data) applyRunState(data);
    predictBtn.disabled = false;
    showToast('Error: ' + (data && data.message ? data.message : 'unknown'), 'error');
  });

  /* ═══════════════ PREDICT ═══════════════ */

  predictBtn.addEventListener('click', function () {
    predictBtn.disabled = true;
    var seq = parseSequenceInput(seqArea.value);

    resultCard.style.display = 'none';
    if (seqCard) seqCard.style.display = 'none';
    resetProgress();
    pipelineStartTime = Date.now();

    var params = { sequence: seq, params: collectParams() };

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || e.detail || 'predict failed'); });
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

      // Mount the 3D viewer
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
        console.error('viewer mount failed:', e);
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
          showToast('PDB load failed: ' + err.message, 'error');
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
  if (TF.Setup) TF.Setup.init();

  // Console clear/copy buttons
  var consoleClear = $('console-clear');
  var consoleCopy = $('console-copy');
  if (consoleClear) consoleClear.addEventListener('click', function () { TF.Console && TF.Console.clear(); });
  if (consoleCopy) consoleCopy.addEventListener('click', function () { TF.Console && TF.Console.copyAll(); });

  /* Re-attach on load. A refresh throws away the job id the browser held, and
     the run itself lives on in a server thread, so the page asks what is
     running instead of presenting a fresh Predict button for a job that is
     already going. Deferred one tick so the module inits above have finished
     wiring the console before a re-attached stream starts writing to it. */
  if (runReconnect) {
    runReconnect.addEventListener('click', function () {
      runReconnect.hidden = true;
      reconnectToRunningJob();
    });
  }
  var logPathCopy = $('run-logpath-copy');
  if (logPathCopy) {
    logPathCopy.addEventListener('click', function () {
      var p = runLogPathValue ? runLogPathValue.textContent : '';
      if (!p) return;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(p).then(function () {
          showToast('Log path copied', 'success');
        }).catch(function () {
          showToast('Copy failed — select the path manually', 'error');
        });
      } else {
        showToast('Clipboard unavailable in this browser', 'error');
      }
    });
  }
  setTimeout(reconnectToRunningJob, 0);

})();
