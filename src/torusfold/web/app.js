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

  /* ═══════════════ DEMO STRUCTURE ═══════════════ */

  /* Two entry points, one action: the placeholder's button, which is what is on
     screen when the panel is empty, and the toolbar's, which stays reachable after a
     run has put its own structure there. Both call the same function.
     
     `showDemoStructure` is declared further down, beside the other viewer paths.
     A function declaration hoists, so these listeners can name it here; call it at
     click time and it is defined by then either way. */
  var demoBtn = $('btn-demo-structure');
  if (demoBtn) demoBtn.addEventListener('click', function () { showDemoStructure(); });
  var demoToolbarBtn = $('btn-demo-structure-toolbar');
  if (demoToolbarBtn) demoToolbarBtn.addEventListener('click', function () {
    showDemoStructure();
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

  /* The pipeline's stages, in order, as level NAMES. The server decides this
     list (see /api/schema and state.levels); this is only the fallback used
     before the first payload arrives.
     
     It is a list of levels rather than an index, because the two are not the
     same thing here and used to be confused: levels run 0, 1, 1.5, 2, 2.3, 2.5,
     2.6, 3, 3.5, 4, 5, 5.5, so "level 2.5" is index 5. Code that treated the
     server's level number as an array position highlighted the wrong stage from
     level 2.3 onwards. */
  var PIPELINE_LEVELS = [
    { level: '0',   name: 'Secondary structure' },
    { level: '1',   name: '3D prediction' },
    { level: '1.5', name: 'Global relaxation' },
    { level: '2',   name: 'CG folding' },
    { level: '2.3', name: '5-bead refinement' },
    { level: '2.5', name: 'All-atom placement' },
    { level: '2.6', name: 'PyRosetta refine' },
    { level: '3',   name: 'RL fine-tuning' },
    { level: '3.5', name: 'Metadynamics' },
    { level: '4',   name: 'REST2 refinement' },
    { level: '5',   name: 'Amber refinement' },
    { level: '5.5', name: 'PPR repair' },
  ];
  // Replaced by the server's own ladder on the first payload.
  var activeLevels = PIPELINE_LEVELS;
  // Accepts either a number or the string form, so '2.5' and 2.5 agree.
  function levelKey(v) { return v == null ? '' : String(v); }
  function levelIndex(level) {
    var want = levelKey(level);
    for (var i = 0; i < activeLevels.length; i++) {
      if (levelKey(activeLevels[i].level) === want) return i;
    }
    return -1;
  }
  // The name of the last stage, for the "we are done" calls. An index into the
  // list will not do: updateProgress takes a level name, and the last level is
  // '5.5' at position 11, so an index of 11 would name a stage that does not exist.
  function lastLevel() {
    return activeLevels.length ? levelKey(activeLevels[activeLevels.length - 1].level)
                               : '5.5';
  }

  /* A stage's NAME from its POSITION.

     The server publishes both, under names that do not say which is which:
     `current_level` is the index (0-11) and `level_name` is the label ('2.5').
     Both are plausible-looking values — small numbers either way — so using one
     where the other belongs is silent. It was done in two places:

       - the heartbeat fed `current_level` to updateProgress, which looks a name up;
         index 4 became the name '4', and '4' is a real level (REST2) at position 9,
         so the strip highlighted the wrong tile and drifted further off each stage
         while indices 1-3 matched nothing at all;
       - applyRunState fed it to levelIndex for the ladder, which missed every time
         and left the ladder with no stage marked.

     A function rather than a comment because the trap is in the naming, and the
     next person to read `current_level` will make the same inference. */
  function levelNameAt(index) {
    if (index == null) return null;
    if (typeof index === 'number' && isFinite(index)) {
      return index >= 0 && index < activeLevels.length
        ? levelKey(activeLevels[index].level) : null;
    }
    return String(index);
  }

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
  // The running stage's own step count, distinct from the modelled percentage.
  var runStageProgress = $('run-stage-progress');
  var runLadder = $('run-ladder');
  var runJob = $('run-job');
  var runReconnect = $('run-reconnect');
  var runLogPathWrap = $('run-logpath');
  var runLogPathValue = $('run-logpath-value');
  var ladderBuilt = false;

  /* The stage ladder is published by the server, so the bar and the list cannot
     disagree about how many stages there are or what they are called.
     
     currentIdx is a POSITION in that list, not a level name; applyRunState
     converts. */
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

  /* Thousands separators, so "5000 / 200000" reads as "5,000 / 200,000" and a
     long step count is not miscounted at a glance. */
  function fmtCount(n) {
    if (n == null) return '—';
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
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
      if (runLevel) { runLevel.textContent = '—'; runLevel.title = ''; }
      if (runLabel) runLabel.textContent = 'Waiting for the first stage…';
      if (runElapsed) runElapsed.textContent = '0:00';
      if (runRemaining) runRemaining.textContent = 'measuring…';
      if (runConfidence) runConfidence.textContent = 'estimating from 0 stage boundaries';
      if (runStageProgress) { runStageProgress.textContent = 'this stage —'; runStageProgress.title = ''; }
      if (progressBarFill) {
        progressBarFill.style.width = '0%';
        progressBarFill.classList.remove('indeterminate');
      }
      if (runJob) runJob.textContent = '';
      if (runLadder) buildLadder(null, -1);
      renderPlan(null);
      return;
    }

    // Adopt the server's stage list before anything is looked up in it: levelIndex,
    // levelNameAt and labelHeaderStrip all read activeLevels, so the names and the
    // order have to come from the server rather than from the copy in this file.
    if (state.levels && state.levels.length) {
      activeLevels = state.levels.map(function (lv) {
        return { level: lv.level, name: lv.label };
      });
    }

    // buildLadder compares POSITIONS, and current_level is a position, so it goes
    // straight through. It used to be wrapped in levelIndex() as though it were a
    // name, which missed every time — the ladder never marked a stage.
    buildLadder(state.levels, state.current_level);
    labelHeaderStrip(state.levels);
    renderPlan(state.plan);

    if (runLevel) {
      // "Stage 5 of 12" rather than "Level 3.5". The count is a fact and it is what
      // the ladder below draws; the level name is kept in the tooltip for anyone
      // cross-referencing the pipeline documentation.
      var k = state.stage_index, n = state.stage_total;
      runLevel.textContent = (k && n) ? ('Stage ' + k + ' / ' + n)
                                      : (state.level_name != null ? 'Level ' + state.level_name : '—');
      runLevel.title = state.level_name != null ? ('pipeline level ' + state.level_name) : '';
    }
    if (runLabel) {
      runLabel.textContent = state.stage_label || state.message || '—';
      runLabel.classList.toggle('is-error', state.status === 'error');
      runLabel.classList.toggle('is-done', state.status === 'done');
    }
    if (runElapsed) runElapsed.textContent = fmtClock(state.elapsed);

    /* The bar shows THIS STAGE, nothing else.
     *
     * It used to show a whole-run percentage: the stage's own ratio multiplied by
     * its estimated share of the run, from a hand-written weight table. That made
     * the bar and the step counts the pipeline prints look like they contradicted
     * each other — the stage could be 99% through its steps while the bar read 49%,
     * because the bar had the later stages still ahead of it. Both numbers were
     * correct and neither could be made to match.
     *
     * So the bar now measures the only thing that is actually measurable while a
     * stage runs: how far through that stage we are. Stages that report no progress
     * of their own leave it empty, which is honest — a modelled bar would be a
     * guess wearing the costume of a measurement.
     */
    var sp = state.stage_progress;
    var stageRatio = (sp && sp.total) ? sp.ratio : null;
    if (progressBarFill) {
      if (stageRatio == null) {
        progressBarFill.style.width = (state.status === 'running' ? 1 : 0) + '%';
        progressBarFill.classList.add('indeterminate');
      } else {
        progressBarFill.classList.remove('indeterminate');
        progressBarFill.style.width = Math.max(1, stageRatio * 100) + '%';
      }
    }
    if (runStageProgress) {
      if (stageRatio == null) {
        runStageProgress.textContent = state.status === 'running'
          ? 'this stage — no step count reported'
          : 'this stage —';
        runStageProgress.title = 'This stage does not report progress of its own, ' +
          'so the bar stays empty rather than showing an estimate.';
      } else {
        runStageProgress.textContent = 'this stage: ' + fmtCount(sp.step) + ' / ' +
          fmtCount(sp.total) + '  (' + (sp.ratio * 100).toFixed(1) + '%)';
        runStageProgress.title = 'Steps this stage has reported. The bar shows the ' +
          'same number, relative to this stage only — not to the whole run.';
      }
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
      // "projected" means nothing has been measured yet and this came from the
      // weight table. Marked in the text, not only in the tooltip: it is the
      // difference between a measurement and a guess.
      if (eta.state === 'projected' && eta.remaining_low != null) {
        runRemaining.textContent = '~' + runRemaining.textContent;
      }
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

    showStructure(state.structure, state.job_running === true);
    renderLiveMetrics(state.metrics);
  }

  /* Fill the readout panels from the measurements published with the run state.

   * Every panel on the right is built from a `result`, which the server produces
   * only when a run finishes. So for the whole of a run — hours — those tabs held
   * nothing but their static labels and a row of "--", which is exactly what they
   * looked like: broken.
   *
   * The server now measures whichever structure is on screen and publishes it as
   * `metrics`. This folds those numbers into the panel renderers, tagged so the
   * source of each figure is visible and nobody reads a coarse-grained trace's
   * numbers as a finished structure's.
   */
  var liveMetricsDigest = null;
  var lastMetrics = null;
  var metricsRetryTimer = null;

  /* Re-ask for the measurements a few times while they are still being computed.

     Bounded: the measurement takes seconds on a large structure, so a handful of
     attempts covers it, and a page left open for hours does not keep polling. If it
     has not arrived by then the source note says so rather than the page quietly
     hammering the server. */
  var metricsRetryLeft = 0;
  var metricsRetryDigest = null;
  // Budget for one structure: the analysis takes seconds, so a handful of attempts
  // covers it. Reset whenever a different structure appears, because that is a new
  // measurement with its own wait.
  var METRICS_RETRY_BUDGET = 10;
  function resetMetricsRetry() {
    metricsRetryLeft = METRICS_RETRY_BUDGET;
    if (metricsRetryTimer) { clearTimeout(metricsRetryTimer); metricsRetryTimer = null; }
  }
  function scheduleMetricsRetry(digest) {
    // A different structure means a new measurement: give it a fresh budget.
    if (digest && digest !== metricsRetryDigest) {
      metricsRetryDigest = digest;
      resetMetricsRetry();
    }
    if (metricsRetryTimer) return;
    if (metricsRetryLeft <= 0) {
      if (metricsRetryLeft === 0) {
        metricsRetryLeft = -1;      // stop; do not re-arm
        var note = $('live-metrics-source');
        if (note) {
          note.hidden = false;
          note.textContent = 'the current structure is still being measured';
        }
      }
      return;
    }
    metricsRetryLeft--;
    metricsRetryTimer = setTimeout(function () {
      metricsRetryTimer = null;
      fetch('/api/current').then(function (r) { return r.json(); }).then(function (s) {
        if (s && s.metrics) renderLiveMetrics(s.metrics);
      }).catch(function () { /* the next attempt or nothing */ });
    }, 2500);
  }

  /* Hand the viewer the per-residue series the server measured.

     Separate from renderLiveMetrics because the two need different timing: the
     panel renderers work on any payload, while the viewer's cards need an instance
     that exists. On first load renderLiveMetrics runs before showStructure has
     created it. */
  function applyLiveFingerprint() {
    var v = TF.Viewer && TF.Viewer.instance;
    if (!v || typeof v.setFingerprint !== 'function') return;
    if (!lastMetrics || !lastMetrics.per_residue) return;
    try {
      v.setFingerprint({ per_residue: lastMetrics.per_residue,
                         scalar: lastMetrics.physical || {},
                         signals: lastMetrics.structural_3d || {} });
    } catch (e) { console.warn('per-residue cards:', e); }
  }

  function renderLiveMetrics(metrics) {
    if (!metrics || !metrics.live) return;
    var src = metrics.source || {};

    /* Nothing to show yet: the server measures on a worker thread because the
       analysis costs seconds on a large structure. Do NOT render in that state.

       This was the bug that left every panel on "--" for a whole run. The first
       poll arrived before the measurement existed, the panels were rendered from
       that placeholder payload, and the cache key was recorded — so when the real
       numbers landed seconds later the key looked unchanged and the render was
       skipped. The placeholder had overprinted the data and nothing said so.
       Waiting for real values means the panels simply fill in a few seconds later. */
    if (metrics.pending) {
      // Ask again shortly. There is no SSE stream on an idle page — the heartbeat
      // only flows while a job runs — so without this the panels would stay empty
      // until the reader reloaded, which is exactly what happened: the measurement
      // finishes a few seconds after the page asks for it, and nobody asks again.
      scheduleMetricsRetry(metrics.for_digest);
      return;
    }

    // Keyed on the structure's digest, not on its name and a measurement: those two
    // are identical before and after a measurement completes for the same file, so
    // keying on them cannot detect "fresh numbers have arrived".
    var key = (metrics.for_digest || '') + '|' + (src.name || '') + '|' +
              (metrics.stale ? 'stale' : 'fresh');
    if (key === liveMetricsDigest) return;
    // Stale figures are the previous structure's. Shown once, labelled, and then
    // superseded — not re-rendered on every heartbeat.
    liveMetricsDigest = key;

    if (TF.Panels) {
      try { TF.Panels.renderQualityMetrics(metrics); } catch (e) { console.warn('quality metrics:', e); }
      try { TF.Panels.renderPhysical(metrics); } catch (e) { console.warn('physical:', e); }
      try { TF.Panels.renderShape(metrics); } catch (e) { console.warn('shape:', e); }
      try { TF.Panels.renderPairQuality(metrics); } catch (e) { console.warn('pair quality:', e); }
      try { TF.Panels.renderPairDist(metrics); } catch (e) { console.warn('pair dist:', e); }
    }

    /* The viewer draws the per-residue cards from its own fingerprint object, and
       loading a checkpoint passes none — so "Structure statistics" said "No
       per-residue data" while the server was publishing a series per residue.

       Stored rather than applied here: on first load this runs before the viewer
       exists, because showStructure creates it asynchronously. Applying it at this
       point silently did nothing and the cards stayed empty. It is applied again
       from showStructure once the viewer is up. */
    lastMetrics = metrics;
    applyLiveFingerprint();

    var note = $('live-metrics-source');
    if (note) {
      note.hidden = false;
      if (metrics.stale) {
        note.textContent = 'measured from the previous structure — the current one ' +
          'is still being measured';
      } else if (src.delivered) {
        note.textContent = 'measured from the delivered model (' + (src.atoms || 0) +
          ' atoms) — not from a run';
      } else {
        note.textContent = 'measured live from stage ' + (src.level || '?') + ' · ' +
          (src.name || '') + ' · ' + (src.atoms || 0) + ' atoms';
      }
    }
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
  // True while the panel is showing the requested demo rather than a run's output.
  // Kept as a flag rather than folded into shownStructure because the two answer
  // different questions: what is loaded, and whether the reader asked for it.
  var showDemoFlag = false;
  function showStructure(stage, runIsActive) {
    var labelEl = $('viewer-stage');
    var levelEl = $('viewer-stage-level');
    var descEl = $('viewer-stage-desc');
    var sameEl = $('viewer-stage-same');

    if (!stage) return;
    // A run's own structure takes precedence over a demo the reader asked for: once
    // there is real output, the demo is stale by definition. Outside a run the demo
    // stays put, so a heartbeat does not throw away what someone chose to look at.
    //
    // `runIsActive` comes from the server's own job status rather than from
    // TF.State.jobId, which survives the end of a run and would therefore keep
    // "a run is happening" true forever.
    if (showDemoFlag && !runIsActive) return;
    showDemoFlag = false;
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
      .then(function () {
        // The instance exists only now, so this is the first point at which the
        // per-residue cards can be filled from what the server measured.
        applyLiveFingerprint();
      })
      .catch(function (e) {
        if (window.console) console.warn('checkpoint load failed:', e.message);
      });
  }

  /* Exposed so the checkpoint path can be driven directly: the stages that hand the
     same coordinates forward only occur deep inside a multi-hour run, and this is
     the branch that decides whether the panel reloads or says "unchanged". */
  TF.App.showStructure = showStructure;

  /* ── Manual refresh for the two panels that are filled from the server ──

     WHY THE BUTTONS EXIST, given the page already updates itself.

     Neither panel is on a timer, and that is deliberate — but it leaves two gaps:

       1. The 3D panel follows a run on the server's heartbeat, and the heartbeat sends
          only when the stage CHANGES. During a long stage nothing is pushed, so a
          checkpoint written in between is not shown until the next boundary.

       2. The Scoring panel's measurements arrive after the structure does, and the
          retry that waits for them has a BUDGET (scheduleMetricsRetry). When the budget
          is spent it stops for good, and the panel keeps whatever it last had. Before
          this button the only way to re-measure was a page reload.

     Both refresh paths therefore FORCE the read rather than re-entering the code that
     decides whether to read, and both keep what the user is looking at:

       - the structure refresh calls updateStructure(), not mount(), so the camera,
         the representation and the colour scheme survive (the viewer's own note says a
         moving camera makes the same molecule look like a different one);
       - the score refresh re-arms the retry budget first, so the automatic path takes
         over again if the measurement is still running on the server. */

  function _busy(btn, on) {
    if (!btn) return;
    btn.disabled = !!on;
    btn.classList.toggle('is-busy', !!on);
    btn.setAttribute('aria-busy', on ? 'true' : 'false');
  }

  function _flashDone(btn) {
    if (!btn) return;
    btn.classList.remove('is-busy');
    btn.classList.add('is-done');
    setTimeout(function () { btn.classList.remove('is-done'); }, 1000);
  }

  function refreshStructure(btn) {
    _busy(btn, true);
    return fetch('/api/structure')
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (!s || !s.available) {
          showToast('No structure on disk yet.', 'info');
          return;
        }
        // Two identifiers, because two different guards stand between the request and
        // the screen: showStructure returns early when the file key (name|mtime) is
        // unchanged, and again when the CONTENT digest is unchanged. Asking both here
        // is what lets this report "already current" honestly instead of claiming a
        // refresh that the guards then swallowed.
        var sameFile = (s.name + '|' + s.mtime) === shownStructure;
        var sameBody = !!(s.digest && s.digest === shownDigest);
        if (sameFile || sameBody) {
          showToast('Already showing the newest structure (' + (s.desc || s.level) + ').',
                    'info');
          return;
        }
        // Force the read: clear both caches so neither guard can return early, and
        // pass runIsActive = true so the demo guard does not either. Reaching here
        // means the file on disk really is a different one — and if its contents
        // happen to match what is already drawn, updateStructure() re-adds the same
        // model and the camera does not move, so a pointless reload costs a redraw
        // rather than a jump.
        shownStructure = null;
        shownDigest = null;
        showStructure(s, true);
        _flashDone(btn);
      })
      .catch(function (e) {
        showToast('Could not read the structure: ' + e.message, 'error');
      })
      .then(function () { _busy(btn, false); });
  }

  function refreshScores(btn) {
    _busy(btn, true);
    return fetch('/api/current')
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (!s || !s.metrics) {
          // Nothing measured yet. The server measures on a worker thread, so this is
          // the normal first answer rather than a failure.
          showToast('No measurements yet — the server is still working.', 'info');
          return;
        }
        var before = lastMetrics;
        // Re-arm the budget: the server may still be measuring, and the automatic
        // retry is what carries the result the rest of the way.
        resetMetricsRetry();
        scheduleMetricsRetry(s.metrics.source && s.metrics.source.digest);
        renderLiveMetrics(s.metrics);
        _flashDone(btn);
        // The digest lives at metrics.source.digest, NOT on the payload — measured
        // against the running server, which returns source={"level","name","atoms",
        // "delivered","digest"} and no top-level digest. Comparing a field that does
        // not exist made this branch unreachable the first time it was written:
        // undefined !== undefined is false, so the toast never appeared.
        var was = before && before.source && before.source.digest;
        var now = s.metrics.source && s.metrics.source.digest;
        if (was && now && was !== now) {
          showToast('Re-measured: the structure changed, so these are new numbers.',
                    'success');
        }
      })
      .catch(function (e) {
        showToast('Could not read the scores: ' + e.message, 'error');
      })
      .then(function () { _busy(btn, false); });
  }

  TF.App.refreshStructure = refreshStructure;
  TF.App.refreshScores = refreshScores;

  /* The idle path into the 3D panel: ask which structure is the newest one on
     disk and show it. Reached on load when the server has no job, so a finished
     prediction is still on screen after a refresh instead of an empty panel that
     looks like the run was lost. */
  function showLastStructure() {
    fetch('/api/structure').then(function (r) { return r.json(); }).then(function (s) {
      if (s && s.available) {
        showStructure(s, false);
      } else {
        // Nothing to show. Say so, and offer the demo, rather than presenting a
        // borrowed structure as the current one — which is what the panel used to
        // do, and why it displayed a 2,013 nt model over a run that had produced
        // nothing, and a 10-atom leftover trace over a run that had been stopped.
        showViewerEmpty();
      }
    }).catch(function () { /* panel stays as it was */ });
  }

  /* The viewer's empty state, with a reason.
     The placeholder's text is set here rather than left static because the two ways
     of being empty need different words: nothing has run yet, versus a run exists
     and left no displayable result. */
  function showViewerEmpty(reason) {
    showDemoFlag = false;
    var ph = $('viewer-placeholder');
    var titleEl = $('viewer-placeholder-title');
    var hintEl = $('viewer-placeholder-hint');
    var labelEl = $('viewer-stage');
    if (titleEl) titleEl.textContent = 'No structure loaded';
    if (hintEl) {
      hintEl.textContent = reason ||
        'Run a prediction, load a local PDB, or open the demo structure below';
    }
    if (ph) ph.style.display = '';
    if (labelEl) labelEl.hidden = true;
  }

  /* Show the delivered 2,013 nt model, because someone asked for it.
     Deliberately not automatic. It is a showcase rather than an output — it belongs
     to no run on this machine — and a panel that fills itself with it is claiming
     "this is your result" about a structure the run had nothing to do with. */
  function showDemoStructure() {
    var levelEl = $('viewer-stage-level');
    var descEl = $('viewer-stage-desc');
    var labelEl = $('viewer-stage');
    var sameEl = $('viewer-stage-same');

    fetch('/api/structure/demo').then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    }).then(function (pdb) {
      if (!viewer) viewer = new CircRNAViewer('viewer');
      TF.Viewer = TF.Viewer || {};
      TF.Viewer.instance = viewer;
      showDemoFlag = true;
      // Claim the shown-structure key so the next heartbeat does not immediately
      // reload something else over it.
      shownStructure = 'demo';
      shownDigest = null;
      if (labelEl) {
        labelEl.hidden = false;
        if (levelEl) levelEl.textContent = 'Demo';
        if (descEl) descEl.textContent = 'delivered model · 2013 nt · 42,831 atoms ' +
                                          '· not from your run';
      }
      if (sameEl) sameEl.hidden = true;
      // `mount`, not `updateStructure`. updateStructure deliberately keeps the camera
      // so a run's checkpoints do not appear to jump — but the demo is a different
      // molecule at a very different size, and inheriting a camera framed on a
      // 12-residue leftover would leave a 2,013 nt model either invisible or clipped.
      // mount re-frames with zoomTo.
      return viewer.mount(pdb, viewer.fp || {});
    }).then(function () {
      var ph = $('viewer-placeholder');
      if (ph) ph.style.display = 'none';
      // Ask for the demo's own measurements. The heartbeat's `metrics` describe
      // whatever the idle page was showing, so without this the readout would go on
      // reporting a previous run's leftover while a 2,013 nt model fills the panel.
      //
      // The server measures before answering, which takes ~11 s on 42,831 atoms, so
      // the readout is cleared first: leaving the previous structure's numbers under
      // a picture of a different one is the disagreement the source line exists to
      // prevent. `renderPhysical` with no numbers empties the gauges; the note says
      // why. Nothing re-renders on its own while nothing is running — the SSE
      // heartbeat only flows during a run — so a reply that is still stale is handed
      // to scheduleMetricsRetry, which already exists for exactly this wait.
      renderLiveMetrics({ live: true, pending: true });
      if (TF.Panels && TF.Panels.renderPhysical) TF.Panels.renderPhysical({ physical: {} });
      var noteEl = $('live-metrics-source');
      if (noteEl) {
        noteEl.hidden = false;
        noteEl.textContent = 'measuring the demo structure — SASA on 42,831 atoms ' +
                             'takes about ten seconds';
      }
      if (TF.App && TF.App.showToast) {
        TF.App.showToast('Measuring the demo structure. This takes about ten ' +
                         'seconds on 42,831 atoms.', 'info');
      }
      return fetch('/api/metrics/demo').then(function (r) { return r.json(); });
    }).then(function (m) {
      if (m && m.metrics) {
        // The server waits for the measurement, so this is usually the real thing.
        // If it ran out of patience the payload is stale and the retry path finishes
        // the job — the same path that covers a slow measurement on first page load.
        renderLiveMetrics(m.metrics);
        if (m.metrics.stale || m.metrics.pending) {
          scheduleMetricsRetry(m.metrics.for_digest);
        }
      }
      if (TF.App && TF.App.showToast) {
        TF.App.showToast('Loaded the demo structure (2013 nt). The readout has its ' +
                         'own numbers and says they are not from your run.', 'info');
      }
    }).catch(function (e) {
      if (TF.App && TF.App.showToast) {
        TF.App.showToast('Could not load the demo structure: ' + e.message, 'error');
      }
    });
  }

  /* A page refresh loses the job id the browser was holding. Rather than
     offering to start a second run — which the server would reject as a
     conflict anyway — ask the server what it is doing and re-attach. */
  function reconnectToRunningJob() {
    fetch('/api/current').then(function (r) { return r.json(); }).then(function (s) {
      if (!s.has_job) {
        /* No run in progress, but a finished one may still be lying in the output
           directory. Show its last checkpoint instead of an empty 3D panel. */
        // Adopt the server's stage list and naming even when nothing is running.
        // Without this the strip and the progress card keep the labels baked into
        // the markup until a run starts, so the idle page describes the pipeline
        // slightly differently from the run that is about to happen.
        if (s.levels && s.levels.length) {
          activeLevels = s.levels.map(function (lv) {
            return { level: lv.level, name: lv.label };
          });
          labelHeaderStrip(s.levels);
          renderPlan(s.plan);
        }
        // The measurements for whatever structure is on disk. Also on this path:
        // the panels are not only for a run in progress — with no job at all, the
        // page still has a structure to measure and the tabs would otherwise stay
        // blank until a run finished.
        renderLiveMetrics(s.metrics);
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
        updateProgress(levelNameAt(s.current_level), 'running');
        showToast('Re-attached to running job ' + s.job_id, 'info');
        if (TF.Console && TF.Console.connect) TF.Console.connect(s.job_id);
        startPolling(s.job_id);
      } else if (s.status === 'done' && s.job_id) {
        /* Finished while the page was away. Pull the result rather than asking
           the user to run a multi-hour job again. */
        currentJobId = s.job_id;
        TF.State.jobId = s.job_id;
        if (progressCard) progressCard.style.display = '';
        updateProgress(lastLevel(), 'done');
        fetchResult(currentJobId);
        showToast('Recovered the previous run (' + s.job_id + ')', 'success');
      } else if (s.status === 'error') {
        if (progressCard) progressCard.style.display = '';
        applyRunState(s);
        showToast('Last run failed: ' + (s.error || 'unknown'), 'error');
      }
    }).catch(function () { /* nothing to re-attach to */ });
  }

  /* Mirror the run onto the pipeline strip in the header, so the twelve stages
     are readable without opening the Parameters column. Class names differ on
     purpose: .pipe-step is the monospace strip, .progress-step is the card.

     Matching is by data-level, not by position. The tiles are in order, but their
     labels are level names (1.5, 2.3, 3.5 ...), so a positional comparison only
     happens to work while the ordering is right and breaks silently the moment it
     is not — which is exactly what happened when the strip showed six tiles for
     twelve stages. */
  function syncHeaderStrip(activeLevel, statusText) {
    var strip = $('hero-pipeline');
    if (!strip) return;
    var active = levelKey(activeLevel);
    var tiles = strip.querySelectorAll('.pipe-step');
    // "done" needs to know what comes before, so compare positions in the
    // authoritative list rather than the number itself.
    var idx = levelIndex(activeLevel);
    tiles.forEach(function (el) {
      el.classList.remove('active', 'done', 'failed');
      if (statusText === 'idle' || !active) return;
      var mine = levelKey(el.dataset.level);
      var myIdx = levelIndex(el.dataset.level);
      if (myIdx >= 0 && idx >= 0) {
        if (myIdx < idx) el.classList.add('done');
        else if (myIdx === idx) {
          el.classList.add(statusText === 'error' ? 'failed'
                            : statusText === 'done' ? 'done' : 'active');
        }
        return;
      }
      // A tile the server does not list: fall back to a direct label match so it
      // still lights up rather than being silently ignored.
      if (mine === active) el.classList.add('active');
    });
  }

  /* Name the tiles from the server's own list.

     The markup carries labels so the strip reads correctly before any payload
     arrives, but the server's names are authoritative — a hand-written copy in the
     HTML drifted from it (the markup said "All-atom placement" for level 2.5 where
     the server says "CG to all-atom"), and two sources of truth for the same
     label is how this strip came to describe a different pipeline than the one
     running. */
  function labelHeaderStrip(levels) {
    var strip = $('hero-pipeline');
    if (!strip || !levels || !levels.length) return;
    var byLevel = {};
    levels.forEach(function (lv) { byLevel[levelKey(lv.level)] = lv.label; });
    strip.querySelectorAll('.pipe-step').forEach(function (el) {
      var name = byLevel[levelKey(el.dataset.level)];
      var nameEl = el.querySelector('.pipe-name');
      if (name && nameEl && nameEl.textContent !== name) nameEl.textContent = name;
    });
  }

  /* Update every progress display from a level NAME and a status.

     activeLevel is a level NAME ('2.5'). An index is accepted and converted — see
     the note at the top of the body — but a name is what this means.

     Both callers used to pass an index from a six-entry list while the server sent
     a level, and the two meanings collided: the heartbeat's "level 2.5" was read as
     "the 2.5th stage" and clamped, highlighting an unrelated row. That was fixed by
     renaming, and then reintroduced the same way: serve.py publishes `current_level`
     as an index and `level_name` as the label, and the heartbeat handler passed the
     index. Names and positions are both small numbers here, so nothing complains. */
  function updateProgress(activeLevel, statusText) {
    // An index is accepted and converted, because the server publishes the same
    // stage under two names and they are not interchangeable: `current_level` is
    // the position in its twelve-stage list (0-11) and `level_name` is the label
    // ('2.5'). Passing the first where the second belongs is silent, not loud —
    // index 4 was looked up as the name '4', and '4' is a real level (REST2) at
    // position 9, so the header lit the wrong tile and drifted further off with
    // every stage. Indices 1, 2 and 3 matched no tile at all, which is how the
    // strip came to look like it was highlighting stages at random.
    //
    // Normalising here rather than at each call site is deliberate: there were six
    // call sites and every new one was another chance to make the same mistake.
    // -1 keeps its meaning of "no particular stage".
    if (typeof activeLevel === 'number' && isFinite(activeLevel)) {
      activeLevel = activeLevel >= 0 && activeLevel < activeLevels.length
        ? levelKey(activeLevels[activeLevel].level)
        : String(activeLevel);
    }

    var idx = levelIndex(activeLevel);
    if (progressSteps) {
      var steps = progressSteps.querySelectorAll('.progress-step');
      steps.forEach(function (el, i) {
        var ind = el.querySelector('.step-indicator');
        var status = el.querySelector('.step-status');
        el.classList.remove('active-step', 'done-step');
        // The card is rendered from the server's plan, in the same order, so
        // position i there corresponds to position i in activeLevels.
        var mine = levelKey(el.dataset.step);
        var myIdx = levelIndex(mine);
        var before = (myIdx >= 0 && idx >= 0) ? myIdx < idx
                   : (idx >= 0 ? i < idx : false);
        var here = (myIdx >= 0 && idx >= 0) ? myIdx === idx
                 : (idx >= 0 ? i === idx : false);
        if (before) {
          ind.className = 'step-indicator done';
          status.className = 'step-status done-text';
          status.textContent = 'Done';
          el.classList.add('done-step');
        } else if (here) {
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
    if (data.current_level != null) updateProgress(levelNameAt(data.current_level), 'running');
  });

  EventBus.on('sse:done', function (data) {
    if (data) applyRunState(data);
    updateProgress(lastLevel(), 'done');
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
        updateProgress(lastLevel(), 'done');
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

  /* Last resort: guess the stage from the job text or from elapsed time.

     This exists only for the early-return paths. The server reports the real
     current_level on every heartbeat, so this is not the normal source.

     Returns a level NAME, not an index, and the mapping matches the server's
     twelve levels. It previously returned 0-5 against a six-entry list, which put
     "all-atom placement" at 3 while the server calls that 2.5 — so on the fallback
     path every stage from there on was labelled with a different stage's name. */
  function inferLevel(s) {
    var lower = ((s.status || '') + ' ' + (s.message || '')).toLowerCase();
    if (lower.includes('secondary') || lower.includes('vienna') || lower.includes('bpp')) return '0';
    if (lower.includes('rhofold') || lower.includes('vfold') || lower.includes('3d')) return '1';
    if (lower.includes('metadyn')) return '3.5';
    if (lower.includes('rest2') || lower.includes('remd')) return '4';
    if (lower.includes('amber') || lower.includes('all-atom')) return '5';
    if (lower.includes('minim') || lower.includes('ppr')) return '5.5';
    if (lower.includes('place') && lower.includes('atom')) return '2.5';
    if (lower.includes('pyrosetta')) return '2.6';
    if (lower.includes('5-bead') || lower.includes('bead')) return '2.3';
    if (lower.includes('cg') || lower.includes('coarse') || lower.includes('relax')) return '2';
    // Nothing matched: the level depends entirely on the sequence length, so this
    // cannot be more than a rough position. The server path is authoritative.
    var elapsed = pipelineStartTime ? (Date.now() - pipelineStartTime) / 1000 : 0;
    if (elapsed < 60) return '0';
    if (elapsed < 180) return '1';
    if (elapsed < 420) return '2';
    if (elapsed < 600) return '2.5';
    if (elapsed < 900) return '3';
    if (elapsed < 3600) return '3.5';
    if (elapsed < 5400) return '4';
    return '5';
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
        updateProgress(activeLevels[Math.min(activeLevels.length - 1,
            Math.floor((idx / stepOrder.length) * activeLevels.length))].level, 'running');
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
      updateProgress(lastLevel(), 'done');
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

  /* The two manual-refresh buttons. Wired here rather than where the panels are built,
     because the handlers are only ever invoked after init has run — the caches they
     clear (shownStructure, shownDigest) are populated by the first render, not at
     wiring time. */
  var refreshStructBtn = $('btn-refresh-structure');
  if (refreshStructBtn) {
    refreshStructBtn.addEventListener('click', function () {
      refreshStructure(refreshStructBtn);
    });
  }
  var refreshScoresBtn = $('btn-refresh-scores');
  if (refreshScoresBtn) {
    refreshScoresBtn.addEventListener('click', function () {
      refreshScores(refreshScoresBtn);
    });
  }

  /* Shift+R does the same thing without reaching for the mouse, which matters while
     watching a run: the panel is on the far side of a wide window. */
  document.addEventListener('keydown', function (e) {
    if (!e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    if ((e.key || '').toLowerCase() !== 'r') return;
    var t = e.target || {};
    // Not while typing: a sequence box or a parameter field owns the key there.
    if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) return;
    e.preventDefault();
    refreshStructure(refreshStructBtn);
    refreshScores(refreshScoresBtn);
  });

  setTimeout(reconnectToRunningJob, 0);

})();
