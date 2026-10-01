/* setup.js — the "Set up environment" pane.
 *
 * Drives POST /api/install-deps, which runs tools/install_deps.py as a sidecar.
 * The installer emits one JSON event per line; the server forwards those into
 * the same log the console panel already reads, so progress appears in two
 * places without a second transport.
 *
 * The pane is deliberately explicit about the three outcomes, because conflating
 * them is what makes an installer untrustworthy:
 *   installed  — fetched and now usable
 *   skipped    — already present on this machine
 *   manual     — no public download source; a human has to place it
 * A "manual" entry is not a failure of the installer and is not shown as one.
 */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Setup = TF.Setup = {};
  const $ = (id) => document.getElementById(id);

  let pollTimer = null;

  function openModal() {
    const m = $('setup-modal');
    if (!m) return;
    m.classList.add('open');
    refresh();
  }

  function closeModal() {
    const m = $('setup-modal');
    if (!m) return;
    m.classList.remove('open');
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  }

  function setStatus(text, isError) {
    const el = $('setup-status');
    if (!el) return;
    el.textContent = text || '';
    el.style.color = isError ? 'var(--err)' : '';
  }

  function renderResults(payload) {
    const host = $('setup-results');
    if (!host) return;
    host.innerHTML = '';
    const res = payload && payload.last_result;

    if (!res) {
      host.innerHTML = '<div class="legend">No setup has been run from here yet. ' +
        'A run installs the Python packages, the tool sources with a public ' +
        'download, and the RNAbpFlow checkpoint, then reports what still needs ' +
        'a manual step.</div>';
      return;
    }

    const group = (title, items) => {
      if (!items || !items.length) return;
      const box = document.createElement('div');
      box.className = 'setup-group';
      const h = document.createElement('h3');
      h.textContent = title;
      box.appendChild(h);
      items.forEach((it) => {
        const row = document.createElement('div');
        row.className = 'setup-item ' + (it.kind || 'ok');
        row.innerHTML =
          '<span class="si-mark"></span><span class="si-label"></span>' +
          (it.note ? '<span class="si-note"></span>' : '');
        row.querySelector('.si-mark').textContent = it.mark;
        row.querySelector('.si-label').textContent = it.label;
        if (it.note) row.querySelector('.si-note').textContent = it.note;
        box.appendChild(row);
      });
      host.appendChild(box);
    };

    group('Installed', (res.installed || []).map((n) => ({ mark: 'added', label: n })));
    group('Already present', (res.skipped || []).map((n) => ({ mark: 'skipped', label: n, kind: 'skip' })));

    // The Python environment is the first thing that has to work, so it is shown
    // first among the findings and names the interpreter it settled on.
    if (res.environment) {
      const env = res.environment;
      group('Python environment', [{
        mark: env.complete ? 'ready' : 'partial',
        label: env.path,
        note: (env.gpu ? 'GPU torch' : 'CPU torch') +
          (env.complete ? '; all required packages import'
                        : '; missing: ' + (env.missing || []).join(', ')),
        kind: env.complete ? 'skip' : 'doc',
      }]);
    }

    const failed = res.failed || [];
    group('Failed', failed.map((f) => ({
      mark: 'failed', label: f.name, note: f.why, kind: 'fail',
    })));

    // Manual steps: split by whether the tool is actually on this machine, so a
    // present tool is never presented as something the user must go and fetch.
    const manual = (res.manual || []).filter((m) => m.id !== 'python-packages');
    const missing = manual.filter((m) => !m.detected);
    const present = manual.filter((m) => m.detected);
    group('No public download — place manually', missing.map((m) => ({
      mark: 'manual', label: m.label, note: m.why + '  →  ' + m.env, kind: 'doc',
    })));
    group('Found on this machine', present.map((m) => ({
      mark: 'found', label: m.label, note: m.detected, kind: 'skip',
    })));

    // Package gaps are their own group: they are fixed by installing, not by
    // fetching a file, so mixing them in would suggest the wrong remedy.
    const pkg = (res.manual || []).filter((m) => m.id === 'python-packages');
    group('Packages to install', pkg.map((m) => ({
      mark: 'install', label: m.label, note: m.why, kind: 'doc',
    })));

    if (res.paths && Object.keys(res.paths).length) {
      group('Paths discovered', Object.keys(res.paths).map((k) => ({
        mark: 'set', label: k, note: res.paths[k], kind: 'skip',
      })));
    }

    const tail = document.createElement('div');
    tail.className = 'legend';
    tail.style.marginTop = '10px';
    tail.textContent = 'Run  python tools/configure_deps.py write  (or restart with ' +
      'start_torusfold.bat) to write these paths into activate_deps.bat.';
    host.appendChild(tail);
  }

  function setRunningUI(running) {
    const run = $('setup-run');
    const cancel = $('setup-cancel');
    const prog = $('setup-progress');
    if (run) run.disabled = !!running;
    if (cancel) cancel.hidden = !running;
    if (prog) prog.hidden = !running;
  }

  function cancel() {
    setStatus('cancelling…');
    fetch('/api/install-deps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cancel: true }),
    }).then((r) => r.json()).then((j) => {
      setStatus(j.ok ? 'cancelled' : 'could not cancel', !j.ok);
      setRunningUI(false);
      refresh();
    }).catch(() => setStatus('could not cancel', true));
  }

  function refresh() {
    fetch('/api/deps').then((r) => r.json()).then((p) => {
      populateSources(p);
      renderResults(p);
      setRunningUI(p.install_running);
      if (p.install_running) {
        setStatus('setup running…');
        if (!pollTimer) pollTick();
      } else if (p.last_result) {
        setStatus('last run finished');
      }
    }).catch(() => setStatus('could not read setup state', true));
  }

  /* The model-host list comes from the installer, so the two cannot disagree
     about which sources exist. 'auto' stays first and is the default: it probes
     each host, which is what makes one button work regardless of network. */
  function populateSources(payload) {
    const sel = $('setup-source');
    if (!sel || !payload || !payload.model_sources) return;
    const wanted = payload.default_source || 'auto';
    const sig = wanted + '|' + payload.model_sources.map((s) => s.id).join(',');
    if (sel.dataset.sig === sig) return;
    sel.dataset.sig = sig;
    sel.innerHTML = '';
    const auto = document.createElement('option');
    auto.value = 'auto';
    auto.textContent = 'auto — probe each host';
    sel.appendChild(auto);
    payload.model_sources.forEach((s) => {
      const o = document.createElement('option');
      o.value = s.id;
      o.textContent = s.label;
      o.title = s.note || '';
      sel.appendChild(o);
    });
    sel.value = wanted;
  }

  function pollTick() {
    pollTimer = setTimeout(() => {
      pollTimer = null;
      fetch('/api/deps').then((r) => r.json()).then((p) => {
        if (p.install_running) {
          const bar = $('setup-bar');
          // The installer does not know the total number of steps up front, so
          // the bar is indeterminate: a made-up percentage would be a lie.
          if (bar) bar.style.width = '40%';
          const prog = $('setup-progress');
          if (prog) prog.hidden = false;
          setStatus('setup running…');
          pollTick();
        } else {
          setRunningUI(false);
          renderResults(p);
          setStatus(p.last_result ? 'setup finished' : 'setup stopped');
          if (TF.Console && TF.Console.connect && TF.State && TF.State.jobId) {
            TF.Console.connect(TF.State.jobId);
          }
        }
      }).catch(() => { pollTimer = null; });
    }, 1500);
  }

  function run() {
    const body = {
      skip_downloads: !!($('setup-skip-downloads') || {}).checked,
      skip_git: !!($('setup-skip-sources') || {}).checked,
      install_missing: !!($('setup-install-missing') || {}).checked,
      source: ($('setup-source') || {}).value || 'auto',
    };
    setStatus('starting…');
    fetch('/api/install-deps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => r.json().then((j) => ({ ok: r.ok, j }))).then(({ ok, j }) => {
      if (!ok) {
        setStatus(j.error || 'could not start', true);
        return;
      }
      const prog = $('setup-progress');
      if (prog) prog.hidden = false;
      setRunningUI(true);
      setStatus('setup running…');
      // Show the installer's own output in the console panel: it streams through
      // the same SSE log the pipeline uses, so there is one place to look.
      const consoleTab = document.querySelector('label[for=tab-console]');
      if (consoleTab) consoleTab.click();
      pollTick();
    }).catch((e) => setStatus('could not start: ' + e.message, true));
  }

  Setup.init = function () {
    const btn = $('btn-setup');
    if (btn) btn.addEventListener('click', openModal);
    const close = $('setup-close');
    if (close) close.addEventListener('click', closeModal);
    const modal = $('setup-modal');
    if (modal) {
      modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
    }
    const runBtn = $('setup-run');
    if (runBtn) runBtn.addEventListener('click', run);
    const cancelBtn = $('setup-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', cancel);
    const refreshBtn = $('setup-refresh');
    if (refreshBtn) refreshBtn.addEventListener('click', refresh);
  };
})();
