/* checkpoint.js — the resume-state panel.
 *
 * WHY THIS EXISTS
 * ---------------
 * A checkpoint decides whether a run redoes six hours of work or starts from Level
 * 0, and until now the only way to learn which was to read the run log. Worse, the
 * two answers look identical from outside: a run that resumes and a run that
 * refuses both print a level banner and carry on, and the refused state is moved
 * aside rather than announced — so "why did it start over" had no answer available
 * in the interface at all.
 *
 * The panel shows what the directory actually holds. Every value comes from
 * /api/checkpoint, which reads the manifest without judging it; nothing here
 * computes a verdict of its own, because a second opinion about resumability that
 * could disagree with the server's would be worse than none.
 */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Ckpt = TF.Checkpoint = {};

  // What the badge says, and how alarming it is. The server decides `state`; this
  // only words it.
  const STATES = {
    usable: {
      cls: 'ok', text: 'will resume',
      line: 'Written by this arrangement. The next run of the same sequence with the ' +
            'same parameters continues from here.',
    },
    legacy: {
      cls: 'warn', text: 'will not resume',
      line: 'This checkpoint records no configuration, so it cannot be shown to ' +
            'belong to the current run. It will be kept aside and the run will ' +
            'start from Level 0.',
    },
    quarantined: {
      cls: 'bad', text: 'set aside',
      line: 'A previous checkpoint was rejected and kept for inspection. The next ' +
            'run starts from Level 0.',
    },
    absent: {
      cls: '', text: 'empty',
      line: 'No resume state in this directory. The next run computes everything.',
    },
  };

  function $(id) { return document.getElementById(id); }

  function bytes(n) {
    if (n == null) return '—';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1048576).toFixed(2) + ' MB';
  }

  function card(label, value, color) {
    return '<div class="scalar-card"><div class="k">' + label +
           '</div><div class="v"' + (color ? ' style="color:' + color + '"' : '') +
           '>' + value + '</div></div>';
  }

  Ckpt.render = function (rep) {
    const stateEl = $('ckpt-badge');
    const lineEl = $('ckpt-line');
    const gridEl = $('ckpt-grid');
    const fieldsEl = $('ckpt-fields');
    const arraysEl = $('ckpt-arrays');
    if (!stateEl) return;

    if (!rep || rep.error) {
      stateEl.className = 'ckpt-badge bad';
      stateEl.textContent = 'unavailable';
      if (lineEl) lineEl.textContent = (rep && rep.error) || 'no response';
      return;
    }

    const s = STATES[rep.state] || STATES.absent;
    stateEl.className = 'ckpt-badge ' + s.cls;
    stateEl.textContent = s.text;
    // The extra clause is only true when there is something to be a mismatch with.
    if (lineEl) {
      lineEl.textContent = s.line + (rep.job_running
        ? ' A run is in progress; this is its state so far.'
        : '');
    }

    if (gridEl) {
      const cfg = rep.config_count
        ? rep.config_count + ' parameter' + (rep.config_count === 1 ? '' : 's')
        : 'not recorded';
      gridEl.innerHTML =
        card('recorded level', rep.level == null ? '—' : 'Level ' + rep.level) +
        card('configuration', cfg,
             rep.config_count ? '' : 'var(--warn)') +
        card('fields', rep.fields ? rep.fields.length : 0) +
        card('arrays on disk', (rep.arrays ? rep.arrays.length : 0) +
             ' · ' + bytes(rep.array_bytes)) +
        card('orphaned arrays', (rep.orphans || []).length,
             (rep.orphans || []).length ? 'var(--err)' : 'var(--ok)') +
        card('signature', rep.config_sig ? rep.config_sig.slice(0, 8) : '—');
    }

    if (fieldsEl) {
      const fields = rep.fields || [];
      const orphans = rep.orphans || [];
      if (!fields.length && !orphans.length) {
        fieldsEl.innerHTML = '<span class="legend">nothing recorded</span>';
      } else {
        fieldsEl.innerHTML = fields.map(function (f) {
          return '<span class="ckpt-chip">' + f + '</span>';
        }).join('') + orphans.map(function (f) {
          return '<span class="ckpt-chip orphan" title="no field references this ' +
                 'file">' + f + '</span>';
        }).join('');
      }
    }

    if (arraysEl) {
      const arrays = rep.arrays || [];
      if (!arrays.length) {
        arraysEl.innerHTML = '<span class="legend">no arrays</span>';
      } else {
        arraysEl.innerHTML = arrays.map(function (a) {
          return '<div class="ckpt-array' + (a.exists ? '' : ' missing') + '">' +
                 '<span>' + a.field + ' <span class="sz">' + a.file + '</span></span>' +
                 '<span class="sz">' + (a.exists ? bytes(a.bytes) : 'MISSING') +
                 '</span></div>';
        }).join('');
      }
    }

    const qPanel = $('ckpt-quarantine-panel');
    const qNote = $('ckpt-quarantine-note');
    const qExtra = $('ckpt-quarantine-extra');
    if (qPanel) {
      if (rep.quarantined) {
        qPanel.hidden = false;
        if (qNote) {
          qNote.textContent = 'A checkpoint was declined and moved to ' +
            '_checkpoint.incompatible.json rather than deleted, because "a previous ' +
            'run left state here that does not apply" is worth being able to inspect.';
        }
        if (qExtra) {
          qExtra.innerHTML =
            card('its level', rep.quarantine_level == null ? '—' : rep.quarantine_level) +
            card('its sequence', rep.quarantine_seq || '—') +
            card('its signature', rep.quarantine_sig || '—');
        }
      } else {
        qPanel.hidden = true;
      }
    }
  };

  Ckpt.refresh = function () {
    return fetch('/api/checkpoint').then(function (r) { return r.json(); })
      .then(function (rep) { Ckpt.render(rep); return rep; })
      .catch(function (e) { Ckpt.render({ error: String(e) }); });
  };

  function act(url, body, label) {
    const note = $('ckpt-action-note');
    if (note) { note.className = 'ckpt-note'; note.textContent = label + '…'; }
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    }).then(function (r) { return r.json().then(function (j) {
      if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
      return j;
    }); }).then(function (j) {
      if (note) {
        note.className = 'ckpt-note';
        note.textContent = j.note || (label + ' done');
      }
      return Ckpt.refresh();
    }).catch(function (e) {
      if (note) { note.className = 'ckpt-note is-error'; note.textContent = label + ' failed: ' + e.message; }
    });
  }

  function wire() {
    const refresh = $('ckpt-refresh');
    if (refresh) refresh.addEventListener('click', function () { Ckpt.refresh(); });

    const clean = $('ckpt-clean');
    if (clean) clean.addEventListener('click', function () {
      act('/api/checkpoint/clean', {}, 'Cleaning orphans');
    });

    const wipe = $('ckpt-wipe');
    if (wipe) wipe.addEventListener('click', function () {
      if (!window.confirm('Delete all resume state in output_web?\n\n' +
                          'PDB files and result JSON are kept. The next run will ' +
                          'compute every stage from Level 0.')) return;
      act('/api/checkpoint/delete', {}, 'Deleting resume state');
    });

    const drop = $('ckpt-drop');
    if (drop) drop.addEventListener('click', function () {
      const v = window.prompt(
        'Roll the checkpoint back so the next run recomputes from a level.\n\n' +
        'Enter a level: 2 (REMD), 2.5 (all-atom), 2.6 (PyRosetta), 3 (RL),\n' +
        '3.5 (metadynamics), 4 (REST2), 5 (Amber), 5.5 (PPR)', '2.5');
      if (v == null) return;
      const lvl = parseFloat(v);
      if (!isFinite(lvl)) {
        const note = $('ckpt-action-note');
        if (note) { note.className = 'ckpt-note is-error'; note.textContent = 'not a level: ' + v; }
        return;
      }
      act('/api/checkpoint/rollback', { level: lvl }, 'Rolling back to before Level ' + lvl);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { wire(); Ckpt.refresh(); });
  } else {
    wire();
    Ckpt.refresh();
  }

  // Refresh when the tab is opened, and after a run ends. Not on a timer: the
  // panel is a snapshot of a directory, and a directory only changes when a run
  // writes to it.
  document.addEventListener('change', function (e) {
    if (e.target && e.target.id === 'tab-checkpoint' && e.target.checked) {
      Ckpt.refresh();
    }
  });
})();
