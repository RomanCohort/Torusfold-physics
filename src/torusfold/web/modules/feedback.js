/* feedback.js — Feedback modal with localStorage persistence */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Feedback = TF.Feedback = {};

  const STORAGE_KEY = 'torusfold_feedback';
  let _rating = 0;

  function getEntries() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    } catch (e) {
      return [];
    }
  }

  function saveEntries(entries) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  }

  /** Open the feedback modal */
  Feedback.open = function () {
    const modal = document.getElementById('feedback-modal');
    if (modal) {
      modal.classList.add('open');
      _rating = 0;
      updateStars(0);
      const comments = document.getElementById('feedback-comments');
      if (comments) comments.value = '';
      // Uncheck all tags
      document.querySelectorAll('.feedback-tag input').forEach(function (cb) { cb.checked = false; });
      const note = document.getElementById('feedback-status');
      if (note) { note.hidden = true; note.textContent = ''; }
      renderHistory();
      // Ask where it goes before the reader commits to pressing Submit.
      Feedback.loadStatus();
    }
  };

  /** Close the feedback modal */
  Feedback.close = function () {
    const modal = document.getElementById('feedback-modal');
    if (modal) modal.classList.remove('open');
  };

  /** Submit feedback */
  Feedback.submit = function () {
    const comments = document.getElementById('feedback-comments');
    const tags = Array.from(document.querySelectorAll('.feedback-tag input:checked'))
      .map(function (cb) { return cb.value; });

    const state = TF.State || {};
    const entry = {
      id: 'fb_' + Date.now(),
      timestamp: new Date().toISOString(),
      jobId: state.jobId || null,
      sequenceLength: state.length || 0,
      rating: _rating,
      comments: comments ? comments.value : '',
      tags: tags,
      result_summary: state.result ? {
        closure_distance: state.result.physical && state.result.physical.closure_distance_Ang,
        pair_rate: state.result.structural_3d && state.result.structural_3d.pair_satisfaction_rate,
      } : null,
      // Which build and which screen, so a report can be placed without asking.
      // Collected here rather than server-side: the server sees its own machine,
      // not the reporter's, and "the panel was blank on my laptop" is only
      // actionable with the second one.
      context: {
        userAgent: navigator.userAgent,
        viewport: window.innerWidth + 'x' + window.innerHeight,
        language: navigator.language,
        page: location.pathname,
        jobStatus: (TF.State && TF.State.jobId) ? 'job ' + TF.State.jobId : 'no job',
      },
    };

    const entries = getEntries();
    entries.push(entry);
    saveEntries(entries);

    const btn = document.getElementById('feedback-submit');
    const note = document.getElementById('feedback-status');
    if (btn) btn.disabled = true;
    setNote(note, 'Sending…', '');

    // The message the reader gets must match what actually happened. The form used
    // to say "Feedback saved" and stop there, which read as "sent" — and it was
    // not: the entry went to localStorage and a file beside the server, and the
    // maintainer never saw it. "Saved on this machine" and "emailed to the
    // maintainer" are different promises and are reported separately.
    fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entry),
    }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (res) {
      if (btn) btn.disabled = false;
      if (res && res.emailed) {
        const to = (res.mail && res.mail.to) || 'the maintainer';
        setNote(note, 'Emailed to ' + to + '. Thank you.', 'ok');
        if (TF.App) TF.App.showToast('Feedback emailed. Thank you.', 'success');
        Feedback.close();
      } else {
        // Saved here, not delivered. Say where it went, why it did not go further,
        // and what would fix that — the reader can act on the third one.
        const why = (res && res.mail && (res.mail.hint || res.mail.reason)) ||
                    'the server did not accept it';
        setNote(note, 'Saved on this machine, but not emailed: ' + why, 'warn');
        if (TF.App) TF.App.showToast('Feedback saved locally — email is not set up.',
                                     'info');
      }
    }).catch(function (e) {
      if (btn) btn.disabled = false;
      // Kept in localStorage either way, so nothing written is lost.
      setNote(note, 'Saved in this browser, but the server could not be reached (' +
                    e.message + '). It was not emailed.', 'warn');
    });
  };

  function setNote(el, text, kind) {
    if (!el) return;
    el.textContent = text;
    el.className = 'feedback-status' + (kind ? ' is-' + kind : '');
    el.hidden = false;
  }

  /* Where feedback goes, and whether it can get there. Asked once when the modal
     opens so the button's promise is visible before it is pressed rather than
     discovered afterwards. */
  Feedback.loadStatus = function () {
    const note = document.getElementById('feedback-status');
    fetch('/api/feedback/status').then(function (r) { return r.json(); })
      .then(function (s) {
        if (!s) return;
        if (s.configured) {
          setNote(note, 'Submitting sends this by email to ' + s.to + '.', 'ok');
        } else {
          setNote(note, 'Email is not configured on this server, so submitting will ' +
                        'only save the note here. To enable it, set ' +
                        (s.missing && s.missing.length ? s.missing.join(' and ')
                                                       : 'the mail settings') +
                        ' in .env.local.', 'warn');
        }
      }).catch(function () { /* the note stays as it was */ });
  };

  function updateStars(n) {
    _rating = n;
    document.querySelectorAll('#star-rating .star').forEach(function (star) {
      const v = parseInt(star.dataset.value);
      star.classList.toggle('active', v <= n);
    });
  }

  function renderHistory() {
    const container = document.getElementById('feedback-history');
    if (!container) return;
    const entries = getEntries().slice(-5).reverse();
    if (entries.length === 0) {
      container.innerHTML = '<h3>No previous feedback</h3>';
      return;
    }
    let html = '<h3>Recent Feedback</h3>';
    for (const e of entries) {
      const stars = '★'.repeat(e.rating) + '☆'.repeat(5 - e.rating);
      const date = new Date(e.timestamp).toLocaleDateString();
      html += '<div class="scalar-card"><div class="k">' + date + ' — ' + stars + '</div>';
      if (e.comments) html += '<div class="v" style="font-size:12px;color:var(--t2)">' + escapeHtml(e.comments.slice(0, 100)) + '</div>';
      html += '</div>';
    }
    container.innerHTML = html;
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /** Bind events */
  Feedback.init = function () {
    const btnFeedback = document.getElementById('btn-feedback');
    const btnClose = document.getElementById('feedback-close');
    const btnSubmit = document.getElementById('feedback-submit');
    const modal = document.getElementById('feedback-modal');

    if (btnFeedback) btnFeedback.addEventListener('click', Feedback.open);
    if (btnClose) btnClose.addEventListener('click', Feedback.close);
    if (btnSubmit) btnSubmit.addEventListener('click', Feedback.submit);
    if (modal) modal.addEventListener('click', function (e) {
      if (e.target === modal) Feedback.close();
    });

    // Star rating clicks
    document.querySelectorAll('#star-rating .star').forEach(function (star) {
      star.addEventListener('click', function () {
        updateStars(parseInt(star.dataset.value));
      });
    });
  };

})();
