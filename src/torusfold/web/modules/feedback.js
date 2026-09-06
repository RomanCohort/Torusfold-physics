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
      renderHistory();
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
    };

    const entries = getEntries();
    entries.push(entry);
    saveEntries(entries);

    // Also POST to server (fire and forget)
    fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entry),
    }).catch(function () { /* ignore */ });

    TF.App && TF.App.showToast('Feedback saved', 'success');
    Feedback.close();
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
