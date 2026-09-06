/* minigame.js — Interactive RNA rotation, highlight, measure, auto-rotate */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const MiniGame = TF.MiniGame = {};

  let _autoRotate = false;
  let _autoRotateRAF = null;
  let _highlightMode = false;
  let _measureMode = false;
  let _measurePoints = [];

  /** Show the minigame toolbar */
  MiniGame.show = function () {
    const toolbar = document.getElementById('minigame-toolbar');
    if (toolbar) toolbar.classList.add('visible');
  };

  /** Hide the minigame toolbar */
  MiniGame.hide = function () {
    const toolbar = document.getElementById('minigame-toolbar');
    if (toolbar) toolbar.classList.remove('visible');
  };

  /** Toggle auto-rotation */
  MiniGame.toggleAutoRotate = function () {
    _autoRotate = !_autoRotate;
    const btn = document.getElementById('mg-autorotate');
    if (btn) btn.classList.toggle('active', _autoRotate);

    if (_autoRotate) {
      startAutoRotate();
    } else {
      stopAutoRotate();
    }
  };

  function startAutoRotate() {
    const viewer = TF.Viewer && TF.Viewer.instance;
    if (!viewer || !viewer.plugin) return;

    const plugin = viewer.plugin;
    let angle = 0;

    function rotate() {
      if (!_autoRotate) return;
      angle += 0.003;
      try {
        const cam = plugin.canvas3d && plugin.canvas3d.camera;
        if (cam && cam.state) {
          // Slowly rotate around Y axis
          const state = cam.state;
          const cos = Math.cos(0.005);
          const sin = Math.sin(0.005);
          // Apply rotation to eye vector
          const eye = state.eye;
          const newEye = [
            eye[0] * cos + eye[2] * sin,
            eye[1],
            -eye[0] * sin + eye[2] * cos,
          ];
          cam.setState({ eye: newEye });
        }
      } catch (e) { /* ignore */ }
      _autoRotateRAF = requestAnimationFrame(rotate);
    }
    _autoRotateRAF = requestAnimationFrame(rotate);
  }

  function stopAutoRotate() {
    if (_autoRotateRAF) {
      cancelAnimationFrame(_autoRotateRAF);
      _autoRotateRAF = null;
    }
  }

  /** Toggle residue highlight mode */
  MiniGame.toggleHighlight = function () {
    _highlightMode = !_highlightMode;
    _measureMode = false;
    const btnH = document.getElementById('mg-highlight');
    const btnM = document.getElementById('mg-measure');
    if (btnH) btnH.classList.toggle('active', _highlightMode);
    if (btnM) btnM.classList.remove('active');
    _measurePoints = [];
    hideTooltip();
  };

  /** Toggle measure mode */
  MiniGame.toggleMeasure = function () {
    _measureMode = !_measureMode;
    _highlightMode = false;
    const btnH = document.getElementById('mg-highlight');
    const btnM = document.getElementById('mg-measure');
    if (btnH) btnH.classList.remove('active');
    if (btnM) btnM.classList.toggle('active', _measureMode);
    _measurePoints = [];
    hideTooltip();
  };

  /** Focus on a specific region */
  MiniGame.focusRegion = function (region) {
    const state = TF.State;
    if (!state || !state.result) return;

    const bounds = {
      IRES: state.result.ires_bounds,
      CDS: state.result.cds_bounds,
      '5UTR': { start: 0, end: state.result.ires_bounds ? state.result.ires_bounds.start : 0 },
      BSJ: { start: 0, end: 1 },
    };

    const b = bounds[region];
    if (!b) return;

    TF.App && TF.App.showToast('Focusing on ' + region + ' (pos ' + b.start + '-' + b.end + ')', 'info');
  };

  function showTooltip(x, y, html) {
    const tip = document.getElementById('residue-tooltip');
    if (!tip) return;
    tip.innerHTML = html;
    tip.style.display = 'block';
    // Position near cursor but keep in viewport
    const rect = document.getElementById('viewer-wrap').getBoundingClientRect();
    tip.style.left = Math.min(x - rect.left + 12, rect.width - 230) + 'px';
    tip.style.top = Math.min(y - rect.top - 10, rect.height - 100) + 'px';
  }

  function hideTooltip() {
    const tip = document.getElementById('residue-tooltip');
    if (tip) tip.style.display = 'none';
  }

  /** Bind toolbar events */
  MiniGame.init = function () {
    const btnSpin = document.getElementById('mg-spin');
    const btnHighlight = document.getElementById('mg-highlight');
    const btnMeasure = document.getElementById('mg-measure');
    const btnRegion = document.getElementById('mg-region');
    const btnAutoRotate = document.getElementById('mg-autorotate');

    if (btnSpin) btnSpin.addEventListener('click', function () {
      TF.App && TF.App.showToast('Drag to spin the structure', 'info');
    });
    if (btnHighlight) btnHighlight.addEventListener('click', MiniGame.toggleHighlight);
    if (btnMeasure) btnMeasure.addEventListener('click', MiniGame.toggleMeasure);
    if (btnAutoRotate) btnAutoRotate.addEventListener('click', MiniGame.toggleAutoRotate);

    if (btnRegion) {
      btnRegion.addEventListener('click', function () {
        // Cycle through regions
        const regions = ['IRES', 'CDS', '5UTR'];
        const current = btnRegion.dataset.region || '';
        const idx = regions.indexOf(current);
        const next = regions[(idx + 1) % regions.length];
        btnRegion.dataset.region = next;
        MiniGame.focusRegion(next);
      });
    }

    // Canvas click handler for highlight/measure
    const viewerWrap = document.getElementById('viewer-wrap');
    if (viewerWrap) {
      viewerWrap.addEventListener('click', function (e) {
        if (!_highlightMode && !_measureMode) return;
        // Get pick position from Mol* plugin
        const viewer = TF.Viewer && TF.Viewer.instance;
        if (!viewer || !viewer.plugin) return;
        // Simple fallback: show cursor position
        const rect = viewerWrap.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        if (_highlightMode) {
          showTooltip(e.clientX, e.clientY,
            '<div class="tt-title">Residue</div>' +
            '<div class="tt-row"><span class="tt-label">Click on structure atoms for details</span></div>'
          );
          setTimeout(hideTooltip, 3000);
        }
        if (_measureMode) {
          _measurePoints.push({ x, y });
          if (_measurePoints.length === 2) {
            const dx = _measurePoints[1].x - _measurePoints[0].x;
            const dy = _measurePoints[1].y - _measurePoints[0].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            showTooltip(e.clientX, e.clientY,
              '<div class="tt-title">Distance</div>' +
              '<div class="tt-row"><span class="tt-label">Screen</span><span class="tt-val">' + dist.toFixed(1) + ' px</span></div>' +
              '<div class="tt-row"><span class="tt-label">Note</span><span class="tt-val">2D projection only</span></div>'
            );
            setTimeout(hideTooltip, 4000);
            _measurePoints = [];
          }
        }
      });
    }
  };

})();
