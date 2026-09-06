/* export.js — Screenshot / PDB / JSON export for TorusFold */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Export = TF.Export = {};

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 100);
  }

  function timestamp() {
    return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  }

  /** Export screenshot from Mol* viewport */
  Export.screenshot = async function () {
    const viewer = TF.Viewer && TF.Viewer.instance;
    if (!viewer || !viewer.plugin) {
      TF.App && TF.App.showToast('No structure loaded', 'error');
      return;
    }

    try {
      // Method 1: Mol* viewport screenshot API
      const helpers = viewer.plugin.helpers;
      if (helpers && helpers.viewportScreenshot) {
        const dataUri = await helpers.viewportScreenshot.getImageDataUri();
        const response = await fetch(dataUri);
        const blob = await response.blob();
        downloadBlob(blob, 'torusfold_' + timestamp() + '.png');
        TF.App && TF.App.showToast('Screenshot saved', 'success');
        return;
      }
    } catch (e) {
      console.warn('Mol* screenshot failed, trying fallback:', e);
    }

    // Method 2: Direct canvas read
    try {
      const canvas = viewer.plugin.canvas3d && viewer.plugin.canvas3d.webgl && viewer.plugin.canvas3d.webgl.gl.canvas;
      if (canvas) {
        canvas.toBlob(function (blob) {
          if (blob) {
            downloadBlob(blob, 'torusfold_' + timestamp() + '.png');
            TF.App && TF.App.showToast('Screenshot saved (fallback)', 'success');
          }
        }, 'image/png');
        return;
      }
    } catch (e) {
      console.warn('Canvas screenshot fallback failed:', e);
    }

    TF.App && TF.App.showToast('Screenshot not available', 'error');
  };

  /** Export PDB file */
  Export.pdb = function () {
    const state = TF.State;
    if (!state || !state.pdb) {
      TF.App && TF.App.showToast('No PDB data', 'error');
      return;
    }
    const blob = new Blob([state.pdb], { type: 'chemical/x-pdb' });
    const name = 'torusfold_' + (state.length || 'unknown') + 'nt.pdb';
    downloadBlob(blob, name);
    TF.App && TF.App.showToast('PDB downloaded', 'success');
  };

  /** Export full result JSON */
  Export.json = function () {
    const state = TF.State;
    if (!state || !state.result) {
      TF.App && TF.App.showToast('No result data', 'error');
      return;
    }
    const json = JSON.stringify(state.result, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const name = 'torusfold_' + (state.length || 'unknown') + 'nt.json';
    downloadBlob(blob, name);
    TF.App && TF.App.showToast('JSON downloaded', 'success');
  };

  /** Bind export buttons */
  Export.init = function () {
    const btnScreenshot = document.getElementById('btn-screenshot');
    const btnPdb = document.getElementById('btn-pdb');
    const btnJson = document.getElementById('btn-json');

    if (btnScreenshot) btnScreenshot.addEventListener('click', Export.screenshot);
    if (btnPdb) btnPdb.addEventListener('click', Export.pdb);
    if (btnJson) btnJson.addEventListener('click', Export.json);

    // Keyboard shortcut: Ctrl+Shift+S
    document.addEventListener('keydown', function (e) {
      if (e.ctrlKey && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        Export.screenshot();
      }
    });
  };

})();
