/* console.js — SSE terminal panel for TorusFold */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Console = TF.Console = {};

  let _eventSource = null;
  let _lineCount = 0;
  const MAX_LINES = 2000;
  const PRUNE_TO = 1500;

  const LEVEL_CLASSES = {
    info: 'info', step: 'step', warn: 'warn', error: 'error',
    success: 'success', progress: 'progress', heartbeat: 'info',
  };

  function getOutput() {
    return document.getElementById('console-output');
  }

  function getWelcome() {
    return document.getElementById('console-welcome');
  }

  /** Append a log line to the terminal */
  Console.appendLine = function (entry) {
    const output = getOutput();
    if (!output) return;

    // Hide welcome message on first real line
    const welcome = getWelcome();
    if (welcome && _lineCount === 0) welcome.style.display = 'none';

    const div = document.createElement('div');
    div.className = 'console-line ' + (LEVEL_CLASSES[entry.level] || 'info');

    const ts = new Date(entry.timestamp * 1000);
    const timeStr = ts.toLocaleTimeString('en-US', { hour12: false });

    const tsSpan = document.createElement('span');
    tsSpan.className = 'ts';
    tsSpan.textContent = timeStr;

    div.appendChild(tsSpan);
    div.appendChild(document.createTextNode(entry.message));
    output.appendChild(div);

    _lineCount++;

    // Prune if too many lines
    if (_lineCount > MAX_LINES) {
      const children = output.children;
      for (let i = 0; i < _lineCount - PRUNE_TO; i++) {
        if (children[0]) children[0].remove();
      }
      _lineCount = PRUNE_TO;
    }

    // Auto-scroll
    const autoScroll = document.getElementById('console-autoscroll');
    if (autoScroll && autoScroll.checked) {
      output.scrollTop = output.scrollHeight;
    }
  };

  /** Connect to SSE stream */
  Console.connect = function (jobId) {
    Console.disconnect(); // clean up any existing

    const url = '/api/sse/' + jobId;
    try {
      _eventSource = new EventSource(url);
    } catch (e) {
      Console.appendLine({ timestamp: Date.now() / 1000, level: 'warn', message: 'SSE not supported, falling back to polling' });
      return false;
    }

    _eventSource.onmessage = function (e) {
      try {
        const data = JSON.parse(e.data);
        if (data.level === 'heartbeat') {
          // Update progress from heartbeat
          TF.EventBus && TF.EventBus.emit('sse:heartbeat', data);
          return;
        }
        Console.appendLine(data);
        TF.EventBus && TF.EventBus.emit('sse:log', data);
      } catch (err) { /* ignore parse errors */ }
    };

    _eventSource.addEventListener('done', function (e) {
      try {
        const data = JSON.parse(e.data);
        Console.appendLine({ timestamp: Date.now() / 1000, level: 'success', message: '=== Pipeline Complete ===' });
        TF.EventBus && TF.EventBus.emit('sse:done', data);
      } catch (err) { /* ignore */ }
      Console.disconnect();
    });

    _eventSource.addEventListener('error', function (e) {
      try {
        const data = JSON.parse(e.data);
        Console.appendLine({ timestamp: Date.now() / 1000, level: 'error', message: 'Error: ' + (data.message || 'unknown') });
        TF.EventBus && TF.EventBus.emit('sse:error', data);
      } catch (err) { /* ignore */ }
      Console.disconnect();
    });

    _eventSource.onerror = function () {
      // Connection lost — will auto-reconnect by EventSource spec
      // After a few failures, stop retrying
      if (_eventSource && _eventSource.readyState === EventSource.CLOSED) {
        Console.appendLine({ timestamp: Date.now() / 1000, level: 'warn', message: 'SSE connection closed' });
      }
    };

    Console.appendLine({ timestamp: Date.now() / 1000, level: 'step', message: '=== Connected to SSE stream ===' });
    return true;
  };

  /** Disconnect SSE */
  Console.disconnect = function () {
    if (_eventSource) {
      _eventSource.close();
      _eventSource = null;
    }
  };

  /** Clear console */
  Console.clear = function () {
    const output = getOutput();
    if (output) {
      output.innerHTML = '';
      _lineCount = 0;
    }
    const welcome = getWelcome();
    if (welcome) {
      output.appendChild(welcome);
      welcome.style.display = '';
    }
  };

  /** Copy all console text to clipboard */
  Console.copyAll = function () {
    const output = getOutput();
    if (!output) return;
    const text = Array.from(output.querySelectorAll('.console-line'))
      .map(el => el.textContent).join('\n');
    navigator.clipboard.writeText(text).then(function () {
      TF.App && TF.App.showToast('Console output copied', 'success');
    });
  };

  /** Check if SSE is available */
  Console.isAvailable = function () {
    return typeof EventSource !== 'undefined';
  };

})();
