/* charts.js — Canvas 2D chart primitives for TorusFold */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Charts = TF.Charts = {};

  /* Canvas 2D and SVG attributes cannot resolve var(), so the design tokens are
     read off the document once and cached. Never hard-code a colour here again:
     the charts would stop matching the panels around them, and would not follow
     the theme switch. */
  const FALLBACK = {
    ok:'#1d7a4c', warn:'#8a5f10', err:'#b3352a', accent:'#0f7d72',
    'accent-ink':'#ffffff', ink:'#151a18', 'ink-2':'#46504d',
    'ink-3':'#5e6a66', 'hair-2':'#c4bfb1', panel:'#fffefb', well:'#f1efe9',
  };
  let _tokenCache = null;
  function tokenColor(name) {
    if (!_tokenCache) {
      _tokenCache = {};
      const probe = document.createElement('canvas').getContext('2d');
      for (const k in FALLBACK) {
        const raw = getComputedStyle(document.documentElement).getPropertyValue('--' + k).trim();
        probe.fillStyle = '#000';
        probe.fillStyle = raw || FALLBACK[k];   // an invalid value leaves fillStyle untouched
        _tokenCache[k] = probe.fillStyle;
      }
    }
    return _tokenCache[name] || FALLBACK[name] || '#000';
  }
  /* Charts drawn before a theme switch keep the old colours until redrawn. */
  Charts.refreshTokens = function () { _tokenCache = null; };

  function C(name) { return tokenColor(name); }

  /* ── SVG Gauge (semicircle) ── */
  Charts.createGauge = function (value, max, title, unit, thresholds, invert) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 120 75');
    const cx = 60, cy = 60, r = 45, sa = Math.PI, ea = 0;

    // Background arc
    const bg = document.createElementNS(ns, 'path');
    bg.setAttribute('d', describeArc(cx, cy, r, sa, ea));
    bg.setAttribute('fill', 'none');
    bg.setAttribute('stroke', C('hair-2'));
    bg.setAttribute('stroke-width', '8');
    bg.setAttribute('stroke-linecap', 'round');
    svg.appendChild(bg);

    // Value arc
    const frac = Math.max(0, Math.min(1, value / max));
    const va = sa - frac * Math.PI;
    const col = gaugeColor(value, thresholds, invert);

    const va2 = document.createElementNS(ns, 'path');
    va2.setAttribute('d', describeArc(cx, cy, r, sa, va));
    va2.setAttribute('fill', 'none');
    va2.setAttribute('stroke', col);
    va2.setAttribute('stroke-width', '8');
    va2.setAttribute('stroke-linecap', 'round');
    svg.appendChild(va2);

    // Dot at end
    const nx = cx + (r - 2) * Math.cos(va);
    const ny = cy - (r - 2) * Math.sin(va);
    const dot = document.createElementNS(ns, 'circle');
    dot.setAttribute('cx', nx);
    dot.setAttribute('cy', ny);
    dot.setAttribute('r', '3');
    dot.setAttribute('fill', col);
    svg.appendChild(dot);

    // Title
    const t1 = document.createElementNS(ns, 'text');
    t1.setAttribute('x', cx);
    t1.setAttribute('y', 12);
    t1.setAttribute('text-anchor', 'middle');
    t1.setAttribute('fill', C('ink-3'));
    t1.setAttribute('font-size', '10');
    t1.textContent = title;
    svg.appendChild(t1);

    // Value
    const t2 = document.createElementNS(ns, 'text');
    t2.setAttribute('x', cx);
    t2.setAttribute('y', cy + 4);
    t2.setAttribute('text-anchor', 'middle');
    t2.setAttribute('fill', col);
    t2.setAttribute('font-size', '14');
    t2.setAttribute('font-weight', '700');
    t2.textContent = (typeof value === 'number' ? value.toFixed(2) : value) + (unit ? ' ' + unit : '');
    svg.appendChild(t2);

    return svg;
  };

  function describeArc(cx, cy, r, sa, ea) {
    const x1 = cx + r * Math.cos(sa), y1 = cy - r * Math.sin(sa);
    const x2 = cx + r * Math.cos(ea), y2 = cy - r * Math.sin(ea);
    const sw = sa - ea;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${sw > Math.PI ? 1 : 0} 1 ${x2} ${y2}`;
  }

  function gaugeColor(v, th, inv) {
    if (inv) return v > th.good ? C('ok') : v > th.warn ? C('warn') : C('err');
    return v < th.good ? C('ok') : v < th.warn ? C('warn') : C('err');
  }

  /* ── Canvas Histogram ── */
  Charts.drawHistogram = function (canvas, values, opts) {
    if (!canvas || !values || values.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || canvas.width;
    const H = canvas.clientHeight || canvas.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const binWidth = opts.binWidth || 2;
    const barColor = opts.barColor || C('accent');
    const label = opts.label || '';

    // Compute bins
    const maxVal = Math.max(...values);
    const binCount = Math.ceil(maxVal / binWidth) + 1;
    const bins = new Array(binCount).fill(0);
    for (const v of values) bins[Math.floor(v / binWidth)]++;

    const maxBin = Math.max(...bins);
    const pad = { top: 16, right: 10, bottom: 24, left: 30 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;
    const barW = Math.max(2, cw / binCount - 1);

    // Grid
    ctx.strokeStyle = C('hair-2');
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = pad.top + (i / 3) * ch;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
    }

    // Bars
    for (let i = 0; i < binCount; i++) {
      const x = pad.left + (i / binCount) * cw;
      const h = maxBin > 0 ? (bins[i] / maxBin) * ch : 0;
      ctx.fillStyle = barColor;
      ctx.globalAlpha = 0.8;
      ctx.fillRect(x, pad.top + ch - h, barW, h);
      ctx.globalAlpha = 1;
    }

    // X-axis label
    ctx.fillStyle = C('ink-3');
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label || `bin width=${binWidth}`, W / 2, H - 4);

    // Y-axis max
    ctx.textAlign = 'right';
    ctx.fillText(maxBin.toString(), pad.left - 4, pad.top + 4);
  };

  /* ── Canvas Bar Chart (horizontal stacked) ── */
  Charts.drawStackedBar = function (canvas, segments, opts) {
    if (!canvas || !segments || segments.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || canvas.width;
    const H = canvas.clientHeight || canvas.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const pad = { top: 4, right: 10, bottom: 4, left: 10 };
    const barH = H - pad.top - pad.bottom;
    const total = segments.reduce((s, seg) => s + seg.value, 0);
    if (total === 0) return;

    let x = pad.left;
    for (const seg of segments) {
      const w = (seg.value / total) * (W - pad.left - pad.right);
      ctx.fillStyle = seg.color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x, pad.top, w, barH);
      ctx.globalAlpha = 1;

      // Label inside if wide enough
      if (w > 30) {
        ctx.fillStyle = C('accent-ink');
        ctx.font = '10px "Segoe UI", Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(seg.label, x + w / 2, pad.top + barH / 2 + 4);
      }
      x += w;
    }
  };

  /* ── Canvas Donut Chart ── */
  Charts.drawDonut = function (canvas, segments, opts) {
    if (!canvas || !segments || segments.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || canvas.width;
    const H = canvas.clientHeight || canvas.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const r = Math.min(W, H) / 2 - 4;
    const innerR = r * 0.55;
    const total = segments.reduce((s, seg) => s + seg.value, 0);
    if (total === 0) return;

    let startAngle = -Math.PI / 2;
    for (const seg of segments) {
      const sweep = (seg.value / total) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, startAngle + sweep);
      ctx.arc(cx, cy, innerR, startAngle + sweep, startAngle, true);
      ctx.closePath();
      ctx.fillStyle = seg.color;
      ctx.fill();
      startAngle += sweep;
    }

    // Center text
    if (opts && opts.centerText) {
      ctx.fillStyle = C('ink');
      ctx.font = 'bold 14px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(opts.centerText, cx, cy + 5);
    }
  };

  /* ── Canvas Heatmap Strip (per-residue) ── */
  Charts.drawHeatmapStrip = function (canvas, values, regionBounds) {
    if (!canvas || !values || values.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || 600;
    const H = canvas.clientHeight || 20;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // Find range
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const range = Math.max(Math.abs(lo), Math.abs(hi)) || 1;

    const perNt = W / values.length;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      const normalized = v / range; // -1 to 1
      ctx.fillStyle = divergingColor(normalized);
      ctx.fillRect(i * perNt, 0, perNt + 0.5, H);
    }

    // Region boundaries
    if (regionBounds) {
      ctx.strokeStyle = C('hair-2');
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1;
      for (const [name, bounds] of Object.entries(regionBounds)) {
        if (bounds.start != null) {
          const x = bounds.start * perNt;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, H);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }
  };

  /* Parse any resolved CSS colour into [r, g, b] so two tokens can be mixed. */
  function rgbOf(name) {
    const probe = document.createElement('canvas').getContext('2d');
    probe.fillStyle = '#000';
    probe.fillStyle = tokenColor(name);
    const hex = probe.fillStyle;
    if (hex.charAt(0) === '#') {
      return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
    }
    const m = hex.match(/\d+/g);
    return m ? [+m[0], +m[1], +m[2]] : [128, 128, 128];
  }

  /* Diverging ramp anchored on tokens, so it stays legible on either theme:
     accent for negative, panel at zero, error for positive. A white midpoint --
     what this used to hard-code -- vanishes against the dark panel. */
  function divergingColor(t) {
    t = Math.max(-1, Math.min(1, t));
    const end = t < 0 ? rgbOf('accent') : rgbOf('err');
    const mid = rgbOf('panel');
    const k = Math.abs(t);
    return 'rgb(' + Math.round(mid[0] + (end[0] - mid[0]) * k) + ', ' +
                    Math.round(mid[1] + (end[1] - mid[1]) * k) + ', ' +
                    Math.round(mid[2] + (end[2] - mid[2]) * k) + ')';
  }

  /* ── Canvas Line Chart (energy convergence) ── */
  Charts.drawLineChart = function (canvas, data, opts) {
    if (!canvas || !data || data.length < 2) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || canvas.width;
    const H = canvas.clientHeight || canvas.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const pad = { top: 20, right: 16, bottom: 24, left: 48 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;

    let yMin = Infinity, yMax = -Infinity;
    for (const v of data) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    const yRange = (yMax - yMin) || 1;
    yMin -= yRange * 0.1;
    yMax += yRange * 0.1;

    const toX = i => pad.left + (i / (data.length - 1)) * cw;
    const toY = v => pad.top + (1 - (v - yMin) / (yMax - yMin)) * ch;

    // Grid
    ctx.strokeStyle = C('hair-2');
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * ch;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      ctx.fillStyle = C('ink-3');
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText((yMax - (i / 4) * (yMax - yMin)).toFixed(0), pad.left - 6, y + 3);
    }

    // Gradient fill
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
    grad.addColorStop(0, C('hair-2'));
    grad.addColorStop(1, C('well'));
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(toX(i), toY(data[i]));
    ctx.lineTo(toX(data.length - 1), pad.top + ch);
    ctx.lineTo(toX(0), pad.top + ch);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    const lineGrad = ctx.createLinearGradient(pad.left, 0, W - pad.right, 0);
    lineGrad.addColorStop(0, C('accent'));
    lineGrad.addColorStop(0.5, C('accent'));
    lineGrad.addColorStop(1, C('accent'));
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(toX(i), toY(data[i]));
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // End dot
    const lastX = toX(data.length - 1), lastY = toY(data[data.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fillStyle = C('accent');
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastX, lastY, 8, 0, Math.PI * 2);
    ctx.fillStyle = C('hair-2');
    ctx.fill();

    // X label
    ctx.fillStyle = C('ink-3');
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(opts && opts.xLabel || 'Step', W / 2, H - 4);
  };

  /* ── Canvas Bar Chart (vertical, for eigenvalues etc.) ── */
  Charts.drawBarChart = function (canvas, items, opts) {
    if (!canvas || !items || items.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || canvas.width;
    const H = canvas.clientHeight || canvas.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const pad = { top: 16, right: 16, bottom: 28, left: 50 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;
    const maxVal = Math.max(...items.map(i => i.value));
    const barW = Math.min(40, cw / items.length * 0.6);

    const colors = [C('accent'), C('accent'), C('accent'), C('ok'), C('warn')];

    for (let i = 0; i < items.length; i++) {
      const x = pad.left + ((i + 0.5) / items.length) * cw - barW / 2;
      const h = maxVal > 0 ? (items[i].value / maxVal) * ch : 0;
      ctx.fillStyle = items[i].color || colors[i % colors.length];
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.roundRect(x, pad.top + ch - h, barW, h, [4, 4, 0, 0]);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Label
      ctx.fillStyle = C('ink-2');
      ctx.font = '10px "Segoe UI", Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(items[i].label, x + barW / 2, H - 8);

      // Value on top
      ctx.fillStyle = C('ink');
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(items[i].value.toFixed(1), x + barW / 2, pad.top + ch - h - 4);
    }
  };

  /* ── Metric Value Counter Animation ── */
  Charts.animateValue = function (element, start, end, duration, formatter) {
    const startTime = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const current = start + (end - start) * eased;
      element.textContent = formatter ? formatter(current) : current.toFixed(2);
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  };

})();
