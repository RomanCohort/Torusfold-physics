/* charts.js — Canvas 2D chart primitives for TorusFold */
(function () {
  'use strict';
  const TF = window.TorusFold = window.TorusFold || {};
  const Charts = TF.Charts = {};

  const COLORS = {
    ok: '#34d399', warn: '#fbbf24', err: '#f87171',
    accent: '#4d8dff', accent2: '#00d4ff', accent3: '#a855f7',
    t1: '#f0f2f8', t2: '#8899cc', t3: '#556699',
    grid: 'rgba(56,89,160,0.12)', bg: 'rgba(4,6,14,0.4)',
  };

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
    bg.setAttribute('stroke', 'rgba(56,89,160,0.15)');
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
    t1.setAttribute('fill', '#556699');
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
    if (inv) return v > th.good ? COLORS.ok : v > th.warn ? COLORS.warn : COLORS.err;
    return v < th.good ? COLORS.ok : v < th.warn ? COLORS.warn : COLORS.err;
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
    const barColor = opts.barColor || COLORS.accent;
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
    ctx.strokeStyle = COLORS.grid;
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
    ctx.fillStyle = COLORS.t3;
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
        ctx.fillStyle = '#fff';
        ctx.font = '10px Inter, sans-serif';
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
      ctx.fillStyle = COLORS.t1;
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
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
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

  function divergingColor(t) {
    // t in [-1, 1]: blue (negative) → white (zero) → red (positive)
    t = Math.max(-1, Math.min(1, t));
    if (t < 0) {
      const k = 1 + t; // 0 at t=-1, 1 at t=0
      return `rgb(${Math.round(44 + (240 - 44) * k)}, ${Math.round(62 + (240 - 62) * k)}, ${Math.round(178 + (240 - 178) * k)})`;
    } else {
      const k = t;
      return `rgb(${Math.round(240 + (248 - 240) * k)}, ${Math.round(240 - (240 - 113) * k)}, ${Math.round(240 - (240 - 113) * k)})`;
    }
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
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * ch;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.t3;
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText((yMax - (i / 4) * (yMax - yMin)).toFixed(0), pad.left - 6, y + 3);
    }

    // Gradient fill
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
    grad.addColorStop(0, 'rgba(77,141,255,0.15)');
    grad.addColorStop(1, 'rgba(77,141,255,0.01)');
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
    lineGrad.addColorStop(0, COLORS.accent);
    lineGrad.addColorStop(0.5, COLORS.accent2);
    lineGrad.addColorStop(1, COLORS.accent3);
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
    ctx.fillStyle = COLORS.accent3;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastX, lastY, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(168,85,247,0.2)';
    ctx.fill();

    // X label
    ctx.fillStyle = COLORS.t3;
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

    const colors = [COLORS.accent, COLORS.accent2, COLORS.accent3, COLORS.ok, COLORS.warn];

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
      ctx.fillStyle = COLORS.t2;
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(items[i].label, x + barW / 2, H - 8);

      // Value on top
      ctx.fillStyle = COLORS.t1;
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
