'use strict';

/*
 * Fish Motion Counter
 *
 * Pipeline per frame (all at a reduced "analysis" resolution):
 *   1. colour difference |frame - background| (max over R,G,B)
 *   2. 3x3 box blur of the difference map (suppresses sensor noise)
 *   3. threshold -> binary motion mask
 *   4. background update (running average; slower where motion is detected)
 *   5. morphological opening (remove speckles) + closing (merge fragments)
 *   6. 8-connected component labelling -> groups, filtered by minimum area
 *   7. tracking (tracker.js): groups are linked over time into fish with IDs and paths;
 *      short-lived or static groups are discarded, hidden / out-of-view fish are remembered
 */

const $ = (id) => document.getElementById(id);

const DEFAULTS = {
  threshold: 25,
  minArea: 20,
  merge: 2,
  learn: 0.02,
  procWidth: 320,
  confirmTime: 0.5,
  hiddenMemory: 20,
  outMemory: 120,
  trailSeconds: 10,
  showMask: true,
  showTracks: true,
  showBoxes: true,
};
const TRACKER_KEYS = ['confirmTime', 'hiddenMemory', 'outMemory', 'trailSeconds'];
const STORE_KEY = 'fish-counter-settings-v1';
const WARMUP_FRAMES = 45;          // frames used to build the initial background
const WARM_STEP = 3;               // every 3rd warm-up frame is kept -> 15 samples
const WARM_SAMPLES = Math.ceil(WARMUP_FRAMES / WARM_STEP);
const GLOBAL_CHANGE_FRACTION = 0.4; // more motion than this => camera moved / light changed
const MAX_VIEW_WIDTH = 960;
const STATIC_ABSORB = 0.15;        // background learning rate inside static, unconfirmed groups

// ---------- settings ----------
const settings = loadSettings();

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return Object.assign({}, DEFAULTS, JSON.parse(raw));
  } catch (_) { /* storage unavailable */ }
  return Object.assign({}, DEFAULTS);
}
function saveSettings() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(settings)); } catch (_) { /* ignore */ }
}

// ---------- DOM ----------
const video = $('video');
const view = $('view');
const vctx = view.getContext('2d');
const work = document.createElement('canvas');
const wctx = work.getContext('2d', { willReadFrequently: true });
const maskCanvas = document.createElement('canvas');
const mctx = maskCanvas.getContext('2d');

const ui = {
  placeholder: $('placeholder'),
  stage: document.querySelector('.stage'),
  status: $('status'),
  countTotal: $('countTotal'),
  countVisible: $('countVisible'),
  countHidden: $('countHidden'),
  countOut: $('countOut'),
  countNow: $('countNow'),
  events: $('events'),
  fps: $('fps'),
  cameraSelect: $('cameraSelect'),
  roiHint: $('roiHint'),
  buttons: ['pauseBtn', 'resetBtn', 'roiBtn', 'roiClearBtn', 'recBtn'].map($),
};

// ---------- processing state ----------
let W = 0, H = 0;
let bg, warmBuf, diff, blurTmp, blurred, mask, maskA, maskB, labels, stack, maskImage;
let frameCount = 0;
let lastVideoTime = -1;
let paused = false;
let running = false;
let stream = null;
let roi = null;                // {x0,y0,x1,y1} normalised 0..1, or null for the whole frame
let roiDrawing = null;
let roiMode = false;
let blobs = [];
let fishAreaHistory = [];      // areas of groups that belong to confirmed fish
let fishSize = 8;              // typical fish length in analysis pixels
let lastProcAt = 0;
let lastResult = { raw: 0, fg: 0, warming: true };
const tracker = new FishTracker(trackerOptions());
let recentEvents = [];
let globalChangeUntil = 0;
let lastDims = { w: 0, h: 0 };
let fpsFrames = 0, fpsSince = performance.now();
let wakeLock = null;
let log = [];                  // count rows (one per second while logging)
let pathLog = [];              // fish positions (10 per second while logging)
const PATH_LOG_MAX = 1000000;
let logging = false;
let logStart = 0, lastLogAt = 0, lastPathLogAt = 0;

function trackerOptions() {
  const o = {};
  TRACKER_KEYS.forEach((k) => { o[k] = settings[k]; });
  return o;
}

function allocate() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return false;
  W = Math.min(settings.procWidth, vw);
  H = Math.max(1, Math.round(W * vh / vw));
  const n = W * H;
  work.width = W; work.height = H;
  maskCanvas.width = W; maskCanvas.height = H;
  bg = new Float32Array(n * 3);
  warmBuf = new Uint8Array(WARM_SAMPLES * n * 3);
  diff = new Float32Array(n);
  blurTmp = new Float32Array(n);
  blurred = new Float32Array(n);
  mask = new Uint8Array(n);
  maskA = new Uint8Array(n);
  maskB = new Uint8Array(n);
  labels = new Int32Array(n);
  stack = new Int32Array(n);
  maskImage = mctx.createImageData(W, H);

  const scale = Math.min(1, MAX_VIEW_WIDTH / vw);
  view.width = Math.round(vw * scale);
  view.height = Math.round(vh * scale);
  resetBackground();
  resetTracking();
  return true;
}

function resetBackground() {
  frameCount = 0;
}

function resetTracking() {
  tracker.reset();
  fishAreaHistory = [];
  recentEvents = [];
  renderEvents();
}

// ---------- image processing ----------
function roiPixels() {
  if (!roi) return { x0: 0, y0: 0, x1: W, y1: H };
  return {
    x0: Math.max(0, Math.floor(roi.x0 * W)),
    y0: Math.max(0, Math.floor(roi.y0 * H)),
    x1: Math.min(W, Math.ceil(roi.x1 * W)),
    y1: Math.min(H, Math.ceil(roi.y1 * H)),
  };
}

// Separable box blur, radius 1, on a float map.
function blur3(src, tmp, dst) {
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const l = x > 0 ? src[row + x - 1] : src[row + x];
      const r = x < W - 1 ? src[row + x + 1] : src[row + x];
      tmp[row + x] = (l + src[row + x] + r) / 3;
    }
  }
  for (let y = 0; y < H; y++) {
    const up = (y > 0 ? y - 1 : y) * W, dn = (y < H - 1 ? y + 1 : y) * W, row = y * W;
    for (let x = 0; x < W; x++) {
      dst[row + x] = (tmp[up + x] + tmp[row + x] + tmp[dn + x]) / 3;
    }
  }
}

// Binary erosion/dilation with a (2r+1)^2 square, separable running-sum implementation.
// Pixels outside the image are ignored (neither set nor unset).
function morph(src, dst, tmp, r, erode) {
  if (r <= 0) { dst.set(src); return; }
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let sum = 0;
    for (let x = 0; x <= Math.min(r, W - 1); x++) sum += src[row + x];
    for (let x = 0; x < W; x++) {
      const len = Math.min(W - 1, x + r) - Math.max(0, x - r) + 1;
      tmp[row + x] = erode ? (sum === len ? 1 : 0) : (sum > 0 ? 1 : 0);
      if (x + r + 1 < W) sum += src[row + x + r + 1];
      if (x - r >= 0) sum -= src[row + x - r];
    }
  }
  for (let x = 0; x < W; x++) {
    let sum = 0;
    for (let y = 0; y <= Math.min(r, H - 1); y++) sum += tmp[y * W + x];
    for (let y = 0; y < H; y++) {
      const len = Math.min(H - 1, y + r) - Math.max(0, y - r) + 1;
      dst[y * W + x] = erode ? (sum === len ? 1 : 0) : (sum > 0 ? 1 : 0);
      if (y + r + 1 < H) sum += tmp[(y + r + 1) * W + x];
      if (y - r >= 0) sum -= tmp[(y - r) * W + x];
    }
  }
}

// 8-connected component labelling via an explicit stack flood fill.
function findBlobs(m, minArea) {
  labels.fill(0);
  const out = [];
  let next = 0;
  for (let start = 0; start < m.length; start++) {
    if (!m[start] || labels[start]) continue;
    next++;
    let sp = 0;
    stack[sp++] = start;
    labels[start] = next;
    let area = 0, sx = 0, sy = 0;
    let minX = W, minY = H, maxX = 0, maxY = 0;
    while (sp > 0) {
      const p = stack[--sp];
      const px = p % W, py = (p - px) / W;
      area++; sx += px; sy += py;
      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = py + dy;
        if (ny < 0 || ny >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx;
          if (nx < 0 || nx >= W || (dx === 0 && dy === 0)) continue;
          const q = ny * W + nx;
          if (m[q] && !labels[q]) { labels[q] = next; stack[sp++] = q; }
        }
      }
    }
    if (area >= minArea) {
      out.push({ area, cx: sx / area, cy: sy / area, minX, minY, maxX, maxY, label: next });
    }
  }
  return out;
}

function median(arr) {
  if (!arr.length) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function storeWarmSample(px, s) {
  const n = W * H;
  const off = s * n * 3;
  for (let i = 0, k = 0, j = off; i < n; i++, k += 4, j += 3) {
    warmBuf[j] = px[k]; warmBuf[j + 1] = px[k + 1]; warmBuf[j + 2] = px[k + 2];
  }
}

function medianBackground() {
  const len = W * H * 3;
  const vals = new Uint8Array(WARM_SAMPLES);
  const mid = WARM_SAMPLES >> 1;
  for (let c = 0; c < len; c++) {
    for (let s = 0; s < WARM_SAMPLES; s++) {
      // insertion sort while gathering (15 values)
      const v = warmBuf[s * len + c];
      let t = s - 1;
      while (t >= 0 && vals[t] > v) { vals[t + 1] = vals[t]; t--; }
      vals[t + 1] = v;
    }
    bg[c] = vals[mid];
  }
}

function processFrame() {
  wctx.drawImage(video, 0, 0, W, H);
  const px = wctx.getImageData(0, 0, W, H).data;

  // Warm-up: collect samples, then take the per-pixel temporal median as the
  // background. Unlike an average, the median ignores fish swimming through.
  if (frameCount < WARMUP_FRAMES) {
    if (frameCount % WARM_STEP === 0) storeWarmSample(px, frameCount / WARM_STEP);
    frameCount++;
    if (frameCount === WARMUP_FRAMES) medianBackground();
    blobs = [];
    lastResult = { raw: 0, fg: 0, warming: true };
    return;
  }

  const { x0, y0, x1, y1 } = roiPixels();

  // 1. colour difference to background
  diff.fill(0);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * W + x, j = i * 3, k = i * 4;
      const dr = Math.abs(px[k] - bg[j]);
      const dg = Math.abs(px[k + 1] - bg[j + 1]);
      const db = Math.abs(px[k + 2] - bg[j + 2]);
      diff[i] = dr > dg ? (dr > db ? dr : db) : (dg > db ? dg : db);
    }
  }

  // 2.+3. blur and threshold
  blur3(diff, blurTmp, blurred);
  const thr = settings.threshold;
  let fg = 0;
  const roiArea = Math.max(1, (x1 - x0) * (y1 - y0));
  mask.fill(0);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * W + x;
      if (blurred[i] > thr) { mask[i] = 1; fg++; }
    }
  }
  const fgFraction = fg / roiArea;

  // Camera moved or lighting changed: re-learn the background from scratch.
  if (fgFraction > GLOBAL_CHANGE_FRACTION) {
    globalChangeUntil = performance.now() + 2000;
    resetBackground();
    blobs = [];
    lastResult = { raw: 0, fg: 0, warming: true };
    return;
  }

  // 4. background update (whole frame, so the ROI can be changed without re-learning).
  // Pixels marked as moving adapt 4x slower, so passing fish are not absorbed.
  const alpha = settings.learn;
  const slow = alpha * 0.25;
  const n = W * H;
  for (let i = 0; i < n; i++) {
    const a = mask[i] ? slow : alpha;
    const j = i * 3, k = i * 4;
    bg[j] += a * (px[k] - bg[j]);
    bg[j + 1] += a * (px[k + 1] - bg[j + 1]);
    bg[j + 2] += a * (px[k + 2] - bg[j + 2]);
  }
  frameCount++;

  // 5. opening (radius 1) removes speckles, closing (merge radius) joins fragments
  morph(mask, maskA, maskB, 1, true);
  morph(maskA, mask, maskB, 1, false);
  const r = settings.merge;
  if (r > 0) {
    morph(mask, maskA, maskB, r, false);
    morph(maskA, mask, maskB, r, true);
  }

  // 6. connected groups
  const minArea = Math.max(1, Math.round(settings.minArea * (W / 320) * (W / 320)));
  blobs = findBlobs(mask, minArea);

  // 7. tracking
  const t = performance.now();
  const dt = lastProcAt ? (t - lastProcAt) / 1000 : 1 / 30;
  lastProcAt = t;
  const counts = tracker.update(blobs, dt, { x0, y0, x1, y1 }, fishSize);

  // Groups that stay put without ever becoming a fish (a "ghost" of a fish that sat in
  // the start-up background, a resting snail, a reflection) are absorbed quickly into
  // the background so they stop producing detections.
  for (const c of tracker.staticCandidates(fishSize)) {
    const b = c.box;
    const bx0 = Math.max(0, b.minX - 2), bx1 = Math.min(W - 1, b.maxX + 2);
    const by0 = Math.max(0, b.minY - 2), by1 = Math.min(H - 1, b.maxY + 2);
    for (let y = by0; y <= by1; y++) {
      for (let x = bx0; x <= bx1; x++) {
        const i = y * W + x, j = i * 3, k = i * 4;
        bg[j] += STATIC_ABSORB * (px[k] - bg[j]);
        bg[j + 1] += STATIC_ABSORB * (px[k + 1] - bg[j + 1]);
        bg[j + 2] += STATIC_ABSORB * (px[k + 2] - bg[j + 2]);
      }
    }
  }
  for (const e of tracker.events) {
    if (e.minor) continue; // brief losses of track would flood the list
    recentEvents.unshift(e);
  }
  if (tracker.events.length) { recentEvents.length = Math.min(recentEvents.length, 6); renderEvents(); }

  // Typical fish size, learnt from groups that belong to confirmed fish.
  for (const b of blobs) if (b.trackId) fishAreaHistory.push(b.area);
  while (fishAreaHistory.length > 600) fishAreaHistory.shift();
  const minSize = 6 * (W / 320);
  fishSize = fishAreaHistory.length >= 20
    ? Math.max(minSize, 1.4 * Math.sqrt(median(fishAreaHistory)))
    : Math.max(minSize, 1.4 * Math.sqrt(minArea * 4));

  lastResult = Object.assign({ raw: blobs.length, fg: fgFraction, warming: false }, counts);
}

function renderEvents() {
  ui.events.textContent = '';
  for (const e of recentEvents) {
    const li = document.createElement('li');
    li.textContent = `${formatClock(e.t)}  Fish #${e.id} ${EVENT_TEXT[e.type] || e.type}`;
    ui.events.appendChild(li);
  }
}
const EVENT_TEXT = {
  'new fish': 'detected (new fish)',
  'left view': 'left the view',
  're-entered': 'came back into view',
  'reappeared': 'reappeared after hiding',
  'forgotten': 'forgotten (not seen for too long)',
};
function formatClock(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

// ---------- drawing ----------
function draw() {
  const vw = view.width, vh = view.height;
  vctx.drawImage(video, 0, 0, vw, vh);
  const sx = vw / W, sy = vh / H;

  if (settings.showMask && frameCount > WARMUP_FRAMES) {
    const d = maskImage.data;
    for (let i = 0, k = 0; i < mask.length; i++, k += 4) {
      if (mask[i]) { d[k] = 0; d[k + 1] = 255; d[k + 2] = 200; d[k + 3] = 120; }
      else d[k + 3] = 0;
    }
    mctx.putImageData(maskImage, 0, 0);
    vctx.imageSmoothingEnabled = false;
    vctx.drawImage(maskCanvas, 0, 0, vw, vh);
    vctx.imageSmoothingEnabled = true;
  }

  const showRoi = roiDrawing || roi;
  if (showRoi) {
    const r = normRect(showRoi);
    vctx.save();
    vctx.fillStyle = 'rgba(0,0,0,0.45)';
    vctx.beginPath();
    vctx.rect(0, 0, vw, vh);
    vctx.rect(r.x0 * vw, r.y0 * vh, (r.x1 - r.x0) * vw, (r.y1 - r.y0) * vh);
    vctx.fill('evenodd');
    vctx.setLineDash([8, 6]);
    vctx.lineWidth = 2;
    vctx.strokeStyle = '#ffd34d';
    vctx.strokeRect(r.x0 * vw, r.y0 * vh, (r.x1 - r.x0) * vw, (r.y1 - r.y0) * vh);
    vctx.restore();
  }

  const lw = Math.max(2, vw / 400);
  const fontPx = Math.max(12, Math.round(vw / 45));

  // Candidates (not counted yet): thin grey boxes.
  if (settings.showBoxes) {
    vctx.save();
    vctx.setLineDash([4, 4]);
    vctx.lineWidth = 1;
    vctx.strokeStyle = 'rgba(255,255,255,0.7)';
    for (const t of tracker.tracks) {
      if (t.state !== 'candidate' || t.misses) continue;
      const b = t.box;
      vctx.strokeRect(b.minX * sx, b.minY * sy, (b.maxX - b.minX + 1) * sx, (b.maxY - b.minY + 1) * sy);
    }
    vctx.restore();
  }

  if (settings.showTracks) {
    vctx.save();
    vctx.lineCap = 'round';
    vctx.lineJoin = 'round';
    vctx.font = `bold ${fontPx}px system-ui, sans-serif`;
    vctx.textBaseline = 'bottom';
    for (const t of tracker.tracks) {
      if (!t.id) continue;
      const color = `hsl(${t.hue}, 95%, 60%)`;
      const faded = t.state !== 'visible';

      // path: solid where observed, dashed across gaps (hidden / out of view)
      vctx.globalAlpha = faded ? 0.5 : 0.9;
      vctx.strokeStyle = color;
      vctx.lineWidth = lw;
      for (const dashed of [false, true]) {
        vctx.setLineDash(dashed ? [6, 6] : []);
        vctx.beginPath();
        for (let i = 1; i < t.path.length; i++) {
          if (!!t.path[i].gap !== dashed) continue;
          vctx.moveTo(t.path[i - 1].x * sx, t.path[i - 1].y * sy);
          vctx.lineTo(t.path[i].x * sx, t.path[i].y * sy);
        }
        vctx.stroke();
      }
      vctx.setLineDash([]);
      vctx.globalAlpha = 1;

      if (t.state === 'visible') {
        const b = t.box;
        const x = b.minX * sx, y = b.minY * sy;
        const w = (b.maxX - b.minX + 1) * sx, h = (b.maxY - b.minY + 1) * sy;
        vctx.strokeStyle = color;
        vctx.strokeRect(x, y, w, h);
        drawLabel(`#${t.id}`, x, y, h, color, fontPx);
      } else if (t.state === 'hidden') {
        const r = Math.max(8, fishSize * sx * 0.7);
        vctx.globalAlpha = 0.85;
        vctx.strokeStyle = color;
        vctx.setLineDash([4, 4]);
        vctx.beginPath();
        vctx.arc(t.x * sx, t.y * sy, r, 0, Math.PI * 2);
        vctx.stroke();
        vctx.setLineDash([]);
        drawLabel(`#${t.id} hidden`, t.x * sx - r, t.y * sy - r, 2 * r, color, fontPx);
        vctx.globalAlpha = 1;
      } else if (t.state === 'out') {
        const x = Math.min(vw - 60, Math.max(0, t.exitX * sx - 20));
        const y = Math.min(vh - 4, Math.max(fontPx + 8, t.exitY * sy));
        vctx.globalAlpha = 0.75;
        drawLabel(`#${t.id} out`, x, y, 0, color, fontPx);
        vctx.globalAlpha = 1;
      }
    }
    vctx.restore();
  }
}

function drawLabel(text, x, y, h, color, fontPx) {
  const tw = vctx.measureText(text).width + 8;
  const th = fontPx + 4;
  const ty = y - th < 0 ? y + h + th : y;
  vctx.fillStyle = color;
  vctx.fillRect(x, ty - th, tw, th);
  vctx.fillStyle = '#05141b';
  vctx.fillText(text, x + 4, ty - 2);
}

// ---------- main loop ----------
function loop() {
  if (!running) return;
  requestAnimationFrame(loop);
  if (paused || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  if (!W || video.videoWidth !== lastDims.w || video.videoHeight !== lastDims.h || W !== Math.min(settings.procWidth, video.videoWidth)) {
    if (!allocate()) return;
    lastDims = { w: video.videoWidth, h: video.videoHeight };
  }

  processFrame();
  draw();
  updateReadout();

  fpsFrames++;
  const t = performance.now();
  if (t - fpsSince >= 1000) {
    const st = tracker.stats;
    ui.fps.textContent = `${Math.round(fpsFrames * 1000 / (t - fpsSince))} fps · analysis ${W}×${H} · ` +
      `fish IDs created ${st.created} · re-identified ${st.reidentified} · false detections discarded ${st.discarded}`;
    fpsFrames = 0; fpsSince = t;
  }
  if (logging && !lastResult.warming) {
    const elapsed = ((t - logStart) / 1000).toFixed(1);
    if (t - lastLogAt >= 1000) {
      lastLogAt = t;
      const r = lastResult;
      log.push([new Date().toISOString(), elapsed, r.raw, r.visible, r.hidden, r.out, r.total]);
      $('recBtn').textContent = `Stop log (${log.length} s)`;
    }
    if (t - lastPathLogAt >= 100 && pathLog.length < PATH_LOG_MAX) {
      lastPathLogAt = t;
      for (const tr of tracker.tracks) {
        if (!tr.id || tr.state === 'out') continue;
        pathLog.push([elapsed, tr.id, tr.state, (tr.x / W).toFixed(4), (tr.y / H).toFixed(4)]);
      }
    }
  }
}
function updateReadout() {
  const r = lastResult;
  if (r.warming) {
    ui.countTotal.textContent = ui.countVisible.textContent = ui.countHidden.textContent =
      ui.countOut.textContent = ui.countNow.textContent = '…';
    setStatus(`Learning the background (${Math.min(frameCount, WARMUP_FRAMES)}/${WARMUP_FRAMES}). Keep the camera still.`);
    return;
  }
  ui.countTotal.textContent = r.total;
  ui.countVisible.textContent = r.visible;
  ui.countHidden.textContent = r.hidden;
  ui.countOut.textContent = r.out;
  ui.countNow.textContent = r.raw;
  if (performance.now() < globalChangeUntil) {
    setStatus('Large change detected (camera moved or lighting changed). Re-learning the background…', true);
  } else if (paused) {
    setStatus('Paused.');
  } else {
    const cand = r.candidates ? ` · ${r.candidates} unconfirmed detection${r.candidates > 1 ? 's' : ''}` : '';
    setStatus(`Moving area: ${(r.fg * 100).toFixed(1)} % of the analysed region${cand}.`);
  }
}

function setStatus(text, warn) {
  ui.status.textContent = text;
  ui.status.classList.toggle('warn', !!warn);
}

// ---------- sources ----------
function stopStream() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
}

async function startCamera(deviceId) {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('Camera access needs HTTPS and a browser that supports it (Safari or Chrome).', true);
    return;
  }
  stopStream();
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  const constraints = {
    audio: false,
    video: deviceId ? Object.assign({ deviceId: { exact: deviceId } }, size)
                    : Object.assign({ facingMode: { ideal: 'environment' } }, size),
  };
  try {
    setStatus('Requesting camera…');
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    setStatus(`Could not open the camera: ${err.name === 'NotAllowedError' ? 'permission denied. Allow camera access in the browser settings and reload.' : err.message || err.name}`, true);
    return;
  }
  video.removeAttribute('src');
  video.loop = false;
  video.srcObject = stream;
  await startVideo();
  await listCameras(stream.getVideoTracks()[0]);
}

async function listCameras(track) {
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    if (devices.length < 2) { ui.cameraSelect.hidden = true; return; }
    const current = track && track.getSettings ? track.getSettings().deviceId : '';
    ui.cameraSelect.innerHTML = '';
    devices.forEach((d, i) => {
      const o = document.createElement('option');
      o.value = d.deviceId;
      o.textContent = d.label || `Camera ${i + 1}`;
      if (d.deviceId === current) o.selected = true;
      ui.cameraSelect.appendChild(o);
    });
    ui.cameraSelect.hidden = false;
  } catch (_) { ui.cameraSelect.hidden = true; }
}

function startFile(file) {
  stopStream();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = true;
  startVideo();
}

async function startVideo() {
  try { await video.play(); } catch (_) { /* autoplay may need a gesture; the button is one */ }
  ui.placeholder.hidden = true;
  ui.stage.classList.add('ready');
  ui.buttons.forEach((b) => { b.disabled = false; });
  W = 0;
  lastVideoTime = -1;
  paused = false;
  $('pauseBtn').textContent = 'Pause';
  if (!running) { running = true; requestAnimationFrame(loop); }
  requestWakeLock();
}

async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator && !wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    }
  } catch (_) { /* not supported or refused */ }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && running) requestWakeLock();
});

// ---------- region of interest ----------
function normRect(r) {
  return {
    x0: Math.min(r.x0, r.x1), x1: Math.max(r.x0, r.x1),
    y0: Math.min(r.y0, r.y1), y1: Math.max(r.y0, r.y1),
  };
}
function pointerPos(e) {
  const rect = view.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
  };
}
function setRoiMode(on) {
  roiMode = on;
  ui.roiHint.hidden = !on;
  $('roiBtn').classList.toggle('active', on);
}
view.addEventListener('pointerdown', (e) => {
  if (!roiMode) return;
  view.setPointerCapture(e.pointerId);
  const p = pointerPos(e);
  roiDrawing = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  e.preventDefault();
});
view.addEventListener('pointermove', (e) => {
  if (!roiDrawing) return;
  const p = pointerPos(e);
  roiDrawing.x1 = p.x; roiDrawing.y1 = p.y;
  if (paused) draw();
});
view.addEventListener('pointerup', () => {
  if (!roiDrawing) return;
  const r = normRect(roiDrawing);
  roiDrawing = null;
  if (r.x1 - r.x0 > 0.03 && r.y1 - r.y0 > 0.03) {
    roi = r;
    resetTracking();
  }
  setRoiMode(false);
  if (paused) draw();
});

// ---------- logging ----------
function toggleLog() {
  const btn = $('recBtn');
  if (logging) {
    logging = false;
    btn.textContent = `Start new log (${log.length} rows kept)`;
    $('csvBtn').disabled = log.length === 0;
    $('pathsBtn').disabled = pathLog.length === 0;
    return;
  }
  log = [];
  pathLog = [];
  logging = true;
  logStart = performance.now();
  lastLogAt = 0;
  lastPathLogAt = 0;
  btn.textContent = 'Stop log (0 s)';
  $('csvBtn').disabled = true;
  $('pathsBtn').disabled = true;
}

function saveCsv(name, header, rows) {
  const body = rows.map((r) => r.join(',')).join('\n');
  const blob = new Blob([header + '\n' + body + '\n'], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${name}-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
function downloadCsv() {
  if (log.length) saveCsv('fish-counts', 'timestamp,elapsed_s,raw_groups,visible,hidden,out_of_view,total_fish', log);
}
function downloadPaths() {
  // x, y are fractions of the analysed frame (0..1, origin top-left)
  if (pathLog.length) saveCsv('fish-paths', 'elapsed_s,fish_id,state,x,y', pathLog);
}

// ---------- settings UI ----------
function formatSetting(key, v) {
  if (key === 'learn') return Number(v).toFixed(3);
  if (key === 'minArea') return `${v} px`;
  if (key === 'procWidth') return `${v} px`;
  if (key === 'merge') return `${v} px`;
  if (key === 'confirmTime') return `${Number(v).toFixed(1)} s`;
  if (key === 'hiddenMemory' || key === 'outMemory' || key === 'trailSeconds') return `${v} s`;
  return String(v);
}
function syncSettingsUI() {
  document.querySelectorAll('[data-key]').forEach((el) => {
    const key = el.dataset.key;
    if (el.type === 'checkbox') el.checked = !!settings[key];
    else el.value = settings[key];
    const out = document.querySelector(`output[data-for="${key}"]`);
    if (out) out.textContent = formatSetting(key, settings[key]);
  });
}
document.querySelectorAll('[data-key]').forEach((el) => {
  el.addEventListener('input', () => {
    const key = el.dataset.key;
    settings[key] = el.type === 'checkbox' ? el.checked : Number(el.value);
    const out = document.querySelector(`output[data-for="${key}"]`);
    if (out) out.textContent = formatSetting(key, settings[key]);
    if (TRACKER_KEYS.includes(key)) tracker.setOptions(trackerOptions());
    saveSettings();
    if (paused && W) draw();
  });
});
$('defaultsBtn').addEventListener('click', () => {
  Object.assign(settings, DEFAULTS);
  tracker.setOptions(trackerOptions());
  saveSettings();
  syncSettingsUI();
});

// ---------- buttons ----------
$('startBtn').addEventListener('click', () => startCamera());
$('fileInput').addEventListener('change', (e) => { if (e.target.files[0]) startFile(e.target.files[0]); });
ui.cameraSelect.addEventListener('change', () => startCamera(ui.cameraSelect.value));
$('pauseBtn').addEventListener('click', () => {
  paused = !paused;
  $('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
  if (video.src && !video.srcObject) { if (paused) video.pause(); else video.play(); }
  if (paused) setStatus('Paused.');
});
$('resetBtn').addEventListener('click', () => { resetBackground(); resetTracking(); globalChangeUntil = 0; });
$('roiBtn').addEventListener('click', () => setRoiMode(!roiMode));
$('roiClearBtn').addEventListener('click', () => { roi = null; resetTracking(); if (paused) draw(); });
$('recBtn').addEventListener('click', toggleLog);
$('csvBtn').addEventListener('click', downloadCsv);
$('pathsBtn').addEventListener('click', downloadPaths);

syncSettingsUI();
if (!window.isSecureContext) {
  setStatus('This page must be opened over HTTPS for camera access.', true);
}
