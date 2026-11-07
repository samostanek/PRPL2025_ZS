'use strict';

/*
 * Multi-object tracker for fish (detections = groups of moving pixels).
 *
 * Every track has a lifecycle:
 *
 *   candidate --(persists, moves)--> visible <--> hidden   (lost inside the view: occluded,
 *       |                               |            |      merged with another fish, stopped)
 *       |                               v            v
 *    discarded                         out  -------> forgotten after a memory timeout
 *  (appeared and vanished,        (left through the edge of the analysed area)
 *   or never moved)
 *
 * - Candidates are not counted. They become fish only after being seen in most frames for
 *   `confirmTime` seconds and travelling at least half a fish length, so flickers,
 *   reflections and noise that appear and disappear are discarded.
 * - Motion model: alpha-beta filter (position + velocity). Predicted positions are matched
 *   to detections by greedy nearest-neighbour inside a gate that widens while a fish is hidden.
 * - When a fish disappears away from the edges it stays counted as "hidden" at its last
 *   position for `hiddenMemory` seconds. When it disappears at an edge it is "out of view"
 *   and stays counted for `outMemory` seconds.
 * - When a new candidate is confirmed, it first tries to re-identify a hidden fish
 *   (reappearing near where it was lost) or, if it entered through an edge, an out-of-view
 *   fish (preferring the closest exit point). Only if neither fits is a new fish ID created.
 */
(function (root) {
  const DEFAULTS = {
    confirmTime: 0.5,       // s a candidate must persist before it counts as a fish
    confirmHitRatio: 0.6,   // fraction of frames it must have been detected in
    maxCandidateMiss: 3,    // consecutive missed frames before a candidate is discarded
    hiddenMemory: 20,       // s a fish lost inside the view is still counted
    outMemory: 120,         // s a fish that left the view is still counted
    trailSeconds: 10,       // s of path kept for drawing
    alpha: 0.6,             // position gain of the alpha-beta filter
    beta: 0.2,              // velocity gain
    hiddenVelocityDecay: 0.3, // s time constant: hidden fish drift to a stop
    directResume: 0.3,      // s: a fish hidden longer than this is only resumed through a
                            //    confirmed candidate, never by a single detection
  };

  class FishTracker {
    constructor(opts) {
      this.opts = Object.assign({}, DEFAULTS, opts);
      this.reset();
    }

    reset() {
      this.tracks = [];
      this.clock = 0;
      this.nextId = 1;
      this.nextCandidate = 1;
      this.stats = { created: 0, reidentified: 0, discarded: 0, forgotten: 0 };
      this.events = []; // {type, id, t} for the most recent update (for logging)
    }

    setOptions(opts) { Object.assign(this.opts, opts); }

    /**
     * @param {Array} dets  detections {cx, cy, minX, minY, maxX, maxY, area}
     * @param {number} dt   seconds since the previous frame
     * @param {object} region analysed area in pixels {x0, y0, x1, y1} (x1/y1 exclusive)
     * @param {number} fishSize typical fish length in pixels
     */
    update(dets, dt, region, fishSize) {
      const o = this.opts;
      dt = Math.min(0.25, Math.max(1 / 120, dt || 1 / 30));
      this.clock += dt;
      this.events = [];
      const now = this.clock;
      const diag = Math.hypot(region.x1 - region.x0, region.y1 - region.y0);
      const size = Math.max(3, fishSize || 8);
      const baseGate = Math.max(size * 1.5, 0.04 * diag);
      const margin = Math.max(2, Math.min(size * 0.5, 0.03 * diag));

      // 1. predict
      for (const t of this.tracks) {
        if (t.state === 'out') continue;
        if (t.state === 'hidden') {
          const k = Math.exp(-dt / o.hiddenVelocityDecay);
          t.vx *= k; t.vy *= k;
        }
        t.x += t.vx * dt;
        t.y += t.vy * dt;
        if (t.state === 'hidden') {
          t.x = Math.min(region.x1 - 1, Math.max(region.x0, t.x));
          t.y = Math.min(region.y1 - 1, Math.max(region.y0, t.y));
        }
        t.frames++;
      }

      // 2. associate (greedy on normalised distance inside each track's gate)
      const pairs = [];
      this.tracks.forEach((t, ti) => {
        if (t.state === 'out') return;
        // A fish lost for more than a moment must re-appear as a candidate and pass the
        // confirmation test (see confirm) before it gets its ID back, so a hidden fish
        // cannot latch onto a single flicker or a static reflection.
        if (t.state === 'hidden' && now - t.lastSeen > o.directResume) return;
        const gate = baseGate + Math.hypot(t.vx, t.vy) * dt * 2;
        let penalty = 0;
        if (t.state === 'hidden') {
          penalty = 0.15;
        } else if (t.state === 'candidate') {
          penalty = 0.3;
        }
        dets.forEach((d, di) => {
          const cd = Math.hypot(d.cx - t.x, d.cy - t.y);
          const bx = Math.max(d.minX - t.x, 0, t.x - d.maxX);
          const by = Math.max(d.minY - t.y, 0, t.y - d.maxY);
          const dist = Math.min(cd, Math.hypot(bx, by) + size * 0.5);
          if (dist <= gate) pairs.push({ ti, di, cost: dist / gate + penalty });
        });
      });
      pairs.sort((a, b) => a.cost - b.cost);
      const trackUsed = new Set(), detUsed = new Set();
      const matches = [];
      for (const p of pairs) {
        if (trackUsed.has(p.ti) || detUsed.has(p.di)) continue;
        trackUsed.add(p.ti); detUsed.add(p.di);
        matches.push(p);
      }

      // 3. update matched tracks
      for (const { ti, di } of matches) {
        const t = this.tracks[ti], d = dets[di];
        const reappeared = t.state === 'hidden';
        const rx = d.cx - t.x, ry = d.cy - t.y;
        t.x += o.alpha * rx; t.y += o.alpha * ry;
        if (!reappeared) {
          // velocity from the innovation, limited to one fish length per frame
          const lim = size / dt, k = o.beta / dt;
          t.vx += Math.max(-lim, Math.min(lim, k * rx));
          t.vy += Math.max(-lim, Math.min(lim, k * ry));
          const vmax = diag * 0.75; // px/s sanity cap
          const v = Math.hypot(t.vx, t.vy);
          if (v > vmax) { t.vx *= vmax / v; t.vy *= vmax / v; }
        }
        if (reappeared) { t.state = 'visible'; this.events.push({ type: 'reappeared', id: t.id, t: now, minor: true }); }
        t.hits++;
        t.misses = 0;
        t.streak++;
        t.lastSeen = now;
        if (t.state === 'candidate') t.obs.push({ x: d.cx, y: d.cy });
        t.box = { minX: d.minX, minY: d.minY, maxX: d.maxX, maxY: d.maxY };
        t.atEdge = touchesEdge(d, region, margin);
        t.path.push({ x: t.x, y: t.y, t: now, gap: reappeared });
        d.trackId = t.id;
      }

      // 4. tracks without a detection this frame
      const remove = new Set();
      this.tracks.forEach((t, ti) => {
        if (trackUsed.has(ti)) return;
        t.streak = 0;
        if (t.state === 'candidate') {
          t.misses++;
          if (t.misses > o.maxCandidateMiss) { remove.add(t); this.stats.discarded++; }
          return;
        }
        if (t.state === 'visible') {
          const outside = t.x < region.x0 || t.x >= region.x1 || t.y < region.y0 || t.y >= region.y1;
          if (t.atEdge || outside) this.markOut(t, now);
          else { t.state = 'hidden'; this.events.push({ type: 'hidden', id: t.id, t: now, minor: true }); }
          return;
        }
        if (t.state === 'hidden' && now - t.lastSeen > o.hiddenMemory) {
          remove.add(t); this.stats.forgotten++;
          this.events.push({ type: 'forgotten', id: t.id, t: now });
        } else if (t.state === 'out' && now - t.lastSeen > o.outMemory) {
          remove.add(t); this.stats.forgotten++;
          this.events.push({ type: 'forgotten', id: t.id, t: now });
        }
      });

      // 5. promote candidates that persisted and moved; discard stale ones
      for (const t of this.tracks) {
        if (t.state !== 'candidate' || remove.has(t)) continue;
        const age = now - t.born;
        t.maxTravel = netTravel(t.obs);
        // Confirm only a candidate that persisted, is being seen right now (3 frames in a
        // row) and really moved: net displacement between its first and last few
        // observations, so a single jump onto a nearby flicker is not "movement".
        if (age >= o.confirmTime && t.streak >= 3 && t.hits / t.frames >= o.confirmHitRatio &&
            t.maxTravel >= size * 0.5) {
          this.confirm(t, remove, diag, baseGate, now);
        } else if (age > o.confirmTime * 3) {
          remove.add(t); this.stats.discarded++;
        }
      }
      if (remove.size) this.tracks = this.tracks.filter((t) => !remove.has(t));

      // 6. new candidates from unmatched detections (but not from fragments of a known fish)
      dets.forEach((d, di) => {
        if (detUsed.has(di)) return;
        for (const t of this.tracks) {
          if (t.state === 'visible' && Math.hypot(d.cx - t.x, d.cy - t.y) < size) return;
        }
        this.tracks.push({
          id: 0,
          cid: this.nextCandidate++,
          state: 'candidate',
          x: d.cx, y: d.cy, vx: 0, vy: 0,
          x0: d.cx, y0: d.cy, maxTravel: 0,
          born: now, lastSeen: now, frames: 1, hits: 1, misses: 0, streak: 1,
          obs: [{ x: d.cx, y: d.cy }],
          box: { minX: d.minX, minY: d.minY, maxX: d.maxX, maxY: d.maxY },
          atEdge: touchesEdge(d, region, margin),
          enteredAtEdge: touchesEdge(d, region, margin),
          path: [{ x: d.cx, y: d.cy, t: now, gap: false }],
        });
      });

      // 7. trim drawn paths
      const keepAfter = now - o.trailSeconds;
      for (const t of this.tracks) {
        let i = 0;
        while (i < t.path.length - 1 && t.path[i].t < keepAfter) i++;
        if (i) t.path.splice(0, i);
      }

      return this.counts();
    }

    markOut(t, now) {
      t.state = 'out';
      t.exitX = t.x; t.exitY = t.y;
      t.vx = 0; t.vy = 0;
      this.events.push({ type: 'left view', id: t.id, t: now });
    }

    confirm(c, remove, diag, baseGate, now) {
      // Try to re-identify a fish we lost track of before creating a new one.
      let best = null, bestCost = Infinity;
      for (const t of this.tracks) {
        if (t === c || remove.has(t)) continue;
        let cost = Infinity;
        if (t.state === 'hidden') {
          const r = Math.min(0.35 * diag, baseGate * (1 + 1.5 * Math.min(4, now - t.lastSeen)));
          const d = Math.hypot(c.x0 - t.x, c.y0 - t.y);
          if (d <= r) cost = d / r;
        } else if (t.state === 'out' && c.enteredAtEdge) {
          cost = 0.5 + Math.hypot(c.x0 - t.exitX, c.y0 - t.exitY) / diag;
        }
        if (cost < bestCost) { bestCost = cost; best = t; }
      }
      if (best) {
        c.id = best.id;
        c.hue = best.hue;
        c.firstSeen = best.firstSeen;
        if (c.path.length) c.path[0].gap = true;
        c.path = best.path.concat(c.path);
        remove.add(best);
        this.stats.reidentified++;
        this.events.push({ type: best.state === 'out' ? 're-entered' : 'reappeared', id: c.id, t: now });
      } else {
        c.id = this.nextId++;
        c.hue = (c.id * 137.508) % 360;
        c.firstSeen = c.born;
        this.stats.created++;
        this.events.push({ type: 'new fish', id: c.id, t: now });
      }
      c.state = 'visible';
      c.obs = null;
    }

    /** Candidates that have sat still for a while: background ghosts, resting objects. */
    staticCandidates(fishSize) {
      const minAge = this.opts.confirmTime * 2;
      return this.tracks.filter((t) => t.state === 'candidate' && !t.misses &&
        this.clock - t.born >= minAge && t.maxTravel < Math.max(2, fishSize * 0.25));
    }

    counts() {
      let visible = 0, hidden = 0, out = 0, candidates = 0;
      for (const t of this.tracks) {
        if (t.state === 'visible') visible++;
        else if (t.state === 'hidden') hidden++;
        else if (t.state === 'out') out++;
        else candidates++;
      }
      return { visible, hidden, out, candidates, total: visible + hidden + out };
    }
  }

  // Distance between the mean of the first three and the last three observations.
  function netTravel(obs) {
    if (obs.length < 4) return 0;
    const n = Math.min(3, obs.length >> 1);
    let ax = 0, ay = 0, bx = 0, by = 0;
    for (let i = 0; i < n; i++) {
      ax += obs[i].x; ay += obs[i].y;
      bx += obs[obs.length - 1 - i].x; by += obs[obs.length - 1 - i].y;
    }
    return Math.hypot(bx - ax, by - ay) / n;
  }

  function touchesEdge(d, region, m) {
    return d.minX <= region.x0 + m || d.minY <= region.y0 + m ||
           d.maxX >= region.x1 - 1 - m || d.maxY >= region.y1 - 1 - m;
  }

  root.FishTracker = FishTracker;
  root.FishTracker.DEFAULTS = DEFAULTS;
  if (typeof module !== 'undefined' && module.exports) module.exports = { FishTracker };
})(typeof window !== 'undefined' ? window : globalThis);
