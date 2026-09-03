"""Pure-math human input trajectories for the browser service.

Behavioral bot-detectors (reCAPTCHA v3, DataDome, HUMAN/PerimeterX, Akamai)
do not just fingerprint the browser — they score HOW the pointer moves and how
keys are struck. A straight, constant-time cursor jump from A to B is one of the
loudest tells there is. These helpers generate human-plausible motion:

  * `mouse_path` — a curved (cubic Bézier) trajectory with an eased velocity
    profile (slow → fast → slow, i.e. minimum-jerk-like), per-point jitter, and
    an optional slight overshoot-and-correct near the target.
  * `scroll_ticks` — a long scroll broken into several wheel increments of
    human-plausible size, instead of one teleporting jump.
  * `type_delays` — per-keystroke delays with word-boundary hesitations.

Everything here is deterministic given an `rng`, dependency-free (stdlib only),
and free of any Playwright import, so it ships into the browser container AND is
unit-testable in the plain harness environment. The async I/O (actually moving
the mouse, sleeping between samples) lives in server.py and consumes these.
"""
import math
import random
from typing import List, Tuple

Point = Tuple[float, float]

# Bounds so motion stays human-plausible AND snappy (a real move is ~100-400ms,
# not multiple seconds). Step counts scale with distance between these limits.
_MIN_STEPS = 6
_MAX_STEPS = 32


def ease_in_out(t: float) -> float:
    """Perlin smootherstep: 6t^5 - 15t^4 + 10t^3. Zero first AND second
    derivative at t=0 and t=1, so sampling uniform i/N through it clusters points
    at the ends — producing the slow-accelerate-decelerate velocity a hand makes,
    not the constant velocity a script makes."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return t * t * t * (t * (t * 6 - 15) + 10)


def _cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1.0 - t
    x = (u * u * u * p0[0] + 3 * u * u * t * p1[0]
         + 3 * u * t * t * p2[0] + t * t * t * p3[0])
    y = (u * u * u * p0[1] + 3 * u * u * t * p1[1]
         + 3 * u * t * t * p2[1] + t * t * t * p3[1])
    return (x, y)


def mouse_path(start: Point, end: Point, *, steps: int = None,
               jitter: float = 1.0, overshoot: bool = True,
               rng: random.Random = random) -> List[Point]:
    """A human-plausible cursor trajectory from `start` to `end`.

    - Curved: two Bézier control points are pushed off the straight line
      (perpendicular offset scaled by distance), so the path arcs like a wrist
      pivot rather than tracking a ruler.
    - Eased velocity: the Bézier parameter is passed through `ease_in_out`, so
      consecutive points are close near the ends and far apart in the middle.
    - Jittered: interior points get sub-pixel noise (a hand is never exact).
    - The FINAL point is exactly `end` so clicks/drags still land precisely.
    - Optional slight overshoot-and-correct just before arrival, a strong and
      safe (still-lands-on-target) human signal for longer moves.
    """
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return [(x1, y1)]

    if steps is None:
        steps = int(dist / 12)
    steps = max(_MIN_STEPS, min(_MAX_STEPS, steps))

    # Perpendicular unit vector for the arc offset.
    px, py = -dy / dist, dx / dist
    arc = rng.uniform(-0.16, 0.16) * dist  # signed arc height, proportional to distance
    c1 = (x0 + dx * 0.30 + px * arc * rng.uniform(0.4, 1.0),
          y0 + dy * 0.30 + py * arc * rng.uniform(0.4, 1.0))
    c2 = (x0 + dx * 0.70 + px * arc * rng.uniform(0.4, 1.0),
          y0 + dy * 0.70 + py * arc * rng.uniform(0.4, 1.0))

    pts: List[Point] = []
    for i in range(1, steps + 1):
        t = ease_in_out(i / steps)
        x, y = _cubic_bezier((x0, y0), c1, c2, (x1, y1), t)
        if i < steps:
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        pts.append((x, y))

    # Overshoot-and-correct: for a move long enough to warrant it, replace the
    # final point with a small pass PAST the target, then return exactly to it.
    if overshoot and dist > 120 and steps >= 10:
        ox = x1 + (dx / dist) * rng.uniform(2.0, 6.0)
        oy = y1 + (dy / dist) * rng.uniform(2.0, 6.0)
        pts[-1] = (ox, oy)
        pts.append((x1, y1))
    else:
        pts[-1] = (x1, y1)
    return pts


def scroll_ticks(total: int, *, rng: random.Random = random) -> List[int]:
    """Split a scroll of `total` pixels into several wheel increments whose
    magnitudes sum to `total`. A human rolls a wheel in notches (~90-170px each),
    never one exact teleport. Sign of `total` is preserved; small scrolls stay a
    single tick."""
    total = int(total)
    mag = abs(total)
    if mag == 0:
        return []
    sign = 1 if total > 0 else -1
    if mag <= 170:
        return [total]
    ticks: List[int] = []
    remaining = mag
    while remaining > 0:
        step = min(remaining, int(rng.uniform(90, 170)))
        # Avoid leaving a tiny final crumb; fold it into this tick.
        if 0 < remaining - step < 40:
            step = remaining
        ticks.append(sign * step)
        remaining -= step
    return ticks


def idle_drift_target(cursor: Point, width: int = 1920, height: int = 1080,
                      *, rng: random.Random = random) -> Point:
    """A small nearby point for an idle 'reading' cursor wander (30–140px from the
    current position, at a random angle), clamped to stay on-screen with a margin.
    Real people nudge the pointer while reading; a cursor that only ever teleports
    on click is a behavioral tell. The caller moves there via the normal curved
    path, so the motion itself stays human."""
    x, y = cursor
    ang = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(30, 140)
    nx = min(width - 8, max(8.0, x + math.cos(ang) * dist))
    ny = min(height - 8, max(8.0, y + math.sin(ang) * dist))
    return (nx, ny)


def type_delays(text: str, *, rng: random.Random = random) -> List[float]:
    """Per-character keystroke delays (seconds), with brief hesitations at word
    boundaries — the small pauses a person makes between words.

    The base 60–170ms/char range averages ~110ms — roughly a FAST human typist
    (~100 WPM). The previous 20–90ms averaged ~218 WPM, which is faster than the
    human sustained-typing world record and thus a bot tell in the wrong
    direction. Occasional longer hesitations (thinking / a less-common key) add
    the heavy-tailed variance real typing has. Returned in seconds."""
    delays: List[float] = []
    for ch in text:
        d = rng.uniform(0.06, 0.17)
        if ch == " " and rng.random() < 0.30:
            d += rng.uniform(0.06, 0.22)        # inter-word "thinking" pause
        elif rng.random() < 0.05:
            d += rng.uniform(0.15, 0.45)        # occasional hesitation (heavy tail)
        delays.append(d)
    return delays
