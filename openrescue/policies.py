"""
OpenRescue — Policies
=====================

Baselines and the proposed **Resilient** policy. Every policy shares the same
sensors and the same discrete action space; they differ only in the *decision
layer* — this is what makes the benchmark an ablation:

    Random        — uniform random actions. No map, no comm, no R-index.
    Frontier      — greedy frontier exploration on the LOCAL map only. No
                    communication; degrades hard when GPS is denied because
                    belief and true position diverge.
    IG            — Information-Gain / Next-Best-View scanning: identical to
                    Resilient (same R-gating, hysteresis, cluster and relay
                    modes) except that Explore mode targets the unknown cell
                    maximizing local map-entropy reduction per unit movement
                    cost instead of the nearest unknown cell. A pure ablation
                    of the explore target rule.
    Resilient     — behavior switching gated on the onboard Resilience Index,
                    with a hysteresis dead-band on the mode boundaries so the
                    policy does not thrash when R jitters across a threshold:

                        R >= 0.80          -> pi_Explore-Aggressive
                        0.70 < R < 0.80   -> retain current mode (dead-band)
                        0.35 <= R <= 0.70 -> pi_Coordinated-Cluster
                        0.30 < R < 0.35   -> retain current mode (dead-band)
                        R < 0.30          -> pi_Relay-Mesh / Return

                    Per-agent mode state persists across steps (an agent must
                    cross the full dead-band before switching) and resets on
                    episode reset. The retain path falls back safely when its
                    target no longer exists (e.g. a relay neighbor is gone:
                    nearest-neighbor search -> last trusted base bearing ->
                    hold). The resilient policy fuses neighbor maps received
                    over the (lossy) comm link, so it keeps navigating from
                    shared information even when its own GPS is denied.

All moving policies steer with ``_safe_step``: they never step into a cell
their (local/fused) map marks occupied — known walls are routed around, so
crashes can only come from *blind* contacts (unseen or corrupted cells), which
the environment injects only under sensor/comm/GPS failure.

Policy API: ``act(obs, info, rng) -> (K,) int actions``; ``info`` comes from
``OpenRescueEnv.step`` / ``reset``.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .resilience_index import MODE_CLUSTER, MODE_EXPLORE, MODE_RELAY

# stay, N, S, E, W
_DELTAS = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)


def _cell_of(pos: np.ndarray, grid: int):
    return (int(np.clip(round(pos[0]), 0, grid - 1)),
            int(np.clip(round(pos[1]), 0, grid - 1)))


def _safe_step(belief: np.ndarray, target: np.ndarray, occ: np.ndarray,
               rng: np.random.Generator) -> int:
    """
    Stepping toward ``target`` that never enters a known-occupied cell.

    occ: bool grid (True = occupied) in the agent's own belief frame.
    Returns the action index in {0..4}; 0 (stay) if every move is blocked.
    """
    g = occ.shape[0]
    candidates = []
    for a in range(1, 5):  # N, S, E, W
        nxt = belief + _DELTAS[a].astype(np.float32)
        gx, gy = _cell_of(nxt, g)
        if not occ[gy, gx]:
            candidates.append(a)
    if not candidates:
        return 0

    best = []
    best_cost = float('inf')
    for a in candidates:
        nxt = belief + _DELTAS[a].astype(np.float32)
        cost = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
        if cost < best_cost:
            best_cost, best = cost, [a]
        elif cost == best_cost:
            best.append(a)
    # small stochasticity among near-optimal moves avoids limit cycles
    if len(best) == 0:
        return 0
    return int(best[int(rng.integers(0, len(best)))])


def _nearest_unknown(belief: np.ndarray, m: np.ndarray, radius: int,
                     rng: np.random.Generator, jitter: float = 0.25):
    """Nearest unknown cell to belief within radius (tie-broken randomly)."""
    g = m.shape[0]
    bx, by = _cell_of(belief, g)
    best, best_d = None, float('inf')
    x0, x1 = max(0, bx - radius), min(g, bx + radius + 1)
    y0, y1 = max(0, by - radius), min(g, by + radius + 1)
    for gy in range(y0, y1):
        for gx in range(x0, x1):
            if m[gy, gx] == 0:  # unknown cell
                d = abs(gx - bx) + abs(gy - by)
                if d < best_d or (d == best_d and rng.random() < jitter):
                    best_d = d
                    best = np.array([gx + 0.5, gy + 0.5], dtype=np.float32)
    return best


class RandomPolicy:
    """Baseline 1: uniform random actions."""

    name = 'random'

    def act(self, obs: np.ndarray, info: dict, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 5, size=len(info['agents'])).astype(np.int32)


class FrontierPolicy:
    """Baseline 2: greedy frontier exploration on the LOCAL map only."""

    name = 'frontier'
    search_radius: int = 8

    def act(self, obs: np.ndarray, info: dict, rng: np.random.Generator) -> np.ndarray:
        actions = np.zeros(len(info['agents']), dtype=np.int32)
        for i, ag in enumerate(info['agents']):
            if ag['grounded']:
                actions[i] = 0
                continue
            m = ag['map']
            occ = m == 2
            belief = ag['belief']
            target = _nearest_unknown(belief, m, self.search_radius, rng)
            if target is None:
                actions[i] = rng.integers(0, 5)  # nothing nearby: wander
            else:
                actions[i] = _safe_step(belief, target, occ, rng)
        return actions


class ResilientPolicy:
    """
    The proposed policy: pi(s_i | R_{i,t}) with neighbor map fusion.

    Maintains a *fused* map per agent: its own local map merged with the maps
    of neighbors whose messages arrived this step (higher observation
    confidence wins per cell).

    Mode logic with hysteresis dead-bands (blueprint spec; thresholds
    configurable, defaults 0.80 / 0.30 with retain bands 0.10 / 0.05):
      * Explore  (R >= 0.80): aggressive frontier exploration on the fused map.
      * Cluster  (0.35 <= R <= 0.70): 60% frontier pull, 40% cohesion toward
        the centroid of received neighbor beliefs.
      * Relay    (R < 0.30): mesh maintenance — step toward the nearest
        reachable neighbor; if fully isolated, fall back to the last trusted
        base bearing (cached while GPS was reliable); else hold.
      * Dead-bands 0.70 < R < 0.80 and 0.30 < R < 0.35 retain the current mode
        (per-agent, persisted across steps, reset on episode reset), so an
        agent inside a dead-band keeps its previous behavior instead of
        thrashing between modes. Retaining a mode whose target vanished is
        safe: _cluster falls back to a frontier pull / random move when no
        neighbor message arrived, and _relay falls back to the base bearing
        then to hold when no reachable neighbor remains.
    """

    name = 'resilient'
    search_radius: int = 8

    def __init__(self, explore_threshold: float = 0.80, relay_threshold: float = 0.30,
                 explore_retain: float = 0.10, relay_retain: float = 0.05):
        """Hysteresis band edges: explore >= explore_threshold; the dead-band
        (explore_threshold - explore_retain, explore_threshold) retains the
        current mode, and likewise (relay_threshold, relay_threshold +
        relay_retain) below the cluster band."""
        self.explore_threshold = explore_threshold
        self.relay_threshold = relay_threshold
        self.explore_retain = explore_retain
        self.relay_retain = relay_retain
        # Blueprint defaults give: explore >= 0.80, cluster [0.35, 0.70],
        # relay < 0.30, retain bands (0.70, 0.80) and (0.30, 0.35).
        self.cluster_hi = explore_threshold - explore_retain
        self.cluster_lo = relay_threshold + relay_retain
        self.fused: Dict[int, np.ndarray] = {}
        self.fused_conf: Dict[int, np.ndarray] = {}
        self.modes: Dict[int, str] = {}      # per-agent persistent mode state

    def reset(self):
        self.fused = {}
        self.fused_conf = {}
        self.modes = {}

    def _select_mode(self, i: int, r: float) -> str:
        """pi(s_i | R_{i,t}) with hysteresis dead-bands (blueprint spec).

        Returns and stores the agent's new mode: explore above
        ``explore_threshold``, relay below ``relay_threshold``, cluster inside
        the [cluster_lo, cluster_hi] band; inside the dead-bands the previous
        mode is retained.

        Fresh episode (no mode history yet): there is no previous mode to
        retain, so the agent is anchored to the *committed band nearest its
        current R* — each dead-band is split at its midpoint (0.75 and 0.325
        at the blueprint defaults). This keeps a mid-failure agent from
        hard-defaulting to Explore (R ~ 0.71 at L3 would otherwise stay
        aggressive under GPS denial until R crossed 0.70).
        """
        if i not in self.modes:
            # no retain history: pick the committed band nearest current R
            if r >= self.explore_threshold:
                mode = MODE_EXPLORE
            elif r >= (self.explore_threshold + self.cluster_hi) / 2.0:
                mode = MODE_EXPLORE          # upper dead-band, nearer explore
            elif r >= self.cluster_lo:
                mode = MODE_CLUSTER
            elif r >= (self.cluster_lo + self.relay_threshold) / 2.0:
                mode = MODE_CLUSTER          # lower dead-band, nearer cluster
            else:
                mode = MODE_RELAY
            self.modes[i] = mode
            return mode

        cur = self.modes[i]
        if r >= self.explore_threshold:
            mode = MODE_EXPLORE
        elif r > self.cluster_hi:              # dead-band above cluster
            mode = cur
        elif r >= self.cluster_lo:             # cluster band
            mode = MODE_CLUSTER
        elif r > self.relay_threshold:         # dead-band below cluster
            mode = cur
        else:                                  # relay band
            mode = MODE_RELAY
        self.modes[i] = mode
        return mode

    def act(self, obs: np.ndarray, info: dict, rng: np.random.Generator) -> np.ndarray:
        agents = info['agents']
        n = len(agents)
        actions = np.zeros(n, dtype=np.int32)

        for i, ag in enumerate(agents):
            if i not in self.fused:
                self.fused[i] = np.zeros_like(ag['map'])
                self.fused_conf[i] = np.zeros_like(ag['map_conf'])
            self._fuse_into(i, ag, agents)

        for i, ag in enumerate(agents):
            if ag['grounded']:
                actions[i] = 0
                continue

            r = ag['r_index']
            fused = self.fused[i]
            occ = fused == 2
            belief = ag['belief']
            mode = self._select_mode(i, r)

            if mode == MODE_EXPLORE:
                target = self._explore_target(belief, fused, rng)
                if target is None:
                    actions[i] = rng.integers(0, 5)
                else:
                    actions[i] = _safe_step(belief, target, occ, rng)
            elif mode == MODE_CLUSTER:
                actions[i] = self._cluster(belief, fused, ag, agents, occ, rng)
            else:
                actions[i] = self._relay(belief, ag, agents, occ, rng)

        return actions

    def _explore_target(self, belief: np.ndarray, fused: np.ndarray,
                        rng: np.random.Generator):
        """Explore-mode target: nearest unknown cell on the fused map.

        Overridden by ``IgPolicy`` with an information-gain scorer; keeping it
        a method makes the IG policy a pure ablation of this one rule.
        """
        return _nearest_unknown(belief, fused, self.search_radius, rng)

    # ------------------------------------------------------------------
    def _fuse_into(self, i: int, ag: dict, agents: list):
        """Merge the agent's local map and received neighbor maps into the
        fused map, keeping the highest-confidence label per cell."""
        sources = [(ag['map'], ag['map_conf'])]
        for j in ag['neighbors_received']:
            n_ag = agents[j]
            sources.append((n_ag['map'], n_ag['map_conf']))
        m, c = self.fused[i], self.fused_conf[i]
        for src_map, src_conf in sources:
            better = c < src_conf
            c[better] = src_conf[better]
            m[better] = src_map[better]

    def _cluster(self, belief: np.ndarray, fused: np.ndarray, ag: dict,
                 agents: list, occ: np.ndarray, rng: np.random.Generator) -> int:
        """Coordinated-cluster: cohesion toward received neighbors, with a
        modest frontier pull when the neighborhood is already mapped."""
        nb = [agents[j] for j in ag['neighbors_received']]
        if nb:
            centroid = np.mean([a['belief'] for a in nb], axis=0)
            # prefer cohesion when few neighbors / low agreement
            if rng.random() < 0.6:
                return _safe_step(belief, centroid, occ, rng)
        # frontier pull within a short radius (modest, non-aggressive)
        target = _nearest_unknown(belief, fused, min(self.search_radius, 5), rng)
        if target is not None:
            return _safe_step(belief, target, occ, rng)
        if nb:
            return _safe_step(belief, np.mean([a['belief'] for a in nb], axis=0), occ, rng)
        return rng.integers(0, 5)

    def _relay(self, belief: np.ndarray, ag: dict, agents: list,
               occ: np.ndarray, rng: np.random.Generator) -> int:
        """Relay-mesh / return: nearest reachable neighbor first; if isolated,
        follow the last trusted base bearing; otherwise hold position."""
        pos = ag['pos']  # physical position used for mesh formation
        nearest, best_d = None, float('inf')
        for j, a in enumerate(agents):
            if j == ag['id'] or a['grounded']:
                continue
            d = float(np.linalg.norm(a['pos'] - pos))
            if d < best_d:
                best_d, nearest = d, a
        if nearest is not None:
            return _safe_step(belief, nearest['belief'], occ, rng)
        bearing = ag['last_base_bearing']
        if float(np.linalg.norm(bearing)) > 1e-6:
            return _safe_step(belief, belief + 4.0 * bearing, occ, rng)
        return 0  # hold position


class IgPolicy(ResilientPolicy):
    """Information-Gain / Next-Best-View scanning with the same R-gating.

    Inherits ResilientPolicy unchanged — hysteresis mode selection, neighbor
    map fusion, cluster cohesion, relay-mesh fallbacks — and overrides only
    the Explore-mode target rule: score every unknown cell ``c`` in the
    search radius by the number of unknown cells its sensing window would
    resolve per unit of travel cost,

        Score(c) = |unknown(window(c))| / (1 + ||c - belief||_1),

    and move toward the argmax. This is the next-best-view heuristic from
    informative path planning, made failure-gated: it is used only while
    R >= 0.80, and the policy otherwise consolidates exactly like
    ``ResilientPolicy``, so any measured difference is attributable to the
    scanning rule alone.
    """

    name = 'ig'
    ig_window: int = 3   # half-width of the local information-gain window

    def _explore_target(self, belief: np.ndarray, fused: np.ndarray,
                        rng: np.random.Generator):
        g = fused.shape[0]
        bx, by = _cell_of(belief, g)
        win = int(self.ig_window)
        best, best_score = None, -1.0
        x0, x1 = max(0, bx - self.search_radius), min(g, bx + self.search_radius + 1)
        y0, y1 = max(0, by - self.search_radius), min(g, by + self.search_radius + 1)
        for gy in range(y0, y1):
            for gx in range(x0, x1):
                if fused[gy, gx] != 0:
                    continue  # only unknown cells are candidate targets
                wx0, wx1 = max(0, gx - win), min(g, gx + win + 1)
                wy0, wy1 = max(0, gy - win), min(g, gy + win + 1)
                unknowns = int((fused[wy0:wy1, wx0:wx1] == 0).sum())
                d = abs(gx - bx) + abs(gy - by)
                score = unknowns / (1.0 + d)
                if score > best_score or (score == best_score and rng.random() < 0.25):
                    best_score = score
                    best = np.array([gx + 0.5, gy + 0.5], dtype=np.float32)
        return best


POLICIES = {
    'random': RandomPolicy,
    'frontier': FrontierPolicy,
    'resilient': ResilientPolicy,
    'ig': IgPolicy,
}


def make_policy(name: str, **kwargs):
    name = name.lower()
    if name not in POLICIES:
        raise ValueError(f"Unknown policy '{name}'. Choose from {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)