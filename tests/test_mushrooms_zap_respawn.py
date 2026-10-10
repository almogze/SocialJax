"""Mushrooms: off-grid zaps hit nothing (2026-10-10).

A zap's four targets (one ahead, two ahead, ahead-right, ahead-left) were not bounds-checked. JAX gathers clamp an
index past the end and wrap a negative one, so a zap facing out of the bottom or right edge read the zapper's own
cell and respawned the zapper, and one facing out of the top or left edge hit agents on the far side of the map.

The edge-zap tests fail on a36759f. The in-bounds zap tests pass there too: that behaviour is unchanged.

Run: PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/test_mushrooms_zap_respawn.py
"""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import socialjax
from socialjax.environments.mushrooms.mushrooms import STEP, Actions, Items

N = 6
NI = len(Items)  # agent i is grid value NI + i
Z, S = int(Actions.zap_forward), int(Actions.stay)
FAR = [(3, 14, 0), (7, 16, 0), (9, 18, 0)]  # bystanders out of reach of every zap below

_ENVS = {}


def _env(**kw):
    """(env, jitted step_env), cached per kwargs."""
    k = tuple(sorted(kw.items()))
    if k not in _ENVS:
        env = socialjax.make("mushrooms", num_agents=N, shared_rewards=False, **kw)
        _ENVS[k] = (env, jax.jit(env.step_env))
    return _ENVS[k]


def _state(env, locs, mushrooms=(), fill=None, keep_empty=(), reborn=None, matches=None):
    """A reset state with every mushroom removed, agents at locs [(row, col, heading)] and the given mushrooms
    [(row, col, item)]. fill: put this item on every spawn cell not in keep_empty and not under an agent.
    reborn: per-agent respawn cells scheduled by the previous step (default: none). Labels match the grid."""
    _, st = env.reset(jax.random.PRNGKey(0))
    g = onp.asarray(st.grid).copy()
    g[g >= Items.red_mushrooms] = Items.empty  # mushrooms and agents
    pel = onp.asarray(st.potential_empty_locs)
    if fill is not None:
        g[pel[:, 0], pel[:, 1]] = fill
        for r, c in keep_empty:
            g[r, c] = Items.empty
    for r, c, it in mushrooms:
        g[r, c] = it
    locs = onp.asarray(locs, dtype=onp.int16)
    for i, (r, c, _) in enumerate(locs):
        g[r, c] = NI + i
    st = st.replace(
        grid=jnp.asarray(g, dtype=st.grid.dtype),
        agent_locs=jnp.asarray(locs),
        reborn_locs=jnp.asarray(locs if reborn is None else reborn, dtype=jnp.int16),
        potential_empty_labels=jnp.asarray(g[pel[:, 0], pel[:, 1]], dtype=st.potential_empty_labels.dtype),
    )
    if matches is not None:
        st = st.replace(mushrooms_matches=jnp.asarray(matches, dtype=bool))
    return st


def _step(step, st, acts, key):
    return step(jax.random.PRNGKey(key), st, [jnp.array(a) for a in acts])


def _scheduled(st):
    """Agents the step scheduled for respawn (their reborn_loc differs from their cell)."""
    return onp.any(onp.asarray(st.reborn_locs) != onp.asarray(st.agent_locs), axis=-1)


def _spawn_cells(env):
    return set(map(tuple, onp.asarray(env.SPAWNS_PLAYERS).tolist()))


# heading: (zapper, two agents where an unchecked target clamps or wraps to, or right beside the zapper)
EDGE_ZAPS = {
    0: [(11, 5, 0), (11, 6, 0), (0, 5, 0)],  # +row out of the bottom: clamped onto the zapper itself
    1: [(6, 22, 1), (5, 22, 0), (6, 0, 0)],  # +col out of the right edge: clamped onto the zapper itself
    2: [(0, 5, 2), (11, 5, 0), (10, 5, 0)],  # -row out of the top: wrapped to rows 11 and 10
    3: [(5, 0, 3), (5, 22, 0), (5, 21, 0)],  # -col out of the left edge: wrapped to cols 22 and 21
}


@pytest.mark.parametrize("heading", sorted(EDGE_ZAPS))
def test_zap_off_the_grid_hits_nothing(heading):
    env, step = _env()
    locs = EDGE_ZAPS[heading] + FAR
    st = _state(env, locs)
    _, st1, _, _, _ = _step(step, st, [Z] + [S] * (N - 1), key=1)
    assert not _scheduled(st1).any(), f"heading {heading}: respawn scheduled for {onp.flatnonzero(_scheduled(st1))}"
    assert int((onp.asarray(st1.grid) == Items.interact).sum()) == 0, "an off-grid target was marked"
    _, st2, _, _, _ = _step(step, st1, [S] * N, key=2)
    assert onp.array_equal(onp.asarray(st2.agent_locs), onp.asarray(locs, dtype=onp.int16)), "an agent moved"


def _targets(p, h):
    p, step = onp.asarray(p[:2]), onp.asarray(STEP)[:, :2]
    return [p + step[h], p + 2 * step[h], p + step[h] + step[(h + 1) % 4], p + step[h] + step[(h - 1) % 4]]


@pytest.mark.parametrize("heading", range(4))
def test_zap_in_bounds_respawns_every_target(heading):
    env, step = _env()
    zapper = (5, 10, heading)
    behind = tuple(onp.asarray(zapper[:2]) - onp.asarray(STEP)[heading, :2]) + (0,)
    locs = [zapper] + [tuple(t) + (0,) for t in _targets(zapper, heading)] + [behind]
    st = _state(env, locs)
    _, st1, _, _, _ = _step(step, st, [Z] + [S] * (N - 1), key=3)
    expected = onp.array([False, True, True, True, True, False])
    assert onp.array_equal(_scheduled(st1), expected)
    _, st2, _, _, _ = _step(step, st1, [S] * N, key=4)
    locs2 = onp.asarray(st2.agent_locs)
    assert onp.array_equal(locs2, onp.asarray(st1.reborn_locs)), "respawn not applied"
    assert len({tuple(l[:2]) for l in locs2.tolist()}) == N
    assert all(tuple(l) in _spawn_cells(env) for l in locs2[expected, :2].tolist())


def test_zap_in_bounds_marks_empty_targets():
    env, step = _env()
    zapper = (5, 10, 0)
    st = _state(env, [zapper, (1, 1, 0), (1, 20, 0)] + FAR)
    _, st1, _, _, _ = _step(step, st, [Z] + [S] * (N - 1), key=5)
    marks = sorted(map(tuple, onp.argwhere(onp.asarray(st1.grid) == Items.interact).tolist()))
    assert marks == sorted(tuple(t.tolist()) for t in _targets(zapper, 0))
    assert not _scheduled(st1).any()


def test_zap_at_the_edge_still_hits_the_in_bounds_target():
    # Facing the bottom edge from row 10: one ahead (row 11) is on the grid, two ahead is not.
    env, step = _env()
    st = _state(env, [(10, 5, 0), (11, 5, 0), (11, 6, 0)] + FAR)
    _, st1, _, _, _ = _step(step, st, [Z] + [S] * (N - 1), key=6)
    assert onp.array_equal(_scheduled(st1), [False, True, True, False, False, False])
