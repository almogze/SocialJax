"""Mushrooms: off-grid zaps hit nothing, respawns never delete a mushroom, info['reborn_players'] (2026-10-10).

A zap's four targets (one ahead, two ahead, ahead-right, ahead-left) were not bounds-checked. JAX gathers clamp an
index past the end and wrap a negative one, so a zap facing out of the bottom or right edge read the zapper's own
cell and respawned the zapper, and one facing out of the top or left edge hit agents on the far side of the map.

Spawn cells are the cells where mushrooms regrow. A reborn agent could be placed on a mushroom (deleting it with no
eat event), and a mushroom regrown at the start of a step on that step's respawn cell was deleted the same way.

The edge-zap, respawn, regrowth and info tests fail on a36759f. The in-bounds zap tests pass there too: who an
in-bounds zap hits is unchanged. Its interact mark can differ: before, a non-zapping agent facing off the top or
left edge wrote the old cell value back on the far side of the map, which could erase an in-bounds zapper's mark
there. That write is now dropped.

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
MUSHROOMS = (Items.red_mushrooms, Items.green_mushrooms, Items.blue_mushrooms, Items.orange_mushrooms)

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


@pytest.mark.parametrize("item", [Items.blue_mushrooms, Items.orange_mushrooms])
def test_respawn_never_lands_on_a_mushroom(item):
    # Find where agent 1 respawns after being zapped under a fixed key, then put a mushroom on that cell and zap
    # again under the same key: the respawn must pick another cell, and the mushroom must survive the landing.
    env, step = _env()
    locs = [(5, 5, 0), (6, 5, 0), (1, 1, 0)] + FAR
    _, st1, _, _, _ = _step(step, _state(env, locs), [Z] + [S] * (N - 1), key=7)
    cell = tuple(onp.asarray(st1.reborn_locs)[1, :2].tolist())
    assert _scheduled(st1)[1] and cell in _spawn_cells(env)

    st = _state(env, locs, mushrooms=[cell + (int(item),)])
    _, st1, _, _, _ = _step(step, st, [Z] + [S] * (N - 1), key=7)
    new_cell = tuple(onp.asarray(st1.reborn_locs)[1, :2].tolist())
    assert new_cell != cell, "respawn chose the mushroom's cell"
    assert new_cell in _spawn_cells(env)
    _, st2, _, _, _ = _step(step, st1, [S] * N, key=8)
    g = onp.asarray(st2.grid)
    assert g[cell] == item, "the mushroom was deleted"
    assert int((g == item).sum()) == 1
    assert tuple(onp.asarray(st2.agent_locs)[1, :2].tolist()) == new_cell


def test_respawn_avoids_mushrooms_in_a_random_rollout():
    env, step = _env()
    # On a36759f this rollout schedules two respawns onto a mushroom, the first at step 228.
    rng = onp.random.default_rng(1)
    _, st = env.reset(jax.random.PRNGKey(1))
    hits = 0
    for t in range(500):
        acts = onp.where(rng.random(N) < 0.3, Z, rng.integers(0, 7, size=N))
        _, st, _, _, _ = _step(step, st, acts.tolist(), key=100 + t)
        reborn = _scheduled(st)
        g = onp.asarray(st.grid)
        for r, c in onp.asarray(st.reborn_locs)[reborn, :2].tolist():
            assert g[r, c] not in MUSHROOMS, f"step {t}: respawn scheduled onto a mushroom at {(r, c)}"
        hits += int(reborn.sum())
    assert hits > 10, "too few zap hits: the test checked little"


def test_no_regrowth_on_the_cell_an_agent_respawns_on():
    # Agent 1 was zapped last step and lands on `cell` at the start of this one. Only six spawn cells are empty
    # (orange everywhere else) and red regrows on the five best-ranked ones with certainty. Before the fix `cell`
    # competed for those five slots and its red was then deleted by the landing agent.
    env, step = _env(regrow_rate_red=1.0)
    locs = [(5, 5, 0), (6, 5, 0), (1, 1, 0)] + FAR
    empty = [(0, 3), (2, 7), (4, 12), (9, 2), (10, 20), (11, 11)]
    cell = empty[3]
    reborn = onp.asarray(locs, dtype=onp.int16)
    reborn[1] = cell + (1,)
    matches = onp.zeros((N, 4), dtype=bool)
    matches[:3, 0] = True  # three reds eaten last step: 6 red regrowth trials (capped at 5)
    st = _state(env, locs, fill=int(Items.orange_mushrooms), keep_empty=empty, reborn=reborn, matches=matches)
    for key in range(6):
        _, st1, _, _, _ = _step(step, st, [S] * N, key=key)
        g = onp.asarray(st1.grid)
        assert g[cell] == NI + 1, "agent 1 did not land on its respawn cell"
        reds = sorted(map(tuple, onp.argwhere(g == Items.red_mushrooms).tolist()))
        assert reds == sorted(set(empty) - {cell}), f"key {key}: reds {reds}"
        assert int((g == Items.orange_mushrooms).sum()) == int((onp.asarray(st.grid) == Items.orange_mushrooms).sum())


def test_info_reborn_players_is_the_zap_hit_mask():
    # What marp's SocialJaxWrapper.get_agent_tagged() reads; without it Mushrooms logged peace = 1.0 always.
    env, step = _env()
    zapper = (5, 10, 0)
    locs = [zapper] + [tuple(t) + (0,) for t in _targets(zapper, 0)[:2]] + FAR
    _, st1, _, _, info = _step(step, _state(env, locs), [Z] + [S] * (N - 1), key=9)
    r = onp.asarray(info["reborn_players"])
    assert r.shape == (N,) and r.dtype == bool
    assert onp.array_equal(r, [False, True, True, False, False, False])
    _, _, _, _, info = _step(step, st1, [S] * N, key=10)
    assert not onp.asarray(info["reborn_players"]).any()
    _, _, _, _, info = _step(step, _state(env, EDGE_ZAPS[2] + FAR), [Z] + [S] * (N - 1), key=11)
    assert not onp.asarray(info["reborn_players"]).any()


def test_info_reborn_players_with_shared_rewards():
    env = socialjax.make("mushrooms", num_agents=N, shared_rewards=True)
    _, st = env.reset(jax.random.PRNGKey(0))
    _, _, _, _, info = env.step_env(jax.random.PRNGKey(1), st, [jnp.array(S)] * N)
    r = onp.asarray(info["reborn_players"])
    assert r.shape == (N,) and r.dtype == bool and not r.any()


@pytest.mark.parametrize("zapper", [(0, 5, 2), (1, 5, 2), (5, 0, 3), (5, 1, 3), (11, 5, 0), (6, 22, 1)])
def test_zap_off_the_grid_leaves_no_far_side_mark(zapper):
    # A lone zapper facing out of an edge: only its on-grid targets are marked. Unchecked, the top/left cases wrote
    # interact marks on the far side of the map (rows 10-11 or cols 21-22). The edge tests above cannot see this:
    # their agents stand on the wrapped cells, where a wrapped write just puts the agent's id back.
    env, step = _env()
    R, C = env.GRID_SIZE_ROW, env.GRID_SIZE_COL
    st = _state(env, [zapper] + FAR + [(4, 18, 0), (2, 10, 0)])
    _, st1, _, _, _ = _step(step, st, [Z] + [S] * (N - 1), key=12)
    marks = set(map(tuple, onp.argwhere(onp.asarray(st1.grid) == Items.interact).tolist()))
    p, h = onp.asarray(zapper[:2]), zapper[2]
    step_ = onp.asarray(STEP)[:, :2]
    on = lambda t: 0 <= t[0] < R and 0 <= t[1] < C
    t1 = p + step_[h]
    ts = [t1, p + 2 * step_[h]] + [t if on(t) else t1 for t in (t1 + step_[(h + 1) % 4], t1 + step_[(h - 1) % 4])]
    assert marks == {tuple(int(x) for x in t) for t in ts if on(t)}, sorted(marks)


@pytest.mark.parametrize("locs, hit", [
    ([(1, 5, 2), (0, 5, 0), (11, 5, 0)], [False, True, False]),   # two ahead (-1, 5) wrapped to row 11
    ([(5, 1, 3), (5, 0, 0), (5, 22, 0)], [False, True, False]),   # two ahead (5, -1) wrapped to col 22
])
def test_zap_two_ahead_off_the_grid_hits_only_the_on_grid_target(locs, hit):
    env, step = _env()
    _, st1, _, _, _ = _step(step, _state(env, locs + FAR), [Z] + [S] * (N - 1), key=13)
    assert onp.array_equal(_scheduled(st1), hit + [False] * 3)
