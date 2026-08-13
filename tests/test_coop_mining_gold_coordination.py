"""Regression tests for Coop Mining gold-coordination correctness.

These guard the fix for the never-reset ``ore_miners`` registry and the
previously-dead gold mining window. Before the fix, a single agent could
solo-farm a gold cell for the full reward after one initial pairing (because
the per-cell miner registry persisted across regrowth), and a lone partial
never expired (the countdown was decremented but never read). See the SocialJax
issue / commit that introduced these tests for the full diagnosis.

Runnable either with pytest (``pytest tests/test_coop_mining_gold_coordination.py``)
or directly (``python tests/test_coop_mining_gold_coordination.py``).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from socialjax.environments.coop_mining.coop_mining import CoopMining, Items

MINE, STAY = 7, 6          # mine/stay both leave position + orientation unchanged
WINDOW = 3


def _make_env():
    return CoopMining(num_agents=4, shared_rewards=False, gold_mining_window=WINDOW)


def _crafted_state(env, with_gold=True):
    """A clean empty grid with one gold cell flanked by two agents (a0, a2)
    that face it from opposite sides (both within mining range); a1, a3 parked."""
    R, C, M = env.GRID_SIZE_ROW, env.GRID_SIZE_COL, env.max_miners
    rG, cG = R // 2, C // 2
    grid = jnp.full((R, C), int(Items.empty), dtype=jnp.int32)
    if with_gold:
        grid = grid.at[rG, cG].set(int(Items.gold_ore))
    locs = jnp.array([
        [rG, cG - 1, 1],       # a0 faces right -> gold at dist 1
        [1, 1, 0],             # a1 parked
        [rG, cG + 1, 3],       # a2 faces left  -> gold at dist 1
        [R - 2, C - 2, 0],     # a3 parked
    ], dtype=jnp.int32)
    _, base = env.reset(jax.random.PRNGKey(0))
    state = base.replace(
        agent_locs=locs,
        grid=grid,
        ore_miners=-jnp.ones((R, C, M), dtype=jnp.int32),
        partial_ore_countdown=jnp.zeros((R, C), dtype=jnp.int32),
        inner_t=0,
    )
    return state, (rG, cG)


def _step(env, state, acts):
    # Fixed key -> deterministic; regrow prob is ~1e-4 so it never perturbs the
    # single crafted cell over the handful of steps these tests take.
    obs, ns, rew, done, info = env.step_env(
        jax.random.PRNGKey(123), state, jnp.array(acts, dtype=jnp.int32)
    )
    return ns, jnp.asarray(info["mining_gold"]).reshape(-1)


def test_genuine_pairing_pays_out():
    """Two distinct agents mining one gold cell in the same step finalize it."""
    env = _make_env()
    s, (rG, cG) = _crafted_state(env)
    s, gold = _step(env, s, [MINE, STAY, MINE, STAY])
    assert float(gold[2]) == env.gold_reward * env.num_agents
    assert int(s.grid[rG, cG]) == int(Items.ore_wait)          # finalized
    # registry cleared on finalize so the cell can require a fresh pair next time
    assert [int(x) for x in s.ore_miners[rG, cG]] == [-1] * env.max_miners


def test_no_solo_refarm_after_regrow():
    """After a pairing finalizes a cell and gold regrows there, a single
    already-registered agent must NOT be able to re-finalize it alone."""
    env = _make_env()
    s, (rG, cG) = _crafted_state(env)
    s, _ = _step(env, s, [MINE, STAY, MINE, STAY])             # genuine pairing
    s = s.replace(grid=s.grid.at[rG, cG].set(int(Items.gold_ore)))  # regrow
    s, gold = _step(env, s, [MINE, STAY, STAY, STAY])          # a0 alone
    assert float(gold[0]) == 0.0                               # no solo payout
    assert int(s.grid[rG, cG]) == int(Items.gold_partial)      # only a partial
    assert [int(x) for x in s.ore_miners[rG, cG]] == [0, -1, -1, -1]


def test_lone_partial_times_out():
    """A partial with no second miner reverts to gold_ore once the window ends."""
    env = _make_env()
    s, (rG, cG) = _crafted_state(env)
    s, _ = _step(env, s, [MINE, STAY, STAY, STAY])             # a0 taps -> partial
    assert int(s.grid[rG, cG]) == int(Items.gold_partial)
    for _ in range(WINDOW + 1):
        s, _ = _step(env, s, [STAY, STAY, STAY, STAY])
    assert int(s.grid[rG, cG]) == int(Items.gold_ore)          # reverted
    assert [int(x) for x in s.ore_miners[rG, cG]] == [-1] * env.max_miners


def test_repair_after_timeout():
    """After a timeout revert, a fresh pairing on the same cell still works."""
    env = _make_env()
    s, (rG, cG) = _crafted_state(env)
    s, _ = _step(env, s, [MINE, STAY, STAY, STAY])
    for _ in range(WINDOW + 1):
        s, _ = _step(env, s, [STAY, STAY, STAY, STAY])
    s, gold = _step(env, s, [MINE, STAY, MINE, STAY])
    assert float(gold[2]) == env.gold_reward * env.num_agents


if __name__ == "__main__":
    tests = [
        test_genuine_pairing_pays_out,
        test_no_solo_refarm_after_regrow,
        test_lone_partial_times_out,
        test_repair_after_timeout,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print("=" * 60)
    print("ALL PASS" if failed == 0 else f"{failed} FAILED")
    raise SystemExit(1 if failed else 0)
