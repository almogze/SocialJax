"""Territory: claim rewards and the 100-step timer, as the env actually behaves (SJ-v3, 2026-10-09).

An agent claims a resource cell by facing it (range 1) after moving; the cell takes the colour 1000 + i. Each step
an agent is paid 0.01 per cell of its colour whose timer is >= 100 (individual mode, no x N scaling). The timer
counts steps since the cell last changed *inside the zap phase*; claims are painted before that phase starts, so
claiming or re-colouring (stealing) a cell does not reset it. In practice the 100-step wait applies only from the
episode start and after a zap; a claim on a cell untouched for 100 steps pays at once, and a theft moves the
income to the thief at once. Measured with a random policy (2 seeds x 999 steps, 9 agents): 68 of 171 fresh claims
and 48 of 88 thefts paid on the step they were made. Kept as upstream defines it (documented, not changed).

Run: PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/test_territory_rewards.py
"""

import jax
import numpy as onp

import socialjax


def test_claim_reward_formula_and_timer():
    n = 9
    env = socialjax.make("territory_open", num_agents=n, shared_rewards=False)
    step = jax.jit(env.step_env)
    first_paid, paid_at_once = None, 0
    key = jax.random.PRNGKey(0)
    rng = onp.random.default_rng(0)
    key, sub = jax.random.split(key)
    obs, state = env.reset(sub)
    prev = onp.asarray(state.grid)
    for t in range(999):
        key, sub = jax.random.split(key)
        obs, state, rew, done, info = step(sub, state, [jax.numpy.array(a) for a in rng.integers(0, 9, size=n)])
        grid, timer = onp.asarray(state.grid), onp.asarray(state.claimed_indicator_time_matrix)
        rew = onp.asarray(rew).reshape(-1)
        expected = 0.01 * onp.array([((grid == 1000 + i) & (timer >= 100)).sum() for i in range(n)])
        assert onp.allclose(rew, expected, atol=1e-5), (t, rew, expected)
        if rew.any() and first_paid is None:
            first_paid = t
        newly = (grid >= 1000) & (grid < 1000 + n) & (grid != prev)
        paid_at_once += int((newly & (timer >= 100)).sum())
        prev = grid
    assert first_paid is not None and first_paid >= 99, first_paid  # nothing pays in an episode's first 100 steps
    assert paid_at_once > 0  # documents the quirk: some claims pay on the step they are made
