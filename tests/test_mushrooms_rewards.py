"""Mushrooms: reward accounting per eat event, and independent regrowth draws (SJ-v3, 2026-10-09).

Per step, with r/g/b/o the number of red/green/blue/orange mushrooms eaten (state.mushrooms_matches) and N agents,
the env pays (before its x N scaling): eater of red +1; every agent +2/N per green; every agent except the eater
+3/(N-1) per blue; every agent -0.2 per orange. So the team total is N * (r + 2g + 3b - 0.2*N*o), and a blue
eater gets nothing from its own blue.

Run: PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/test_mushrooms_rewards.py
"""

import jax
import numpy as onp

import socialjax

STEPS = 600


def _rollout(seed=0, n=6):
    env = socialjax.make("mushrooms", num_agents=n, shared_rewards=False)
    key = jax.random.PRNGKey(seed)
    rng = onp.random.default_rng(seed)
    key, sub = jax.random.split(key)
    obs, state = env.reset(sub)
    step = jax.jit(env.step_env)
    for _ in range(STEPS):
        key, sub = jax.random.split(key)
        acts = [jax.numpy.array(a) for a in rng.integers(0, 8, size=n)]
        obs, state, rew, done, info = step(sub, state, acts)
        yield onp.asarray(state.mushrooms_matches), onp.asarray(rew).reshape(-1), state


def test_team_reward_matches_eat_events():
    n = 6
    events = 0
    for m, rew, _ in _rollout(n=n):
        r, g, b, o = m.sum(axis=0)
        events += int(m.sum())
        assert abs(rew.sum() - n * (r + 2 * g + 3 * b - 0.2 * n * o)) < 1e-3
        for i in onp.flatnonzero(m[:, 2]):  # blue eaters: no share of their own blue
            others_blue = b - 1
            expected = n * (m[i, 0] + 2 * g / n + 3 * others_blue / (n - 1) - 0.2 * o)
            assert abs(rew[i] - expected) < 1e-3
    assert events > 0, "random policy ate nothing: the test checked nothing"


def test_regrowth_draws_are_independent_across_colours():
    # Upstream drew red, green and blue regrowth from one key, so a red regrowth in a slot came with a green
    # one whenever green's slot was open. Measured over these fixed seeds (deterministic): steps where red and
    # green both grew / steps where either grew = 0.35 with the shared key, 0.18 with independent keys.
    env = socialjax.make("mushrooms", num_agents=6, shared_rewards=False)
    step = jax.jit(env.step_env)
    both = either = 0
    for seed in range(4):
        key = jax.random.PRNGKey(seed)
        rng = onp.random.default_rng(seed)
        key, sub = jax.random.split(key)
        obs, state = env.reset(sub)
        prev = None
        for _ in range(900):
            key, sub = jax.random.split(key)
            obs, state, rew, done, info = step(sub, state, [jax.numpy.array(a) for a in rng.integers(0, 8, size=6)])
            grid = onp.asarray(state.grid)
            counts = onp.array([(grid == 3).sum(), (grid == 4).sum()])  # red_mushrooms, green_mushrooms
            if prev is not None:
                grew = counts > prev
                both += int(grew.all())
                either += int(grew.any())
            prev = counts
    assert either > 50
    assert both / either < 0.27, f"red and green regrow together in {both}/{either} steps: shared key?"
