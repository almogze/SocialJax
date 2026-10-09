"""Coin Game: reward accounting per pickup (SJ-v3, 2026-10-09).

Agent 0 owns red coins, agent 1 green. Payoff [[1, 1, -2], [1, 1, -2]]: a pickup pays the picker +1 whatever the
colour, and taking the other agent's coin costs the owner -2. So an own pickup is +1 for the team and a taken coin
-1 (the social dilemma MARP is meant to see). Individual mode scales everything by N = 2.

Pickups are read from the grid before the step at each agent's new cell. A coin that spawns on that cell inside
the same step is invisible to this check, so a mismatch is allowed only where the cell was empty before the step.

Run: PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/test_coin_game_rewards.py
"""

import jax
import numpy as onp

import socialjax

RED, GREEN = 3, 4  # Items.red_apple, Items.green_apple


def _pickups(cell):
    own = onp.array([cell[0] == RED, cell[1] == GREEN], dtype=float)
    other = onp.array([cell[0] == GREEN, cell[1] == RED], dtype=float)
    return own, other


def _expected(cell, n):
    own, other = _pickups(cell)
    return n * (own + other - 2 * other[::-1])


def test_rewards_match_pickups():
    n = 2
    env = socialjax.make("coin_game", shared_rewards=False, regrow_rate=0.005)
    step = jax.jit(env.step_env)
    pickups = taken = unexplained = 0
    for seed in range(3):
        key = jax.random.PRNGKey(seed)
        rng = onp.random.default_rng(seed)
        key, sub = jax.random.split(key)
        obs, state = env.reset(sub)
        for _ in range(1000):
            prev = onp.asarray(state.grid)
            key, sub = jax.random.split(key)
            obs, state, rew, done, info = step(sub, state, [jax.numpy.array(a) for a in rng.integers(0, 7, size=n)])
            if int(state.inner_t) == 0:
                continue  # episode-end step: every SocialJax env returns zero reward on it, and state is the reset one
            locs = onp.asarray(state.agent_locs)
            cell = [int(prev[r, c]) for r, c in locs[:, :2]]
            rew = onp.asarray(rew).reshape(-1)
            if not onp.allclose(rew, _expected(cell, n)):
                # only a coin spawned this step on an empty cell may explain the difference
                spawns = [cell[:i] + [col] + cell[i + 1:] for i in range(n) if cell[i] == 0 for col in (RED, GREEN)]
                assert any(onp.allclose(rew, _expected(c, n)) for c in spawns), (cell, rew)
                unexplained += 1
                continue
            own, other = _pickups(cell)
            assert onp.allclose(onp.asarray(info["eat_own_coins"]).reshape(-1), n * own)
            pickups += int(own.sum() + other.sum())
            taken += int(other.sum())
            assert abs(rew.sum() - n * (own.sum() - other.sum())) < 1e-6
    assert pickups > 50 and taken > 10, (pickups, taken)
    assert unexplained <= 0.05 * pickups, (unexplained, pickups)
