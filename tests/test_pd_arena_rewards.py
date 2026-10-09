"""PD Arena: a zap between two agents pays the matrix game on their inventory proportions (SJ-v3, 2026-10-09).

Payoff [[[3, -1], [5, 1]], [[3, 5], [-1, 1]]] (row = zapper's play, column = target's; individual mode scales by
N = 2), both inventories are emptied, and an empty inventory pays nothing. Under a random policy this almost never
happens: 3,000 random steps (2 agents, 1000-step episodes) paid no reward at all, so the env is very sparse.

Note the env teleports every agent to `reborn_locs` at the start of each step, so a hand-built state must set both.

Run: PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/test_pd_arena_rewards.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import socialjax


def _facing_pair():
    env = socialjax.make("pd_arena", num_agents=2, shared_rewards=False, num_inner_steps=1000)
    mod = sys.modules[type(env).__module__]
    obs, st = env.reset(jax.random.PRNGKey(0))
    r0, c0, o0 = [int(x) for x in st.agent_locs[0]]
    d = onp.asarray(mod.STEP[o0])
    r1, c1 = r0 + int(d[0]), c0 + int(d[1])
    g = onp.asarray(st.grid).copy()
    assert g[r1, c1] == 0, "agent 0 does not face an empty cell at seed 0"
    g[int(st.agent_locs[1][0]), int(st.agent_locs[1][1])] = 0
    g[r1, c1] = int(env._agents[1])
    locs = onp.asarray(st.agent_locs).copy()
    locs[1, :2] = [r1, c1]
    locs = jnp.asarray(locs, dtype=st.agent_locs.dtype)
    st = st.replace(grid=jnp.asarray(g, dtype=st.grid.dtype), agent_locs=locs, reborn_locs=locs)
    return env, mod, st


@pytest.mark.parametrize("coop, defect, expected", [
    ([2, 0], [0, 1], [-1, 5]),   # cooperator zaps defector
    ([2, 3], [0, 0], [3, 3]),    # both cooperate
    ([0, 0], [1, 1], [1, 1]),    # both defect
    ([0, 0], [0, 0], [0, 0]),    # empty inventories
])
def test_zap_pays_matrix_game(coop, defect, expected):
    env, mod, st = _facing_pair()
    st = st.replace(coop_resources=jnp.array(coop, dtype=st.coop_resources.dtype),
                    defect_resources=jnp.array(defect, dtype=st.defect_resources.dtype))
    _, st2, rew, _, _ = env.step_env(jax.random.PRNGKey(1), st, [jnp.array(mod.Actions.interact), jnp.array(mod.Actions.stay)])
    assert onp.allclose(onp.asarray(rew).reshape(-1), 2 * onp.array(expected))
    assert onp.asarray(st2.coop_resources).sum() == 0 and onp.asarray(st2.defect_resources).sum() == 0
