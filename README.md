<h1 align="center">SocialJax</h1>


<p align="center">
  <a href="https://arxiv.org/abs/2503.14576">
    <img src="https://img.shields.io/badge/arXiv-2503.14576-B31B1B.svg" alt="arXiv"></a>
  <a href="https://github.com/cooperativex/SocialJax/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0 License"></a>
  <a href="https://github.com/cooperativex/SocialJax/actions/workflows/speet_example.yml">
    <img src="https://github.com/cooperativex/SocialJax/actions/workflows/speet_example.yml/badge.svg" alt="Pylint Status"></a>
</p>

*A suite of sequential social dilemma environments for multi-agent reinforcement learning in JAX*

---

## Fork Notes (`almogze/SocialJax`)

This is a research fork of [cooperativex/SocialJax](https://github.com/cooperativex/SocialJax),
used as the environment backend for the MARP (Multi-Agent Reward Prediction) project
(see the `marp-socialjax` repo). It tracks upstream but carries a handful of environment
fixes and research manipulations. Upstream is the `upstream` remote; this fork is `origin`.

**What this fork changes vs upstream**

| Change | Where | Branch |
|---|---|---|
| `jax.tree_map` → `jax.tree.map` compat + package renamed to `socialjax` | package-wide | `main` |
| **Gift dead-inventory observation fix** — populate `agent_invs` from `agents_bag` at end-of-step so the policy actually observes bag contents (ch0 = total tokens, ch1 = final-tier tokens) | `socialjax/environments/gift/gift.py` | `fix/agent-invs-obs-gift` |
| **Coop Mining gold-coordination fix** — reset the `ore_miners` registry on finalize/revert and enforce the cooperative-mining window, so "gold coordination" can no longer be faked by a stale registry + dead time-window (confirmed a real bug with upstream author Zihao Guo) | `socialjax/environments/coop_mining/coop_mining.py` | `fix/coop-mining-gold-coordination` |
| Coop Mining: surface per-agent iron reward in the `info` dict | `socialjax/environments/coop_mining/coop_mining.py` | `fix/coop-mining-gold-coordination` |
| Gift: double coin regrow probability (0.0002 → 0.0004) — research manipulation, not a bug fix | `socialjax/environments/gift/gift.py` | `fix/coop-mining-gold-coordination` |

**Regression tests for the fixes** live in `tests/` (e.g. `tests/test_coop_mining_gold_coordination.py`),
which exercise the real `step_env` path and were confirmed failing pre-fix.

> The gold-coordination fix is **not** merged into `bench/coop-mining` and no benchmark
> re-run has been done yet — the re-run-vs-caveat decision is pending. Treat `main` as the
> upstream-compat baseline and pull individual fixes from their feature branches as needed.

---



<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="/docs/images/step_150_reward_common_coins.gif" alt="coins_common" width="19.2%">
            <img src="/docs/images/step_150_reward_common_harvestopen.gif" alt="harvest_open_common" width="18.5%">
            <img src="/docs/images/step_150_reward_common_closed.gif" alt="harvest_closed_common" width="18.5%">
            <img src="/docs/images/step_150_reward_common_cleanup.gif" alt="clean_up_common" width="19.8%">
            <img src="/docs/images/step_250_reward_common_coop_mining.gif" alt="coop_mining_common" width="14%">
        </div>
    </div>
</div>

*Common Rewards* : a scenario where all agents share a single, unified reward signal. This approach ensures that all agents are aligned towards achieving the same objective, promoting collaboration and coordination among them.

<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="/docs/images/step_150_reward_individual_coins.gif" alt="coins_individual" width="19.2%">
            <img src="/docs/images/step_150_reward_individual_harvestopen.gif" alt="harvest_open_individual" width="18.5%">
            <img src="/docs/images/step_150_reward_individual_closed.gif" alt="harvest_closed_individual" width="18.5%">
            <img src="/docs/images/step_150_reward_individual_cleanup.gif" alt="clean_up_individual" width="19.8%">
            <img src="/docs/images/step_250_reward_individual_coop_mining.gif" alt="coop_mining_individual" width="14%">
        </div>
    </div>
</div>

***Individual Rewards***: each agent is assigned its own reward, inherently encouraging selfish behavior.


SocialJax leverages JAX's high-performance GPU capabilities to accelerate multi-agent reinforcement learning in sequential social dilemmas. We are committed to providing a more efficient and diverse suite of environments for studying social dilemmas. We provide JAX implementations of the following environments: Coins, Commons Harvest: Open, Commons Harvest: Closed, Clean Up, Territory, and Coop Mining, which are derived from [Melting Pot 2.0](https://github.com/google-deepmind/meltingpot/) and feature commonly studied mixed incentives.


Our [blog](https://sites.google.com/view/socialjax/home) presents more details and analysis on agents' policy and performance.

## Update

***[2025/05/28]*** ✨ Updated [SVO](https://github.com/cooperativex/SocialJax/tree/main/algorithms/SVO) algorithm for all environments.

***[2025/04/29]*** 🚀 Updated [Mushrooms](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/mushrooms) environment.

***[2025/04/28]*** 🚀 Updated [Gift Refinement](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/gift) environment.

***[2025/04/16]*** ✨ Added [MAPPO](https://github.com/cooperativex/SocialJax/tree/main/algorithms/MAPPO) algorithm for all environments.


## Installation

First: Clone the repository


Second: Environment Setup.

Option one: Using peotry, make sure you have python 3.10
  1. Install Peotry
       ```bash
       curl -sSL https://install.python-poetry.org | python3 -
       export PATH="$HOME/.local/bin:$PATH"
       ```

  2. Install requirements     
       ```bash
       poetry install --no-root
       poetry run pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```
  3. Run code
       ```bash
       poetry run python algorithms/IPPO/ippo_cnn_coins.py 
       ```

Option two: conda with requirements.txt
  1. Conda
       ```bash
       conda create -n SocialJax python=3.10
       conda activate SocialJax
       ```

  2. Install requirements
       ```bash
       pip install -r requirements.txt
       pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```

  3. Run code
       ```bash
       python algorithms/IPPO/ippo_cnn_coins.py 
       ```

Option three: conda with environments.yml

  1. Install requirements
       ```bash
       conda env create -f environment.yml
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```

  2. Run code
       ```bash
       python algorithms/IPPO/ippo_cnn_coins.py 
       ```

## Environments

We introduce the environments and use Schelling diagrams to demonstrate whether the environments are social dilemmas. 

| Environment                  | Description                                                                                      | Schelling Diagrams Proof |
|------------------------------|--------------------------------------------------------------------------------------------------|:------------------------:|
| Coins                        | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coins)         |&check;                   |
| Commons Harvest: Open        | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest)|&check;                   |
| Commons Harvest: Closed      | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest)|&check;                   |
| Commons Harvest: partnership | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest)|&check;                   |
| Clean Up                     | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/cleanup)       |&check;                   |
| Territory                    | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/territory)     |&cross;                   |
| Coop Mining                  | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coop_mining)   |&check;                   |
| Mushrooms                    | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/mushrooms)     |&check;                   |
| Gift Refinement              | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/gift)          |&check;                   |
| Prisoners Dilemma: Arena     | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/pd_arena)      |&check;                   |



#### Important Notes:
- *Due to algorithmic limitations, agents may not always learn the optimal actions. As a result, Schelling diagrams can prove that the environment is social dilemmas, but they cannot definitively prove that the environment is not social dilemmas.*

- *Territory might not be Social diagram, but as long as the agents' behaviors are interesting, Territory holds intrinsic value.*
  
## Quick Start

SocialJax interfaces follow [JaxMARL](https://github.com/FLAIROx/JaxMARL/) which takes inspiration from the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [Gymnax](https://github.com/RobertTLange/gymnax).

### Make an Environment
You can create an environment using the ```make``` function:
```python
import jax
import socialjax

env = make('clean_up')
```

### Example

Find more fixed policy [examples](https://github.com/cooperativex/SocialJax/tree/main/fixed_policy).

```python
import jax
import socialjax
from socialjax import make

num_agents = 7
env = make('clean_up', num_agents=num_agents)
rng = jax.random.PRNGKey(259)
rng, _rng = jax.random.split(rng)

for t in range(100):
     rng, *rngs = jax.random.split(rng, num_agents+1)
     actions = [jax.random.choice(
          rngs[a],
          a=env.action_space(0).n,
          p=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
     ) for a in range(num_agents)]

     obs, state, reward, done, info = env.step_env(
          rng, old_state, [a for a in actions]
            )
```

### Speed test

You can test the speed of our environments by running [speed_test_random.py](https://github.com/cooperativex/SocialJax/blob/main/speed_test/speed_test_random.py) or using the [colab](https://colab.research.google.com/github/cooperativex/SocialJax/blob/main/speed_test/speed_test_random.ipynb).



## See Also

[JaxMARL](https://github.com/flairox/jaxmarl): accelerated MARL environments with baselines in JAX.

[PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
