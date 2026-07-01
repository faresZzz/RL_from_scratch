# RL from Scratch

A from-scratch reinforcement-learning curriculum in PyTorch: 25 algorithms, from
tabular Q-learning to an action-conditioned JEPA world model, each rebuilt by hand and
kept short enough to read next to the paper that introduced it. This is a paper-with-code
study project, not a framework and not a benchmark leaderboard. The aim is that a reader
who knows the theory can open any chapter, follow the implementation line by line, and see
the method actually learn.

Every algorithm family lives in its own small package on top of a shared `core/` runtime,
and each is paired with a notebook that develops the intuition, the equations, and a
runnable demo. The ordering is deliberate: each method answers a limitation of the
previous one, and the model-based block builds toward the capstone, **Action-JEPA**: a
decoder-free latent world model in the spirit of V-JEPA 2-AC, at a readable scale.

- **Scope:** 25 registered approaches across 14 algorithm packages, ~26k lines of Python,
  17 walkthrough notebooks, 55 YAML configs, 459 tests.
- **Stack:** PyTorch, Gymnasium (CartPole, Pendulum, HalfCheetah-v5/MuJoCo), PettingZoo
  (Connect Four), `uv` for environment and dependency management.

## Why this project exists

Reinforcement learning is easy to *run* with existing libraries and hard to *reason about*
when the mechanics are hidden behind a `.train()` call. I built this to keep them visible:
how a Q-table updates, how a policy gradient flows, how a target network is soft-updated,
and how a latent world model is rolled out and planned in without ever decoding back to
pixels.

It also maps onto my research interest in representation learning and predictive world
models. The model-based progression (Dyna → PILCO → PETS → MBPO → Dreamer → MuZero) leads
into Action-JEPA: a world model that predicts in representation space and plans with
CEM/MPC. It keeps the representation from collapsing with EMA targets, a stop-gradient, and
VICReg terms. Same idea family as the recent JEPA line of work, small enough to inspect end
to end.

## What the repo covers

| Block | Methods | Environment | Core idea it teaches |
|-------|---------|-------------|----------------------|
| Tabular control | Q-learning, SARSA | CartPole | Off-policy vs on-policy TD, ε-greedy, discretization |
| Deep value methods | DQN, Double DQN, Rainbow | CartPole | Replay, target nets, double/dueling/distributional, PER, noisy nets |
| Policy gradients | REINFORCE, REINFORCE + baseline | CartPole | Score-function estimator, returns, variance reduction |
| Actor-critic | A2C, A2C-GAE, A3C | HalfCheetah | Advantages, GAE, synchronous and asynchronous updates |
| Trust-region / clipped | TRPO, PPO | HalfCheetah | Natural gradient, KL constraint, clipped surrogate |
| Deterministic actor-critic | DDPG, TD3 | HalfCheetah | Deterministic policy gradient, twin critics, target smoothing |
| Maximum-entropy | SAC | HalfCheetah | Entropy-regularized off-policy, learned temperature |
| Dyna | Dyna-Q, Dyna-Q+, Deep Dyna | CartPole / discrete | Learned model + planning updates, exploration bonus |
| Analytic model-based | PILCO, Deep PILCO | InvertedPendulum | GP / BNN dynamics, moment matching, fixed-horizon rollouts |
| Ensemble model-based | PETS, MBPO | HalfCheetah | Probabilistic ensembles, MPC/CEM, short model rollouts |
| Latent world models | Dreamer, MuZero | HalfCheetah / Connect Four | RSSM imagination, value-equivalent latents, MCTS |
| Capstone | Action-JEPA | Pendulum, HalfCheetah | Decoder-free latent prediction + CEM planning |

## Capstone: Action-JEPA

The final chapter is the one I would point a JEPA researcher to first. It builds a small
action-conditioned Joint-Embedding Predictive Architecture and uses it for control:

1. **Masked-JEPA pretraining** learns a representation by predicting masked targets in
   latent space, with an EMA target encoder and a stop-gradient, plus VICReg variance and
   covariance terms to keep the representation from collapsing. No pixel reconstruction
   anywhere.
2. **Action-conditioned post-training** freezes the encoder and trains a predictor
   `(z_t, a_t) → ẑ_{t+1}` with reward and continuation heads.
3. **Planning** runs CEM/MPC entirely in latent space, toward a goal latent or a predicted
   return.

The notebook frames it as a controlled study: **joint end-to-end training vs a stage-wise
pretrain-then-freeze pipeline**, compared at matched data and matched compute budgets, with
diagnostics for collapse (latent std, effective rank), action sensitivity, and open-loop
rollout drift. It is explicit about what the real V-JEPA 2-AC has that this does not (a
frozen large video encoder, a block-causal transformer, goal images instead of goal
states). The point is to demonstrate the mechanism faithfully, not to match a large-scale
result.

## Engineering and design

- **From scratch.** No RL-library black boxes. Networks, replay/rollout buffers, planners,
  losses, and training loops are written out so they can be read and modified.
- **Shared runtime, independent chapters.** `rl_from_scratch.core.*` owns the config and
  agent registries, run artifacts, history recording, the standard metric vocabulary,
  evaluation cadence, environment helpers, and reporting primitives. Each algorithm package
  depends only on `core/` and never on a sibling, so any chapter reads in isolation. An
  architecture test enforces this boundary.
- **One way to add a method.** Register a `BaseConfig` subclass with `@register_config`
  and a `train_<name>(config, *, output_dir, run_name, seed, render)` with
  `@register_agent`; the CLI and benchmark runner discover it automatically. Every training
  function returns the same `{agent, history, metrics, paths}` contract.
- **Reproducible by construction.** Runs are seeded across Python/NumPy/PyTorch and the
  environment; configs are serialized with the run; evaluation is deterministic; YAML keys
  are validated strictly (an unknown key fails loudly rather than being silently ignored).
- **Tested.** 459 tests cover config validation, network shapes, buffer round-trips,
  finite-loss checks, deterministic evaluation, save/load, per-method smoke training against
  the standard contract, and the cross-package architecture boundary.

> The package code, comments, and docstrings are in English. The notebooks are written in
> French as full pedagogical courses (intuition, equations in LaTeX, and small code
> demonstrations) and stay followable alongside the code.

## Results and honest positioning

This is a paper-with-code project. The goal is a faithful, readable reimplementation of
each method and evidence that it learns, not a tuned benchmark sweep and not a claim of
state-of-the-art numbers.

- **Proof of learning lives in committed figures.** The publication gate forbids notebook
  cell outputs, so each notebook embeds the learning curves as committed PNGs rather than
  as stale execution output. Every method was run end to end during development.
- **Continuous control reaches paper-range results.** As a concrete example, **SAC reaches
  roughly 10k return on HalfCheetah-v5 after about 1M environment steps**, in the range
  reported in the original paper. That comes from a single seeded run, not a tuned
  multi-seed sweep. The other continuous-control methods (A2C, PPO, TRPO, DDPG, TD3)
  reproduce the expected learning behavior on the same environment.
- **Classic and value-based methods** (Q-learning, SARSA, DQN, Double DQN, Rainbow,
  REINFORCE) learn their CartPole tasks as expected.
- **Model-based and world-model methods** (PILCO, Deep PILCO, PETS, MBPO, Dreamer, MuZero,
  Action-JEPA) are trainable and demonstrate the mechanism. Some are well known to be hard
  to reproduce faithfully (PILCO in particular), and where a result is only partial it is
  documented honestly rather than dressed up.

No SOTA claim is made anywhere. Configs ship in two flavors: a real config sized to learn,
and a short `*_smoke` config for CI; smoke numbers are sanity checks, not benchmark
results.

## References

Paper-to-code: take one idea from the literature, implement it in PyTorch, verify it in a
notebook, connect it to the shared runtime. Main references by block:

**Foundations**
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) (Sutton & Barto) — tabular control, TD learning, Dyna.

**Value-based**
- [Human-level control through deep RL](https://www.nature.com/articles/nature14236) (Mnih et al., 2015) — DQN.
- [Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2016).
- [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) (Hessel et al., 2018).

**Policy gradient and actor-critic**
- [Simple statistical gradient-following algorithms](https://link.springer.com/article/10.1007/BF00992696) (Williams, 1992) — REINFORCE.
- [Asynchronous Methods for Deep RL](https://arxiv.org/abs/1602.01783) (Mnih et al., 2016) — A3C/A2C.
- [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438) (Schulman et al., 2016).
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (Schulman et al., 2015) — TRPO.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) — PPO.
- [Continuous control with deep RL](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2015) — DDPG.
- [Addressing Function Approximation Error in Actor-Critic](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018) — TD3.
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) — SAC.

**Model-based and world models**
- [PILCO: A Model-Based and Data-Efficient Approach](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Deisenroth_ICML_2011.pdf) (Deisenroth & Rasmussen, 2011).
- [Deep Reinforcement Learning in a Handful of Trials (PETS)](https://arxiv.org/abs/1805.12114) (Chua et al., 2018).
- [When to Trust Your Model (MBPO)](https://arxiv.org/abs/1906.08253) (Janner et al., 2019).
- [Dream to Control (Dreamer)](https://arxiv.org/abs/1912.01603) (Hafner et al., 2020).
- [Mastering Atari, Go, Chess and Shogi by Planning (MuZero)](https://arxiv.org/abs/1911.08265) (Schrittwieser et al., 2020).

**Latent prediction (capstone)**
- [Self-Supervised Learning from Images with I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., 2023).
- [V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) (Assran et al., 2025).
- [VICReg](https://arxiv.org/abs/2105.04906) (Bardes et al., 2021) and [BYOL](https://arxiv.org/abs/2006.07733) (Grill et al., 2020) — anti-collapse.

## Repository structure

```text
src/rl_from_scratch/
  core/                          # BaseConfig/BaseAgent, registries, artifacts,
                                 #   recording, metrics, env helpers, reporting,
                                 #   shared utils (soft_update, diagnostics mixin)
  tabular/  deep_q/  reinforce/  actor_critic/  trust_region/
  deterministic_actor_critic/  sac/
  dyna/  pilco/  pets/  mbpo/  dreamer/  muzero/  action_jepa/
                                 # each: config.py, agent.py, networks/buffers,
                                 #   training.py, reporting.py
  train.py                       # YAML-driven CLI to launch any approach
  benchmark.py                   # multi-seed runner + aggregated summaries

notebooks/                       # 01..15 (+ 02b, 10b) pedagogical walkthroughs
configs/<package>/               # YAML configs per approach and environment
tests/                           # unit, smoke, and architecture tests
```

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

Run a single experiment:

```bash
python -m rl_from_scratch.train --config configs/sac/sac_halfcheetah.yaml
```

Run the same config across seeds and aggregate:

```bash
python -m rl_from_scratch.benchmark --config configs/sac/sac_halfcheetah.yaml --seeds 0 1 2
```

CLI overrides can shorten or redirect a run:

```bash
python -m rl_from_scratch.train \
  --config configs/tabular/q_learning_cartpole.yaml \
  --episodes 100 --output-dir runs --run-name q-learning-debug
```

Each run writes to `runs/<approach>/<run_name>/` (`config.json`, `history.json`,
`metrics.json`, `checkpoints/`, `figures/`); benchmark summaries go to a sibling
`<run_name>-benchmark-<timestamp>/summary.json`. Every algorithm also ships a `*_smoke`
config that runs in seconds for CI.

## What stays local

The repository versions code, configs, and cleaned notebooks. Generated artifacts stay
local and are gitignored:

- `runs/` — per-run `config.json`, `history.json`, `metrics.json`, checkpoints, figures
- `data/` and learned weights / Q-tables
- `.venv/` and local editor state

Keeping these out of git keeps the repository light and reproducible from source.

## Topics covered

- Tabular control: ε-greedy exploration, Q-learning vs SARSA, state discretization
- Deep value methods: replay buffers, target networks, double/dueling/distributional, PER, noisy nets
- Policy gradients: the score function, returns, baselines, variance reduction
- Actor-critic: advantages, GAE, n-step rollouts, synchronous and asynchronous updates
- Trust-region and clipped objectives: natural gradient, KL constraints, PPO clipping
- Continuous control: deterministic policies, twin critics, target policy smoothing, entropy regularization
- Model-based RL: learned dynamics, planning, Gaussian processes, probabilistic ensembles, model rollouts
- Latent world models: recurrent state-space models, value-equivalent latents, MCTS, latent CEM/MPC planning
- Anti-collapse for joint-embedding prediction: EMA targets, stop-gradient, VICReg

## Limitations

- Experiments run at small to moderate scale, developed on a Mac (Apple Silicon); most
  results come from single seeded or 3 seeds runs rather than large multi-seed sweeps.
- Hand-crafted discretization for the tabular chapter is pedagogical and does not scale to
  large state spaces.
- The model-based and world-model implementations are trainable but pedagogical, not
  industrial reproductions; the Action-JEPA capstone illustrates the mechanism rather than
  a state-of-the-art result, and the harder methods (PILCO/Deep-PILCO) only have
  partial results due to computational constraints and stability issues.
- No trained checkpoints, datasets, or benchmark figures are published; runs stay local.

## Status

All 14 algorithm families and their 25 registered approaches are implemented over a shared
`core/` runtime, with YAML configs, a CLI and multi-seed benchmark runner, a 459-test suite
(unit, smoke, and architecture), and a paired walkthrough notebook per block. The package
code is uniform and English-only; the notebooks are complete standalone pedagogical
courses with their proof-of-learning figures committed.

## License

See [LICENSE](LICENSE) (MIT).
