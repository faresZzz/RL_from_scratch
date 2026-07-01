# RL_from_scratch - Plan de projet complet

## Contexte

Le projet `RL_from_scratch` part d'un ancien prototype CartPole (Q-learning tabulaire) deja restructure en layout Python publiable (`src/rl_from_scratch/`). L'objectif est de le transformer en projet portfolio complet qui implemente et compare les algorithmes RL fondamentaux, du tabulaire au model-based, en progressant sur des environnements de difficulte croissante.

L'etat actuel du repo couvre deja les fondations tabulaires, value-based deep, policy gradients on-policy, actor-critic continus et off-policy continus. Ce document doit donc distinguer clairement :

- ce qui existe deja dans le repo ;
- ce qui est la cible proche mais encore a implementer ;
- ce qui releve du capstone plus ambitieux.

## Decisions cles

- **Progression multi-env** : CartPole-v1 (tabular et premiers deep basiques) -> HalfCheetah-v5 MuJoCo continu (controle continu) -> quelques environnements choisis pour les capstones model-based
- **Framework deep** : PyTorch (MPS Apple Silicon disponible)
- **CartPole** : garde comme chapitre pedagogique d'entree, y compris pour DQN et Rainbow
- **HalfCheetah-v5** : environnement de comparaison principal pour les algorithmes continus deja presents et plusieurs algorithmes model-based futurs
- **Model-based progression explicite** : Dyna-Q / Dyna-Q+ / Deep Dyna -> PILCO / Deep PILCO -> PETS -> MBPO -> Dreamer
- **MuZero** : capstone separe, implemente dans `muzero/`, sur environnement discret simple + Connect-4
- **Action-JEPA** : capstone bonus de world model latent predictif, implemente dans `action_jepa/`, avec Pendulum et micro-demonstration HalfCheetah
- **Pont world models** : PETS / MBPO / Dreamer / MuZero / Action-JEPA forment la transition vers les world models latents modernes

## Etat reel du projet aujourd'hui

### Packages deja presents

- `tabular/`
- `deep_q/`
- `reinforce/`
- `actor_critic/`
- `trust_region/`
- `deterministic_actor_critic/`
- `sac/`
- `dyna/`
- `pilco/`
- `pets/`
- `mbpo/`
- `dreamer/`
- `muzero/`
- `action_jepa/`

### Notebooks deja presents

- `01_tabular_rl_cartpole_walkthrough.ipynb`
- `02_dqn_cartpole_walkthrough.ipynb`
- `03_reinforce_cartpole_walkthrough.ipynb`
- `04_a2c_halfcheetah_walkthrough.ipynb`
- `05_a3c_halfcheetah_walkthrough.ipynb`
- `06_trpo_ppo_halfcheetah_walkthrough.ipynb`
- `07_ddpg_td3_halfcheetah_walkthrough.ipynb`
- `08_sac_halfcheetah_walkthrough.ipynb`
- `02b_rainbow_dqn_cartpole_walkthrough.ipynb`
- `09_dyna_q_dyna_q_plus_deep_dyna_walkthrough.ipynb`
- `10_pilco_walkthrough.ipynb`
- `10b_deep_pilco_walkthrough.ipynb`
- `11_pets_halfcheetah_walkthrough.ipynb`
- `12_mbpo_halfcheetah_walkthrough.ipynb`
- `13_dreamer_walkthrough.ipynb`
- `14_muzero_discrete_control_walkthrough.ipynb`
- `15_action_jepa_walkthrough.ipynb`

## References papiers (SpinningUp + originaux)

| Algorithme | Papier cle | Auteurs, annee |
|---|---|---|
| Q-learning | Q-learning | Watkins, 1989 |
| SARSA | On-line Q-learning using connectionist systems | Rummery & Niranjan, 1994 |
| DQN | Playing Atari with Deep RL | Mnih et al., 2013 |
| Double DQN | Deep Reinforcement Learning with Double Q-learning | van Hasselt et al., 2015 |
| Rainbow DQN | Rainbow: Combining Improvements in Deep RL | Hessel et al., 2017 |
| REINFORCE | Policy Gradient Methods for RL with Function Approx | Sutton et al., 2000 |
| A2C/A3C | Asynchronous Methods for Deep RL | Mnih et al., 2016 |
| GAE | High-Dimensional Continuous Control Using GAE | Schulman et al., 2015 |
| TRPO | Trust Region Policy Optimization | Schulman et al., 2015 |
| PPO | Proximal Policy Optimization Algorithms | Schulman et al., 2017 |
| DDPG | Continuous Control With Deep RL | Lillicrap et al., 2015 |
| TD3 | Addressing Function Approx Error in Actor-Critic | Fujimoto et al., 2018 |
| SAC | Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL | Haarnoja et al., 2018 |
| Dyna-Q | Integrated Architectures for Learning, Planning, and Reacting | Sutton, 1990 |
| Dyna-Q+ /Deep Dyna | Reinforcement Learning: An Introduction (2nd ed., chap. Dyna) | Sutton & Barto, 2018 |
| PILCO | PILCO: A Model-Based and Data-Efficient RL Approach | Deisenroth & Rasmussen, 2011 |
| PETS | Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models | Chua et al., 2018 |
| MBPO | When to Trust Your Model: Model-Based Policy Optimization | Janner et al., 2019 |
| Dreamer | Dreamer: Reinforcement Learning with Latent Dynamics Models | Hafner et al., 2019 |
| MuZero | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model | Schrittwieser et al., 2019 |
| Action-JEPA | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | Assran et al., 2025 |

## Progression par complexite croissante

L'ordre suit la logique pedagogique et l'etat reel du repo.

```text
TABULAR (pas de NN)
  0. Q-learning / SARSA              CartPole-v1

VALUE-BASED DEEP
  1. DQN                            CartPole-v1
  2. Double DQN                     CartPole-v1
  2b. Rainbow DQN                   CartPole-v1

POLICY GRADIENT
  3. REINFORCE                      CartPole-v1

ON-POLICY ACTOR-CRITIC
  4. A2C / A2C-GAE / A3C            HalfCheetah-v5
  5. TRPO / PPO                     HalfCheetah-v5

OFF-POLICY CONTINU
  6. DDPG / TD3                     HalfCheetah-v5
  7. SAC                            HalfCheetah-v5

MODEL-BASED
  8. Dyna-Q / Dyna-Q+ / Deep Dyna   Gridworld / CartPole-v1
  9. PILCO                          Pendulum-v1 ou CartPole-v1 continu simplifie
  10. PETS                          HalfCheetah-v5
  11. MBPO                          HalfCheetah-v5
  12. Dreamer                       HalfCheetah-v5 ou env visuel plus adapte

CAPSTONE SEPARE
  13. MuZero                        env discret simple (CartPole, MinAtar, petits jeux grid)
  15. Action-JEPA                   Pendulum-v1 + micro HalfCheetah-v5
```

## Architecture cible

L'architecture cible doit matcher les dossiers reels deja presents et reserver proprement la place aux futures briques model-based. La plomberie partagee ne vit plus a la racine du package: elle appartient a `core/`.

```text
src/rl_from_scratch/
    __init__.py
    benchmark.py
    train.py
    core/
        __init__.py
        artifacts.py
        base.py
        config.py
        env.py
        metrics.py
        normalization.py
        recording.py
        reporting.py
        schedules.py
        utils.py

    tabular/
        __init__.py
        agent.py
        config.py
        discretization.py
        training.py

    deep_q/
        __init__.py
        agent.py
        buffer.py
        config.py
        network.py
        reporting.py
        training.py

    reinforce/
        __init__.py
        agent.py
        config.py
        network.py
        training.py

    actor_critic/
        __init__.py
        agent.py
        buffer.py
        config.py
        network.py
        optim.py
        reporting.py
        training.py

    trust_region/
        __init__.py
        agent.py
        buffer.py
        config.py
        network.py
        training.py

    deterministic_actor_critic/
        __init__.py
        agent.py
        buffer.py
        config.py
        network.py
        noise.py
        reporting.py
        training.py

    sac/
        __init__.py
        agent.py
        config.py
        network.py
        reporting.py
        training.py

    dyna/
        __init__.py
        agent.py
        buffer.py
        config.py
        model.py
        network.py
        reporting.py
        training.py

    pilco/
        __init__.py
        agent.py
        bnn.py
        config.py
        controller.py
        cost.py
        dynamics.py
        gp.py
        kernel.py
        moment_matching.py
        policy.py
        reporting.py
        rollout.py
        training.py

    pets/
        __init__.py
        agent.py
        config.py
        dynamics.py
        planner.py
        reporting.py
        training.py

    mbpo/
        __init__.py
        agent.py
        buffer.py
        config.py
        dynamics.py
        reporting.py
        training.py

    dreamer/
        __init__.py
        agent.py
        buffer.py
        config.py
        networks.py
        reporting.py
        rssm.py
        training.py

    muzero/
        __init__.py
        agent.py
        config.py
        connect_four.py
        mcts.py
        networks.py
        replay.py
        reporting.py
        training.py

    action_jepa/
        __init__.py
        agent.py
        buffer.py
        config.py
        networks.py
        planner.py
        reporting.py
        training.py
```

### Fichiers partages dans `core/`

| Fichier | Role | Utilise par |
|---|---|---|
| `core/base.py` | `BaseAgent` ABC, `BaseConfig` dataclass | Tous les algos |
| `core/env.py` | `make_env()` factory, helpers Gymnasium communs | Tous les algos |
| `core/metrics.py` | resume standardise des histories et metriques comparables | Tous |
| `core/config.py` | registre de configs, chargement YAML, dispatch par approche | Tous |
| `core/artifacts.py` | chemins d'experiences, checkpoints, sauvegardes JSON | Tous |
| `core/recording.py` | `RunRecorder` pour history/metrics/finalisation | Tous |
| `core/reporting.py` | primitives de visualisation et enregistrement greedy | Tous |
| `core/schedules.py` | cadences d'evaluation et checkpoints | Tous |
| `core/normalization.py` | normalisation partagee | Selon besoins |
| `core/utils.py` | seeds, device, moyennes mobiles | Tous |
| `comparison.py` | aggregation multi-runs et notebooks de comparaison | Comparaison finale |

## Notebooks cibles

Les notebooks existants `01` a `08` sont conserves tels quels dans leur numerotation actuelle.

```text
01_tabular_rl_cartpole_walkthrough.ipynb
02_dqn_cartpole_walkthrough.ipynb
02b_rainbow_dqn_cartpole_walkthrough.ipynb
03_reinforce_cartpole_walkthrough.ipynb
04_a2c_halfcheetah_walkthrough.ipynb
05_a3c_halfcheetah_walkthrough.ipynb
06_trpo_ppo_halfcheetah_walkthrough.ipynb
07_ddpg_td3_halfcheetah_walkthrough.ipynb
08_sac_halfcheetah_walkthrough.ipynb
09_dyna_q_dyna_q_plus_deep_dyna_walkthrough.ipynb
10_pilco_walkthrough.ipynb
10b_deep_pilco_walkthrough.ipynb
11_pets_halfcheetah_walkthrough.ipynb
12_mbpo_halfcheetah_walkthrough.ipynb
13_dreamer_walkthrough.ipynb
14_muzero_discrete_control_walkthrough.ipynb
15_action_jepa_walkthrough.ipynb
16_comparison.ipynb
```

## Phases d'implementation

### Phase -1 : Refactoring infrastructure (prerequis) ✅

**But** : Extraire les abstractions de base, generaliser config/training/CLI, deplacer le code tabulaire en sous-package. Tous les tests existants doivent toujours passer.

**Livraisons** :
1. `base.py` - `BaseAgent` ABC + `BaseConfig`
2. `env.py` - `make_env()` factory commune
3. `config.py` refactorise - registre avec `@register_config`, `load_config(path)`
4. `trainer.py` - `Trainer` generique
5. `metrics.py` - `MetricsCollector`
6. `utils.py` - `resolve_device()`, `set_all_seeds()`, `moving_average()`
7. `tabular/` sous-package
8. `train.py` refactorise avec dispatch par approche

### Phase 0 : Q-learning + SARSA tabulaire sur CartPole ✅

**But** : Valider l'infrastructure sur le code existant migre. Ajouter SARSA pour comparaison on-policy vs off-policy.

**Livraisons** : `tabular/agent.py`, `tabular/config.py`, `tabular/training.py`, notebook `01_tabular_rl_cartpole_walkthrough.ipynb`, tests tabular

### Phase 1 : DQN + Double DQN sur CartPole ✅

**But** : Premier deep RL. Replay buffer, target network, epsilon schedule. Double DQN corrige le biais de surestimation en separant selection et evaluation.

**Livraisons** : `deep_q/network.py`, `deep_q/buffer.py`, `deep_q/agent.py`, `deep_q/config.py`, `deep_q/training.py`, notebook `02_dqn_cartpole_walkthrough.ipynb`, tests `deep_q`

### Phase 2b : Rainbow DQN sur CartPole ✅

**But** : Etendre proprement la famille `deep_q/` avec une variante plus complete et plus moderne, sans casser l'histoire pedagogique DQN -> Double DQN -> Rainbow. Rainbow reste sur `CartPole-v1` pour conserver un notebook compact et comparable avec le notebook `02`.

**Livraisons** :
- `deep_q/agent.py` : `RainbowDQNAgent`
- `deep_q/config.py` : `RainbowDQNConfig`
- `deep_q/training.py` : `train_rainbow_dqn`
- `deep_q/network.py` : `NoisyLinear`, `DuelingQNetwork`, `CategoricalDuelingQNetwork`
- `deep_q/buffer.py` : `PrioritizedReplayBuffer`, `NStepTransitionAccumulator`
- config `configs/deep_q/rainbow_dqn_cartpole.yaml`
- notebook `02b_rainbow_dqn_cartpole_walkthrough.ipynb`

### Phase 3 : REINFORCE + baseline sur CartPole ✅

**But** : Premier policy gradient. REINFORCE vanilla puis ajout du baseline `V(s)` pour reduire la variance.

**Livraisons** : `reinforce/network.py`, `reinforce/agent.py`, `reinforce/config.py`, `reinforce/training.py`, notebook `03_reinforce_cartpole_walkthrough.ipynb`, tests `reinforce`

### Phase 4 : A2C + A2C-GAE + A3C sur HalfCheetah ✅

**But** : Actor-critic pour controle continu. A2C vs A2C-GAE, puis A3C comme extension asynchrone.

**Livraisons** : `actor_critic/network.py`, `actor_critic/buffer.py`, `actor_critic/agent.py`, `actor_critic/optim.py`, `actor_critic/config.py`, `actor_critic/training.py`, notebooks `04_a2c_halfcheetah_walkthrough.ipynb` et `05_a3c_halfcheetah_walkthrough.ipynb`, tests `actor_critic`

### Phase 5 : TRPO + PPO sur HalfCheetah ✅

**But** : Contraindre les updates de politique. TRPO via contrainte KL dure, PPO via objectif clippe.

**Livraisons** : `trust_region/agent.py`, `trust_region/config.py`, `trust_region/training.py`, notebook `06_trpo_ppo_halfcheetah_walkthrough.ipynb`, tests `trust_region`

### Phase 6 : DDPG + TD3 sur HalfCheetah ✅

**But** : Deterministic policy gradient pour actions continues, puis TD3 comme stabilisation moderne de DDPG.

**Livraisons** : `deterministic_actor_critic/`, configs associees, notebook `07_ddpg_td3_halfcheetah_walkthrough.ipynb`, tests `deterministic_actor_critic`

### Phase 7 : SAC sur HalfCheetah ✅

**But** : Entropy-regularized off-policy, avec twin critics et temperature learnable.

**Livraisons** : `sac/`, notebook `08_sac_halfcheetah_walkthrough.ipynb`, tests `sac`

### Phase 8 : Dyna-Q + Dyna-Q+ + Deep Dyna (tabular model-based) ✅

**But** : Introduire le model-based de la facon la plus pedagogique possible : apprendre un modele tabulaire de transitions / rewards puis alterner interactions reelles et planning updates.

**Livraisons cibles** :
- `dyna/model.py`
- `dyna/agent.py`
- `dyna/config.py`
- `dyna/training.py`
- notebook `09_dyna_q_dyna_q_plus_deep_dyna_walkthrough.ipynb`
- tests `dyna`

### Phase 9 : PILCO + Deep PILCO (model-based probabiliste et tres sample-efficient) ✅

**But** : Introduire explicitement la branche model-based probabiliste avant les approches deep modernes. PILCO doit etre presente comme une etape conceptuelle cle : tres sample-efficient, elegant, mais peu scalable.

**Positionnement pedagogique** :
- avant PETS
- avant MBPO
- montre pourquoi un modele probabiliste compte vraiment

**Decisions** : moment matching analytique exact (GP from-scratch, equations de Deisenroth) ; un seul package `pilco/` couvrant PILCO (GP) et Deep PILCO (BNN MC-dropout + particules) ; environnement Pendulum-v1 (PILCO analytique infaisable sur HalfCheetah — domaine de PETS/MBPO). Cœur math valide contre Monte-Carlo.

**Livraisons** (package `pilco/` autonome) :
- `kernel.py` : kernel SE/RBF avec ARD
- `gp.py` : regression GP (vraisemblance marginale, Cholesky robuste)
- `moment_matching.py` : propagation analytique fermee (mean/cov/cross-cov)
- `cost.py` : cout saturant + esperance analytique
- `policy.py` : politique RBF + saturation sin (moments fermes)
- `rollout.py` : propagation de croyance + prediction de trajectoire
- `bnn.py` : Deep PILCO (BNN MC-dropout + particules, masques correles dans le temps)
- `agent.py` : `PilcoAgent` (GP) et `DeepPilcoAgent` (BNN)
- `config.py` : `PilcoConfig`, `DeepPilcoConfig`
- `training.py` : `train_pilco`, `train_deep_pilco`
- `reporting.py`
- configs `configs/pilco/{pilco,deep_pilco}_pendulum.yaml`
- notebooks `10_pilco_walkthrough.ipynb` (fondations GP + PILCO) et `10b_deep_pilco_walkthrough.ipynb` (Deep PILCO)
- tests `tests/test_pilco.py` (36 tests, moment matching valide vs Monte-Carlo)

### Phase 10 : PETS (ensembles probabilistes + MPC/CEM) ✅

**But** : Ajouter explicitement PETS comme premiere grande etape deep model-based orientee planning. PETS doit venir avant MBPO et Dreamer.

**Positionnement pedagogique** :
- ensembles probabilistes
- estimation d'incertitude epistemique
- planning MPC avec CEM
- transition naturelle entre PILCO et MBPO

**Livraisons cibles** :
- `pets/dynamics.py` : ensemble probabiliste
- `pets/planner.py` : MPC / CEM
- `pets/agent.py`
- `pets/config.py`
- `pets/training.py`
- notebook `11_pets_halfcheetah_walkthrough.ipynb`
- tests `pets`

### Phase 11 : MBPO ✅

**But** : Ajouter une version model-based deep moderne qui combine donnees reelles et rollouts courts issus d'un modele appris. MBPO vient apres PETS, pas avant.

**Livraisons cibles** :
- `mbpo/dynamics.py`
- `mbpo/agent.py`
- `mbpo/config.py`
- `mbpo/training.py`
- notebook `12_mbpo_halfcheetah_walkthrough.ipynb`
- tests `mbpo`

### Phase 12 : Dreamer ✅

**But** : Apprendre un monde latent et optimiser une politique via imagination. Dreamer doit etre positionne comme la suite naturelle de PETS/MBPO vers les world models latents.

**Livraisons cibles** :
- `dreamer/rssm.py`
- `dreamer/agent.py`
- `dreamer/config.py`
- `dreamer/training.py`
- notebook `13_dreamer_walkthrough.ipynb`
- tests `dreamer`

### Phase 13 : MuZero (capstone separe) ✅

**But** : Faire de MuZero un capstone distinct, pas une simple extension du pipeline HalfCheetah. Le bon point de depart est un environnement discret simple, avec espace d'actions fini et dynamique conceptuellement lisible.

**Contraintes de positionnement** :
- sous-package autonome : `muzero/`
- environnement principal : `CartPole-v1`
- demonstration secondaire : `connect_four_v3` via PettingZoo
- **pas** `HalfCheetah-v5`

**Livraisons** :
- `muzero/config.py`
- `muzero/networks.py`
- `muzero/mcts.py`
- `muzero/replay.py`
- `muzero/agent.py`
- `muzero/training.py`
- `muzero/connect_four.py`
- `muzero/reporting.py`
- configs `configs/muzero/`
- notebook `14_muzero_discrete_control_walkthrough.ipynb`
- tests `muzero`

### Phase 15 : Action-JEPA (capstone bonus world model latent) ✅

**But** : Relier les world models RL aux architectures JEPA recentes : apprendre un latent predictif action-conditionne sans reconstruction, surveiller le collapse, puis planifier dans ce latent avec CEM/MPC.

**Contraintes de positionnement** :
- sous-package autonome : `action_jepa/`
- environnement principal : `Pendulum-v1`
- demonstration secondaire : micro `HalfCheetah-v5`
- pas d'import depuis `sac/`, `td3/`, `dreamer/` ou `muzero/`

**Livraisons** :
- `action_jepa/config.py`
- `action_jepa/networks.py`
- `action_jepa/buffer.py`
- `action_jepa/planner.py`
- `action_jepa/agent.py`
- `action_jepa/training.py`
- `action_jepa/reporting.py`
- configs `configs/action_jepa/`
- notebook `15_action_jepa_walkthrough.ipynb`
- tests `action_jepa`

### Phase 16 : Notebook de comparaison finale

**Livraisons cibles** :
- `comparison.py` - `load_all_runs(root)` et helpers d'agregation
- notebook `16_comparison.ipynb`

**Comparaisons souhaitees** :
- CartPole tabulaire : Q-learning vs SARSA
- CartPole deep : DQN vs Double DQN vs Rainbow
- HalfCheetah continu : A2C / A2C-GAE / TRPO / PPO / DDPG / TD3 / SAC
- Model-based : Dyna-Q / Dyna-Q+ / PILCO / PETS / MBPO / Dreamer
- MuZero a part, comme chapitre capstone, pas force dans les memes courbes que HalfCheetah

## Dependances entre phases

```text
Phase -1 (Infrastructure)
    ├── Phase 0 (Q-learning / SARSA)
    ├── Phase 1 (DQN / Double DQN)
    │       └── Phase 2b (Rainbow DQN)
    ├── Phase 3 (REINFORCE)
    │       └── Phase 4 (A2C / A2C-GAE / A3C)
    │               └── Phase 5 (TRPO / PPO)
    ├── Phase 6 (DDPG / TD3)
    │       └── Phase 7 (SAC)
    ├── Phase 8 (Dyna-Q / Dyna-Q+)
    ├── Phase 9 (PILCO)
    │       └── Phase 10 (PETS)
    │               └── Phase 11 (MBPO)
    │                       └── Phase 12 (Dreamer)
    ├── Phase 13 (MuZero capstone separe)
    └── Phase 15 (Action-JEPA capstone bonus)

Toutes les familles -> Phase 16 (Comparaison finale), sauf MuZero et Action-JEPA qui peuvent rester en chapitres capstone distincts.
```

## Modules existants reutilises

| Fichier actuel | Destination | Disposition |
|---|---|---|
| `artifacts.py` | reste en place | chemins d'experiences, checkpoints, sauvegardes |
| `config.py` | reste en place / etendu | registre central et chargement YAML |
| `normalization.py` | reste en place | normalisation observations pour algos continus |
| `benchmark.py` | reste en place | benchmark multi-seed pour algos continus |
| `tabular/` | deja en place | base pedagogique tabulaire |
| `deep_q/` | deja en place | famille DQN actuelle + futur Rainbow |
| `reinforce/` | deja en place | policy gradient simple |
| `actor_critic/` | deja en place | A2C, A2C-GAE, A3C |
| `trust_region/` | deja en place | TRPO, PPO |
| `deterministic_actor_critic/` | deja en place | DDPG, TD3 |
| `sac/` | deja en place | SAC |

## pyproject.toml

```toml
dependencies = [
    "gymnasium>=1.0",
    "gymnasium[box2d]>=1.0",
    "gymnasium[mujoco]>=1.0",
    "matplotlib>=3.7",
    "numpy>=1.24",
    "PyYAML>=6.0",
    "torch>=2.0",
    "pandas>=2.0",
]
```

## Verification (a chaque phase)

1. `pytest -q`
2. `python -m compileall -q src tests`
3. `python3 /Users/captain/Documents/Projects/_publication/validate_publication.py /Users/captain/Documents/Projects/RL_from_scratch`
4. CLI smoke : `python -m rl_from_scratch.train --config configs/<algo>/<smoke>.yaml`
5. Notebooks sans outputs parasites commites
