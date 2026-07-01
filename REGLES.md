# Règles — `RL_from_scratch`

Charte du projet. `RL_from_scratch` est un cours de RL « from scratch » :
chaque algorithme est un **chapitre autonome** (notebook pédagogique + code
packagé), lisible dans un an comme un cours accompagné de code.

L'objectif n'est pas un framework de production. C'est un code qu'un lecteur
intelligent peut relire et comprendre, et dont **on peut prouver qu'il
apprend**.

> **Différence avec `machine_learning_from_scratch`.** Là-bas le projet est
> *cumulatif* (les briques se réutilisent d'un modèle à l'autre). **Ici, chaque
> algorithme est indépendant** : duplication locale assumée, **aucun import
> inter-algorithme** (seulement `core.*`). Ne pas importer la logique de
> réutilisation du projet ML.

---

## 1. Principe directeur

Un chapitre est réussi seulement si **les trois** sont vrais :

1. **Il enseigne** : le notebook explique le problème, l'intuition, les
   équations et les choix ; le code source reprend les mêmes blocs.
2. **Il apprend** : le modèle s'améliore réellement et on en montre la preuve
   (courbe d'apprentissage). Un modèle qui n'apprend pas = échec, peu importe
   que le code « tourne ».
3. **Il reste propre** : lisible de haut en bas, comparable aux autres méthodes,
   sans casser les chapitres existants.

La **lisibilité prime sur la factorisation maximale**. Une duplication locale
est acceptable si elle rend un algorithme plus autonome et plus lisible.

**Apprentissage réel, pas décoratif (règle absolue).** Le défaut le plus grave
dans ce projet n'est pas un code laid : c'est un chapitre où le modèle
n'apprend pas (config ridicule, notebook jamais exécuté). On ne livre jamais un
chapitre dont on n'a pas vu la courbe monter (ou, pour les méthodes
notoirement dures, dont on n'a pas honnêtement constaté et documenté le
résultat — voir §7).

---

## 2. Frontière `core` vs algorithme

```text
Si ça explique l'algorithme           -> dossier de l'algorithme.
Si ça sert à faire tourner, mesurer,  -> core/.
sauvegarder, comparer, brancher
```

```text
core/                         <algo>/
    base.py                       config.py
    config.py                     network.py
    env.py                        buffer.py
    artifacts.py                  agent.py
    recording.py                  training.py
    metrics.py                    reporting.py
    reporting.py
    schedules.py
    normalization.py
    utils.py
```

Tous les fichiers `<algo>/` ne sont pas obligatoires (un tabulaire n'a pas de
`network.py`, une méthode simple pas de `buffer.py`).

---

## 3. Contenu de `core/`

`core/` est la plomberie **neutre**, partagée, qui ne connaît aucun algorithme
concret (pas de détails PPO/SAC/Dyna/PILCO dedans).

- **`base.py`** — contrats minimaux `BaseConfig`, `BaseAgent` ; interfaces
  petites et stables.
- **`config.py`** — registre des configs et des fonctions d'entraînement,
  chargement YAML, construction d'agents. Ajouter une méthode = un sous-package
  + une config, **sans toucher** aux anciennes.
- **`env.py`** — interface environnement : `make_env`, dimensions obs/action,
  `EnvSpec`/`ObservationSpec`/`ActionSpec`, discret vs continu, bornes,
  clipping/conversion d'action, **wrappers d'env** (ex. horizon fixe sans
  terminaison, cf. §8). Aucune logique d'apprentissage.
- **`artifacts.py`** — disque : dossiers de run, chemins checkpoints/figures/
  métriques/config, sauvegarde JSON, pruning des checkpoints.
- **`recording.py`** — `RunRecorder`/`RunManager` : historique d'épisodes et
  d'évals, métriques d'update, meilleur checkpoint, checkpoint périodique,
  finalisation, export `history.json`/`metrics.json`.
- **`metrics.py`** — vocabulaire standard (cf. §11), helpers de résumé.
- **`reporting.py`** — primitives de figures génériques (reward, éval, pertes)
  headless. Les plots spécifiques restent dans `<algo>/reporting.py` mais
  s'appuient dessus.
- **`schedules.py`** — cadences (éval tous N épisodes/timesteps, checkpoint,
  update, vidéo). Pas de condition de cadence recopiée en dur.
- **`normalization.py`** — `RunningMeanStd`, normalisation obs (et plus tard
  actions/rewards/deltas).
- **`utils.py`** — seeds, device, moyenne mobile, conversions. Si un helper
  porte une logique d'algorithme, il retourne dans le dossier de l'algorithme.

---

## 4. Contenu d'un dossier d'algorithme

Un dossier doit se lire comme un **chapitre autonome**.

- **`config.py`** — dataclass d'hyperparamètres, **explicite et validée**,
  proche des notations du notebook, enregistrée via `@register_config`.
- **`network.py`** — réseaux/tables/modèles de la méthode (duplication locale
  OK si elle rend la méthode autonome).
- **`buffer.py`** — structures de données propres (replay, rollout, dataset de
  modèle). Le buffer fait partie de la compréhension de l'algorithme.
- **`agent.py`** — le **cœur mathématique** : `select_action`,
  `store_transition` si utile, calcul des targets, pertes, update,
  diagnostics, `save`/`load`. **Ne gère pas** figures, dossiers de run,
  benchmark, vidéo, YAML, CLI.
- **`training.py`** — collecte d'expérience lisible (`train_one_episode`,
  `evaluate`) + boucle haut niveau. Doit refléter l'algorithme : si je prends
  le pseudocode du notebook, je retrouve les mêmes blocs.
- **`reporting.py`** — figures spécifiques, bâties sur `core/reporting.py`.

**Ordre canonique de `training.py`** (le seul endroit où il est décrit) :

```text
1. apply_overrides           5. construire l'agent (build_agent)      9. éval périodique
2. set_all_seeds             6. RunManager.from_config(...)          10. finalize_run(...)
3. make_env (+ wrappers)     7. boucle principale                   11. return {agent, history,
4. get_env_info / dims       8. learn_step au bon moment                       metrics, paths}
```

La boucle montre l'algorithme — pas la plomberie (JSON, figures, checkpoints →
`core`).

---

## 5. Indépendance des algorithmes (règle dure)

Les dossiers d'algorithmes **ne dépendent jamais** les uns des autres.

```python
# Autorisé
from rl_from_scratch.core.env import make_env
from rl_from_scratch.core.recording import RunManager
# Interdit
from rl_from_scratch.sac.agent import SACAgent
from rl_from_scratch.actor_critic.buffer import RolloutBuffer
```

Si deux algorithmes se ressemblent : (1) duplication locale claire, ou (2)
extraction dans `core/` **seulement** si la brique est vraiment neutre. Jamais
d'héritage inter-méthodes pour économiser des lignes au prix de la clarté.

---

## 6. Algorithme / environnement

Séparer le moteur de la paramétrisation d'action quand c'est naturel
(REINFORCE/A2C/PPO/TRPO : policy catégorielle sur CartPole, gaussienne sur
HalfCheetah). Pour les méthodes liées à un type d'action, **assumer la
contrainte** dans la config et la doc (DQN discret ; DDPG/TD3/SAC continus ;
une variante discrète serait un autre chapitre). Ne pas bâtir un framework
opaque pour supporter tous les espaces.

---

## 7. Configs qui font apprendre (garde-fou central)

C'est la règle qui aurait évité l'échec des chapitres model-based (PILCO à
`episodes: 8`, Deep Dyna à 50 épisodes).

- **Une config n'est valide que si elle fait démontrablement apprendre.** Le
  budget (épisodes/timesteps/itérations) est dimensionné **à la tâche**, calé
  sur les méthodes model-free comparables, jamais « pour cocher une case ».
- **Deux configs distinctes, clairement nommées :**
  - *config réelle* (ex. `<algo>_<env>.yaml`) : **doit apprendre** ; c'est elle
    qui produit la courbe de preuve.
  - *config smoke* (courte) : pour les tests/CI ; n'a pas à apprendre, mais doit
    tourner et retourner le contrat standard.
- **Critère de fin** = une **courbe d'apprentissage persistée qui s'améliore**
  (récompense d'éval ↑ / coût ↓), reproductible depuis la config + la seed
  committées.
- **Clause d'honnêteté (méthodes dures : PILCO, Deep PILCO).** Certaines
  méthodes sont notoirement difficiles à reproduire (aucune implémentation
  publique de PILCO/Deep PILCO ne reproduit fidèlement les papiers). On vise une
  amélioration réelle ; si elle reste **partielle**, on la **documente
  honnêtement** (résultat, hypothèses de cause, limites). **Jamais** de courbe
  truquée ni de succès prétendu.

---

## 8. Fidélité à l'environnement (surtout model-based)

L'environnement n'est pas un détail : un mauvais réglage d'env empêche
l'apprentissage.

- **Hypothèses de l'algorithme respectées explicitement.** Si la méthode suppose
  un **horizon fixe sans terminaison précoce** (PILCO), le garantir via un
  wrapper `core/env.py` et l'**expliquer** dans le notebook (ce n'est pas un
  départ avantageux, juste un rollout d'horizon fixe).
- **Encodage d'état adapté** : encoder l'angle en `(cos, sin)` quand la
  discontinuité ±π peut casser un modèle ; sinon justifier l'angle brut.
- **Distribution d'état initial alignée sur le reset réel de l'env.** Ne pas
  bricoler un état initial avantageux ; `mu0`/`sigma0` doivent refléter la vraie
  distribution de reset.
- **Changer d'environnement se justifie** dans le notebook (ex. passage du
  CartPole-v1 discret au `InvertedPendulum` continu : même système physique,
  action continue → vraie comparaison discret/continu).

---

## 9. Notebooks

Le notebook est le **cours principal** de la méthode : un lecteur le lit comme
un chapitre autonome (problème → intuition → équations → composants →
implémentation exécutable), puis va dans `src/` pour la version packagée.
**From scratch : aucun import depuis `rl_from_scratch`.**

Rythme imposé : `Concept → Code → Illustration → Interprétation`. Si un concept
corrige une erreur, montrer l'erreur, puis la correction et son effet.

### Qualité, pas quantité (règle absolue)

Un notebook n'est **jamais** jugé au nombre de cellules ou au volume de
markdown. Le seul critère est la **valeur réelle de chaque cellule**, jugée
cellule par cellule :

- chaque cellule de texte apporte une idée utile, expliquée en profondeur — pas
  de remplissage, pas de paraphrase du code ; si on peut la supprimer sans rien
  perdre, elle ne doit pas exister ;
- chaque cellule de code est lisible et illustre exactement le concept du
  moment, avec un bon exemple choisi (pas un test trivial) ;
- chaque visuel est suivi d'une **interprétation réelle** ;
- les **sources sont citées** (article fondateur, implémentation de référence).

### Structure recommandée

1. **Problème et motivation** — ce que la méthode résout, la limite qu'elle
   corrige, sa famille. 2. **Intuition** — idée en mots simples, analogie si
   utile, rôle de chaque brique avant les équations. 3. **Environnement** —
   obs/action/reward/fin d'épisode, mini épisode aléatoire pour voir l'entrée
   (et justification de l'env, cf. §8). 4. **Équations** — équations centrales,
   chaque terme expliqué, une idée par cellule. 5. **Composants** — blocs dans
   l'ordre de compréhension, chacun suivi d'un mini-test/shape. 6. **Pseudocode**
   — **après** les composants, préfigure la boucle. 7. **Training** — boucle
   lisible (collecte/update/éval/log séparés). 8. **Diagnostics** — courbes +
   interprétation par panneau. 9. **Démo finale** — montrer ce que l'agent a
   appris, relié aux courbes. 10. **Limites** — forces, faiblesses, pont vers la
   méthode suivante. 11. **Lien avec `src/`** — où est la version packagée (sans
   l'importer).

### Preuve d'apprentissage + compatibilité avec la barrière de publication

Tension réelle à respecter : le **validateur de publication interdit tout output
de cellule** (`outputs` non vide ou `execution_count` non nul ⇒ FAIL ; gate
obligatoire, jamais contournée). On concilie ainsi :

- le notebook est **exécuté de bout en bout** en développement, et **le modèle
  doit apprendre** (sinon le chapitre n'est pas terminé) ;
- la **preuve** vit dans des **figures PNG committées** (dans le dossier de
  figures du run) **référencées en markdown** (`![](figures/...png)`) — elles
  s'affichent dans le notebook rendu et **ne sont pas des outputs de cellule** ;
- **avant commit** : outputs de cellule nettoyés, `execution_count` à `null`
  (pour passer le validateur) ;
- conclusion **honnête** : si la courbe ne descend/monte pas autant qu'espéré,
  le dire et expliquer (cf. §7).

### Code dans les notebooks

Cellules digestes (viser 80–120 lignes) ; préférer plusieurs petites cellules
(markdown + code + test) à une grosse cellule opaque ; noms proches des
équations ; vérifier les shapes ; mini-test après chaque classe importante ;
pas de code silencieux ; **pas de `sys.path.append`/`insert`**.

---

## 10. Code source

Version packagée du cours, propre mais **pas sur-abstraite**. Un fichier doit
répondre vite à : que fait-il ? quel rôle dans l'algorithme ? où aller ensuite ?

- noms explicites proches des équations (`advantages`, `td_errors`, `log_probs`,
  `target_q`...), jamais `x`/`tmp`/`stuff`/`data` ;
- code dans le même ordre mental que le notebook ; commentaire d'une ligne avant
  un bloc mathématique dense ;
- pas de « magic helper » qui cache la logique ; **pas de dépendance
  inter-algos** ; pas de mutation globale cachée ; seeds explicites ;
- métriques numériques **finies** seulement (convertir les tenseurs en floats,
  refuser NaN/inf) ;
- **bonne vs mauvaise abstraction** : une bonne supprime une plomberie répétée,
  porte un nom évident, ne cache pas les équations ; une mauvaise oblige à
  sauter entre cinq fichiers pour comprendre une étape. En cas de doute,
  dupliquer localement.

**Responsabilités par fichier** : `config.py` (hyperparams+validation) ·
`network.py` (réseaux/tables) · `buffer.py` (stockage) · `agent.py`
(action/loss/update/diag/save-load) · `training.py` (collecte+orchestration) ·
`reporting.py` (figures). `agent.py` ne crée pas de dossiers/figures/vidéos/CLI ;
`training.py` ne contient pas le détail des pertes ni de gros plotting.

**Contrat public** — toute `training.py` retourne :

```python
{"agent": agent, "history": history, "metrics": summary, "paths": paths}
```

Utilisé par CLI, benchmark, notebooks, comparaisons. Ne pas le casser sans
migration.

**Reproductibilité** : seeds Python/NumPy/PyTorch, env, action space, éval ;
device ; config sauvegardée ; **évaluation déterministe** pour le benchmark.
**Save/load** géré par chaque agent (réseaux, optimiseurs, normalizers,
alpha…).

**Tests attendus** (proportionnels au risque) : config valide/refuse l'invalide ;
networks → bonnes shapes ; buffers → round-trip ; losses → scalaires finis ;
agent → action valide ; `learn_step` sur petit batch ; **smoke training** →
`{agent, history, metrics, paths}` ; éval déterministe. Les tests protègent le
**comportement**, pas seulement l'implémentation. Pour les méthodes du §7,
ajouter un test que la métrique d'apprentissage **s'améliore** sur un run court
(coût ↓ / reward ↑).

---

## 11. Reporting et benchmark

Rendre les runs comparables sans changer la logique des algorithmes.

**Clés communes de `history`** : `episode_rewards`, `episode_lengths`,
`eval_steps`, `eval_timesteps` (si budget en timesteps), `eval_mean_rewards`,
`eval_std_rewards`, `eval_min_rewards`, `eval_max_rewards`.

**`metrics`** : résumé court (nb épisodes/timesteps, reward finale/moyenne,
meilleure reward, moyenne des derniers épisodes, meilleure/dernière éval, seed).

**Métriques spécifiques** encouragées si elles aident (noms stables, finis, pas
de tenseurs bruts) : `step_actor_losses`, `step_kls`, `step_alphas`,
`step_model_losses`, `step_prediction_errors`…

**Figures** : au minimum reward + éval + pertes/diagnostics principaux ; chaque
figure a un rôle (pas dix plots décoratifs). **Benchmark** : éval déterministe
obligatoire, seeds explicites, protocole sauvegardé ; ne pas comparer des
conventions d'éval incompatibles sans le signaler. Outil de comparaison
**honnête**, pas un palmarès.

---

## 12. SOLID (rappel court)

- **SRP** : `agent` apprend/agit ; `training` collecte/orchestre ; `recording`
  enregistre ; `reporting` visualise.
- **Open/Closed** : ajouter une méthode = ajouter un dossier + une config, pas
  modifier l'existant.
- **Liskov** : tout agent respecte `BaseAgent` ; pas de sous-type fragile
  supposé.
- **Interface Segregation** : un reporter ne connaît pas le training ; un agent
  ne connaît pas matplotlib.
- **Dependency Inversion** : les algos dépendent de `core`, jamais d'un autre
  algo concret.

---

## 13. Workflow pour ajouter / refaire une méthode

```text
comprendre -> situer -> design pédagogique -> implémenter -> FAIRE APPRENDRE -> vérifier -> documenter
```

1. **Lire le contexte** : ce document, `Objectif.md`, README, notebooks voisins,
   méthodes proches, configs, tests.
2. **Situer** la méthode (famille, limite corrigée, env de démo, métriques de
   succès).
3. **Design pédagogique avant le code** : intuition, équations, composants,
   ordre de construction, diagnostics, limites.
4. **Implémenter** le dossier autonome (config → composants → agent →
   `training.py`), chaque bloc testable en isolation, **seulement** des imports
   `core.*`.
5. **Config + FAIRE APPRENDRE** : écrire la config réelle, **lancer
   l'entraînement, vérifier que la courbe s'améliore** (§7) ; écrire aussi une
   config smoke pour la CI.
6. **Brancher** métriques/reporting/benchmark (clés communes du §11).
7. **Tests** : unitaires + smoke + (méthodes §7) test d'amélioration.
8. **Notebook** : cours complet (§9), exécuté, figures de preuve référencées,
   outputs nettoyés avant commit.
9. **Aligner** source ↔ notebook (mêmes noms, mêmes blocs) ; mettre à jour
   `Objectif.md`/README/roadmap sans la laisser mentir.

---

## 14. Vérification minimale

Dans le venv (`uv`, jamais le Python système) :

```bash
pytest -q
python -m compileall -q src tests
python3 /Users/captain/Documents/Projects/_publication/validate_publication.py \
    /Users/captain/Documents/Projects/RL_from_scratch
```

- Le validateur doit sortir **0 FAIL** — gate obligatoire, **jamais
  `--no-verify`, jamais l'éditer/désactiver, jamais d'attribution AI** dans les
  commits. **Aucun commit sans demande explicite de l'utilisateur.**
- **Critère d'apprentissage** : pour la méthode concernée, un run réel montre la
  courbe qui s'améliore (ou un résultat honnêtement documenté, §7).
- Notebook : JSON valide ; outputs de cellule nettoyés (`execution_count` null) ;
  preuve via figures committées référencées ; pas de hack `sys.path` ; pas
  d'import package.

---

## 15. Règle finale

Quand deux choix sont possibles, choisir celui qui rend le projet plus facile à
**relire, expliquer, vérifier — et prouver qu'il apprend**. La meilleure
architecture n'est pas celle qui minimise les lignes : c'est celle où chaque
algorithme est évident, chaque expérience reproductible, chaque résultat
honnête.
