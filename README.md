# Can't Stop Playing Bot

A reinforcement-learning agent for the dice game *Can't Stop*. The training
algorithm (REINFORCE with self-play) is written by hand; PyTorch is used only
for tensors, autograd and `nn` layers. The game rules are in
[RULES.md](RULES.md) and the milestone status is in [PLAN.md](PLAN.md).

## Setup

Python 3.10+ (developed on 3.12). From the project folder:

```
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,train]"
```

`dev` installs pytest and `train` installs PyTorch (needed for the bot).

## Play a game

You are Player 0 and move first. At each prompt, type the number of the move
you want, or `q` to quit.

```
# Against a random bot
.venv/Scripts/python -m cantstop.play

# Against the trained model (a checkpoint is included in the repo)
.venv/Scripts/python -m cantstop.play --model models/reinforce_seed0.pt

# Two humans at one keyboard
.venv/Scripts/python -m cantstop.play --opponent human
```

Options:

| Option | Effect |
|---|---|
| `--model PATH` | Play against a saved network instead of the random bot |
| `--greedy` | The model always picks its most likely move (a little stronger) instead of sampling |
| `--seed N` | Fix the dice so the same game can be replayed |

### How a turn works

Each turn you roll four dice, split them into two pairs, and advance the
columns whose numbers match the pair sums. Temporary "runners" (up to 3 per
turn) mark your progress. After each successful roll you choose **Continue**
(roll again) or **Stop** (bank your progress). If no pairing is playable you
**bust** and lose the turn's progress. Reach the top of a column and stop to
capture it; the first player to capture 3 columns wins. See
[RULES.md](RULES.md) for the full rules.

The board shows, for each column, its length, each player's banked steps,
your current runner, and who has captured it.

## Train the bot

A trained model is included at `models/reinforce_seed0.pt` (REINFORCE,
seed 0, 600 iterations, about 65-70% against the strongest hand-written bot),
so you can play it straight after cloning. Other checkpoints you train go in
`runs/`, which is git-ignored. To train your own, which takes roughly 6
minutes on a CPU:

```
.venv/Scripts/python -m cantstop.reinforce --iters 600 --games 16 --seed 0 --save runs/my_model.pt
```

Every `--eval-every` iterations (default 10) it prints win rates against the
random bot and the `heur cp.6` bot, plus the KL divergence between the policy
now and at the previous evaluation, as a measure of how much its behaviour is
still changing. Add `--stop-kl X` to stop early once that stays below `X`.

## Compare bots

```
.venv/Scripts/python -m cantstop.tournament --games 200
```

Plays every baseline bot against every other and prints a table of win rates.

## Run the tests

```
.venv/Scripts/python -m pytest
```

## Project layout

| File | Purpose |
|---|---|
| `src/cantstop/state.py`, `dice.py`, `actions.py`, `engine.py` | The game engine: state, dice, legal actions, rolling and applying moves |
| `src/cantstop/env.py` | Training interface: action numbering, legal-move mask, observation encoding, `reset`/`step` |
| `src/cantstop/play.py` | Terminal game |
| `src/cantstop/bots.py` | Baseline bots (random, stop-after-N, column-progress, heuristic) |
| `src/cantstop/tournament.py` | Bot-vs-bot evaluation with confidence intervals |
| `src/cantstop/model.py`, `rollout.py`, `reinforce.py` | Policy network, self-play game collection, REINFORCE training |
| `models/` | Trained checkpoint(s) you can play against |
| `src/cantstop/convergence.py` | KL divergence between policies on a fixed set of positions |
