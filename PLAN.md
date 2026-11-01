# Plan

Status legend: [ ] todo, [x] done

## M0 — Setup
- [x] git init
- [x] RULES.md reviewed/approved by user
- [x] Python env, pyproject, pytest

## M1 — Rules engine
- [x] State, dice, pairings, legal-action generation
- [x] Turn/bust/stop/capture/win logic
- [x] Unit tests for every edge case in RULES.md

## M2 — Environment interface
- [ ] reset/step API, legal action mask, observation encoding
- [ ] Terminal play mode (human vs bot)

## M3 — Baseline bots
- [ ] Random, "stop after N progress", heuristic
- [ ] Bot-vs-bot tournament harness

## M4 — Game records / dataset storage
Goal: store many game histories compactly and load them fast for training.
- [ ] Decide what a record contains: per-step observation (or enough to
      rebuild it), action taken, legal mask, acting player, reward/outcome,
      plus per-game metadata (seed, players/policies, winner, length)
- [ ] Choose format (leading candidate: flat NumPy arrays with an index of
      game boundaries, memory-mappable; compare against .npz / HDF5 / Parquet)
- [ ] Writer that appends games in chunks (no need to hold all in RAM)
- [ ] Reader / PyTorch-friendly loader (random step batches, whole-game
      access, train/val split by game)
- [ ] Replay check: re-simulate from stored seed + actions and verify the
      same outcome (guards against engine/format bugs)
- [ ] Generate a dataset from the baseline bots; measure size and read speed

## M5 — Training (hand-written, PyTorch for tensors/autograd only)
- [ ] REINFORCE baseline
- [ ] PPO with action masking + self-play
- [ ] Optional: supervised/imitation pretraining from stored records

## M6 — Evaluation
- [ ] Seeded win-rate evals vs baselines, learning curves
