# Plan

Status legend: [ ] todo, [x] done

## M0 — Setup
- [x] git init
- [ ] RULES.md reviewed/approved by user
- [ ] Python env, pyproject, pytest

## M1 — Rules engine
- [ ] State, dice, pairings, legal-action generation
- [ ] Turn/bust/stop/capture/win logic
- [ ] Unit tests for every edge case in RULES.md

## M2 — Environment interface
- [ ] reset/step API, legal action mask, observation encoding
- [ ] Terminal play mode (human vs bot)

## M3 — Baseline bots
- [ ] Random, "stop after N progress", heuristic
- [ ] Bot-vs-bot tournament harness

## M4 — Training (hand-written, PyTorch for tensors/autograd only)
- [ ] REINFORCE baseline
- [ ] PPO with action masking + self-play

## M5 — Evaluation
- [ ] Seeded win-rate evals vs baselines, learning curves
