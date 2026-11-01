# Can't Stop Playing Bot

Train an RL agent to play the dice game Can't Stop.

## Primary goal: LEARNING
The point of this project is for the user to learn how model training works.
- Do NOT use high-level RL libraries (Stable-Baselines3, RLlib, CleanRL, Gym
  wrappers that hide the training loop, etc.).
- PyTorch is allowed for tensors, autograd, and nn modules only. Write the
  algorithm itself (returns/advantages, policy gradient/PPO loss, rollout
  buffer, action masking, evaluation) by hand.
- Explain the *why* behind each piece when adding it; prefer small, readable
  implementations over clever ones. When appropriate, offer to let the user
  write a piece themselves instead of doing it for them.

## Source of truth
- `RULES.md` defines game rules. Change the spec first, then code and tests.
- `PLAN.md` holds milestones and status. Update it as work completes.

## Conventions
- Python 3.10+, package in `src/cantstop/`, tests in `tests/` (pytest).
- Engine is pure and deterministic: seedable RNG, no global state.
- Run tests: `python -m pytest`
- Commit per milestone with clear messages.
