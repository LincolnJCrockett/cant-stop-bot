import numpy as np
import pytest
import torch

from cantstop.actions import legal_actions
from cantstop.env import NUM_ACTIONS, OBS_SIZE, CantStopEnv, index_to_action
from cantstop.model import PolicyNet, make_policy, masked_log_probs
from cantstop.reinforce import compute_returns, reinforce_loss, train
from cantstop.rollout import play_episode, to_tensors
from cantstop.state import new_game
from cantstop.tournament import play_one_game


# --- returns --------------------------------------------------------------
def test_returns_are_plus_one_for_winner_minus_one_for_loser():
    assert list(compute_returns([0, 1, 0, 1, 1], winner=0)) == [1, -1, 1, -1, -1]
    assert list(compute_returns([0, 1, 0, 1, 1], winner=1)) == [-1, 1, -1, 1, 1]


# --- masked log-probabilities --------------------------------------------
def test_masked_probs_are_zero_for_illegal_and_sum_to_one():
    logits = torch.randn(3, NUM_ACTIONS)
    mask = torch.zeros(3, NUM_ACTIONS, dtype=torch.bool)
    mask[0, [4, 9]] = True
    mask[1, 0] = True
    mask[2, 10:20] = True
    probs = masked_log_probs(logits, mask).exp()
    assert torch.all(probs[~mask] == 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3))
    assert probs[1, 0] == 1.0                     # only one legal action


# --- the loss -------------------------------------------------------------
def one_decision(net, action, ret, mask_indices=(3, 7, 11)):
    obs = torch.rand(1, OBS_SIZE)
    mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    mask[0, list(mask_indices)] = True
    return (obs, mask, torch.tensor([action]), torch.tensor([ret], dtype=torch.float32))


def prob_of(net, obs, mask, action):
    return masked_log_probs(net(obs), mask)[0, action].exp().item()


@pytest.mark.parametrize("ret, expect_up", [(1.0, True), (-1.0, False)])
def test_gradient_step_raises_or_lowers_probability_by_sign_of_return(ret, expect_up):
    torch.manual_seed(0)
    net = PolicyNet()
    obs, mask, action, g = one_decision(net, action=7, ret=ret)
    before = prob_of(net, obs, mask, 7)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    loss = reinforce_loss(net, obs, mask, action, g)
    opt.zero_grad(); loss.backward(); opt.step()
    after = prob_of(net, obs, mask, 7)
    assert (after > before) if expect_up else (after < before)


def test_loss_gradients_are_finite_despite_masked_actions():
    torch.manual_seed(0)
    net = PolicyNet()
    obs, mask, action, g = one_decision(net, action=3, ret=1.0)
    loss = reinforce_loss(net, obs, mask, action, g)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_zero_returns_give_zero_gradient():
    torch.manual_seed(0)
    net = PolicyNet()
    obs, mask, action, g = one_decision(net, action=3, ret=0.0)
    reinforce_loss(net, obs, mask, action, g).backward()
    assert all((p.grad == 0).all() for p in net.parameters())


# --- rollout --------------------------------------------------------------
def test_episode_records_are_consistent_and_legal():
    torch.manual_seed(0)
    net = PolicyNet()
    ep = play_episode(net, CantStopEnv(), seed=1)
    n = len(ep)
    assert n > 0 and len(ep.observations) == len(ep.masks) == len(ep.players) == n
    assert ep.winner in (0, 1)
    for mask, action in zip(ep.masks, ep.actions):
        assert mask[action]                       # chosen action was legal


def test_episode_is_reproducible_from_seed():
    torch.manual_seed(0)
    net = PolicyNet()
    a = play_episode(net, CantStopEnv(), seed=5)
    b = play_episode(net, CantStopEnv(), seed=5)
    assert a.actions == b.actions and a.winner == b.winner


def test_batch_tensor_shapes():
    torch.manual_seed(0)
    net = PolicyNet()
    eps = [play_episode(net, CantStopEnv(), seed=s) for s in range(3)]
    rets = [compute_returns(e.players, e.winner) for e in eps]
    obs, masks, actions, g = to_tensors(eps, rets)
    total = sum(len(e) for e in eps)
    assert obs.shape == (total, OBS_SIZE) and masks.shape == (total, NUM_ACTIONS)
    assert actions.shape == g.shape == (total,)
    assert masks.dtype == torch.bool and actions.dtype == torch.int64


# --- policy adapter / smoke test -----------------------------------------
def test_network_policy_only_plays_legal_moves_in_the_harness():
    torch.manual_seed(0)
    policy = make_policy(PolicyNet())
    rng = np.random.default_rng(0)
    from cantstop.engine import roll_until_decision, step
    s = roll_until_decision(new_game(), rng)
    while not s.is_over:
        a = policy(s, rng)
        assert a in legal_actions(s)
        s = step(s, a, rng)
    assert play_one_game([policy, policy], 3) in (0, 1)


def test_training_runs_and_changes_the_weights():
    torch.manual_seed(0)
    before = [p.clone() for p in PolicyNet().parameters()]
    net = train(iters=2, games_per_iter=2, eval_every=2, eval_games=2, seed=0)
    assert any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))
