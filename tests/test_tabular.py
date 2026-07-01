import numpy as np
import pytest
import torch

from tests.conftest import (
    call_learn,
    call_sarsa_learn,
    get_bin_counts,
    get_q_table,
    make_agent,
    make_sarsa_agent,
    select_epsilon_action,
    select_greedy_action,
    select_random_action,
)


def test_q_learning_agent_initializes_q_table_with_expected_shape(small_config):
    agent = make_agent(config=small_config, num_actions=2)

    assert get_q_table(agent).shape == get_bin_counts(small_config) + (2,)


def test_q_learning_agent_action_selection_stays_within_action_bounds(small_config):
    agent = make_agent(config=small_config, num_actions=2)
    state = (0, 0, 0, 0)

    # Directly set values on the torch tensor
    agent.q_table[state + (0,)] = 0.0
    agent.q_table[state + (1,)] = 5.0

    random_action = select_random_action(agent)
    greedy_action = select_greedy_action(agent, state)
    epsilon_action = select_epsilon_action(agent, state)

    assert random_action in {0, 1}
    assert greedy_action == 1
    assert epsilon_action in {0, 1}


def test_q_learning_agent_learn_updates_exactly_one_q_value(small_config):
    agent = make_agent(config=small_config, num_actions=2)
    agent.q_table.fill_(0.0)

    state = (1, 2, 3, 4)
    next_state = (2, 3, 4, 5)
    action = 1
    reward = 1.0

    before = agent.q_table.clone().numpy()
    call_learn(
        agent,
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=False,
    )
    after = get_q_table(agent)

    changed_indices = np.argwhere(after != before)
    assert changed_indices.shape[0] == 1

    alpha = getattr(agent, "alpha", getattr(agent, "ALPHA", 0.5))
    gamma = getattr(agent, "gamma", getattr(agent, "GAMMA", 0.9))
    expected = before[state + (action,)] + alpha * (
        reward + gamma * np.max(before[next_state]) - before[state + (action,)]
    )
    assert after[state + (action,)] == pytest.approx(expected)


# =====================================================================
# SARSA Agent Tests
# =====================================================================


def test_sarsa_agent_initializes_q_table_with_expected_shape(small_config):
    agent = make_sarsa_agent(config=small_config, num_actions=2)

    assert get_q_table(agent).shape == get_bin_counts(small_config) + (2,)


def test_sarsa_agent_action_selection_stays_within_action_bounds(small_config):
    agent = make_sarsa_agent(config=small_config, num_actions=2)
    state = (0, 0, 0, 0)

    agent.q_table[state + (0,)] = 0.0
    agent.q_table[state + (1,)] = 5.0

    random_action = select_random_action(agent)
    greedy_action = select_greedy_action(agent, state)
    epsilon_action = select_epsilon_action(agent, state)

    assert random_action in {0, 1}
    assert greedy_action == 1
    assert epsilon_action in {0, 1}


def test_sarsa_agent_learn_uses_next_action_not_max(small_config):
    """SARSA must use Q[s', next_action], NOT max Q[s', :].

    Set Q[s', 0] = 1.0 and Q[s', 1] = 5.0, then call learn with
    next_action=0. The update should use bootstrap value 1.0 (on-policy),
    not 5.0 (off-policy / Q-learning).
    """
    agent = make_sarsa_agent(config=small_config, num_actions=2)
    agent.q_table.fill_(0.0)

    state = (1, 2, 3, 4)
    next_state = (2, 3, 4, 5)
    action = 1
    next_action = 0
    reward = 1.0

    # Set up asymmetric Q-values in the next state
    agent.q_table[next_state + (0,)] = 1.0
    agent.q_table[next_state + (1,)] = 5.0

    call_sarsa_learn(
        agent,
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        next_action=next_action,
        done=False,
    )

    alpha = getattr(agent, "alpha", 0.5)
    gamma = getattr(agent, "gamma", 0.9)
    # SARSA uses Q[s', next_action] = Q[s', 0] = 1.0
    expected = 0.0 + alpha * (reward + gamma * 1.0 - 0.0)
    actual = get_q_table(agent)[state + (action,)]
    assert actual == pytest.approx(expected)

    # Verify it did NOT use max (which would be 5.0)
    wrong = 0.0 + alpha * (reward + gamma * 5.0 - 0.0)
    assert actual != pytest.approx(wrong)
