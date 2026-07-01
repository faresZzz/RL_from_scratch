import pytest

from rl_from_scratch.core.config import CONFIG_REGISTRY
from rl_from_scratch.tabular.config import QLearningConfig
from tests.conftest import get_bin_counts, get_config_value, make_config


def test_q_learning_config_is_registered_in_core_registry():
    assert CONFIG_REGISTRY["q_learning"] is QLearningConfig


def test_q_learning_config_exposes_portfolio_defaults():
    config = make_config()

    assert get_config_value(config, "alpha") == pytest.approx(0.1)
    assert get_config_value(config, "gamma") == pytest.approx(1.0)
    assert get_config_value(config, "epsilon") == pytest.approx(0.2)
    assert get_config_value(
        config, "num_episodes", "episodes", "number_of_epoch"
    ) == 15_000
    assert get_bin_counts(config) == (30, 30, 30, 30)


@pytest.mark.parametrize(
    ("override_name", "override_value"),
    [
        ("epsilon", 1.5),
        ("alpha", -0.01),
        ("bins", (30, 0, 30, 30)),
    ],
)
def test_q_learning_config_rejects_invalid_values(override_name, override_value):
    with pytest.raises((TypeError, ValueError)):
        make_config(**{override_name: override_value})
