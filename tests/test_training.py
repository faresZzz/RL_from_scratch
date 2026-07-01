from tests.conftest import (
    call_with_supported_kwargs,
    make_sarsa_config,
    maybe_set_config_value,
    require_sarsa_training_entrypoint,
    require_training_entrypoint,
    unwrap_result_field,
)


def test_training_smoke_run_returns_metrics_and_writes_artifacts(
    small_config, tmp_run_dir,
):
    _, train = require_training_entrypoint()

    result = call_with_supported_kwargs(
        train,
        config=small_config,
        run_dir=tmp_run_dir,
        output_dir=tmp_run_dir,
        artifacts_dir=tmp_run_dir,
        num_episodes=2,
        episodes=2,
        render=False,
        seed=0,
    )

    history = unwrap_result_field(result, "history")
    metrics = unwrap_result_field(result, "metrics")

    assert history is not None
    assert metrics is not None
    assert any(tmp_run_dir.iterdir())


def test_sarsa_training_smoke_run_returns_metrics_and_writes_artifacts(tmp_run_dir):
    config = make_sarsa_config(
        bins=(3, 4, 5, 6),
        num_bins=(3, 4, 5, 6),
        state_bins=(3, 4, 5, 6),
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        num_episodes=3,
        episodes=3,
        total_episodes=3,
        render=False,
        seed=0,
    )
    maybe_set_config_value(config, (3, 4, 5, 6), "bins", "num_bins", "state_bins")
    maybe_set_config_value(
        config, 3, "num_episodes", "episodes", "total_episodes", "number_of_epoch"
    )
    maybe_set_config_value(config, 0.5, "alpha", "ALPHA")
    maybe_set_config_value(config, 0.9, "gamma", "GAMMA")
    maybe_set_config_value(config, 0.0, "epsilon", "EPSILON")

    _, train = require_sarsa_training_entrypoint()

    result = call_with_supported_kwargs(
        train,
        config=config,
        run_dir=tmp_run_dir,
        output_dir=tmp_run_dir,
        artifacts_dir=tmp_run_dir,
        num_episodes=2,
        episodes=2,
        render=False,
        seed=0,
    )

    history = unwrap_result_field(result, "history")
    metrics = unwrap_result_field(result, "metrics")

    assert history is not None
    assert metrics is not None
    assert any(tmp_run_dir.iterdir())
