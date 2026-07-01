from __future__ import annotations

from rl_from_scratch.core.schedules import (
    every_n_episodes,
    every_n_steps,
    should_record_video,
)


def test_every_n_episodes_uses_1_based_periodic_trigger() -> None:
    assert every_n_episodes(episode=1, every=5) is False
    assert every_n_episodes(episode=5, every=5) is True
    assert every_n_episodes(episode=10, every=5) is True


def test_every_n_steps_uses_positive_multiples() -> None:
    assert every_n_steps(step=0, every=100) is False
    assert every_n_steps(step=99, every=100) is False
    assert every_n_steps(step=100, every=100) is True


def test_should_record_video_disabled_when_schedule_is_missing() -> None:
    assert should_record_video(episode=10, record_every=None) is False
