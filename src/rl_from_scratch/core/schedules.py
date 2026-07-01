"""Small explicit periodic schedule helpers."""

from __future__ import annotations


def every_n_episodes(*, episode: int, every: int | None) -> bool:
    if every is None:
        return False
    if every <= 0:
        raise ValueError("every must be positive when set.")
    return episode > 0 and episode % every == 0


def every_n_steps(*, step: int, every: int | None) -> bool:
    if every is None:
        return False
    if every <= 0:
        raise ValueError("every must be positive when set.")
    return step > 0 and step % every == 0


def should_eval_episode(*, episode: int, eval_every: int | None) -> bool:
    return every_n_episodes(episode=episode, every=eval_every)


def should_eval_timestep(*, timestep: int, eval_every_steps: int | None) -> bool:
    return every_n_steps(step=timestep, every=eval_every_steps)


def should_checkpoint_episode(*, episode: int, checkpoint_every: int | None) -> bool:
    return every_n_episodes(episode=episode, every=checkpoint_every)


def should_checkpoint_timestep(*, timestep: int, checkpoint_every: int | None) -> bool:
    return every_n_steps(step=timestep, every=checkpoint_every)


def should_record_video(*, episode: int, record_every: int | None) -> bool:
    return every_n_episodes(episode=episode, every=record_every)
