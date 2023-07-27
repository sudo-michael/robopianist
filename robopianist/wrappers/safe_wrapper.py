"""Composer-related utilities."""

from dm_control import composer, mjcf
import dm_env
from dm_env import StepType
from dm_control.rl import control
from dm_control.composer.environment import _CommonEnvironment

from absl import logging

from mujoco_utils import types

from typing import Any, NamedTuple


class SafeTimeStep(dm_env.TimeStep):
    """Returned with every call to `step` and `reset` on an environment.

    A `TimeStep` contains the data emitted by an environment at each step of
    interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
    NumPy array or a dict or list of arrays), and an associated `reward` and
    `discount`.

    The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
    `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
    have `StepType.MID.

    Attributes:
      step_type: A `StepType` enum value.
      reward:  A scalar, NumPy array, nested dict, list or tuple of rewards; or
        `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
        sequence.
      discount: A scalar, NumPy array, nested dict, list or tuple of discount
        values in the range `[0, 1]`, or `None` if `step_type` is
        `StepType.FIRST`, i.e. at the start of a sequence.
      observation: A NumPy array, or a nested dict, list or tuple of arrays.
        Scalar values that can be cast to NumPy arrays (e.g. Python floats) are
        also valid in place of a scalar array.
    """

    # TODO(b/143116886): Use generics here when PyType supports them.
    step_type: Any
    reward: Any
    cost: Any
    discount: Any
    observation: Any

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST


class CMDPEnvironment(composer.Environment):
    def _reset_attempt(self):
        self._hooks.initialize_episode_mjcf(self._random_state)
        self._recompile_physics_and_update_observables()
        with self._physics.reset_context():
            self._hooks.initialize_episode(self._physics_proxy, self._random_state)
        self._observation_updater.reset(self._physics_proxy, self._random_state)
        self._reset_next_step = False
        return SafeTimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            cost=None,
            discount=None,
            observation=self._observation_updater.get_observation(),
        )

    def step(self, action):
        """Updates the environment using the action and returns a `TimeStep`."""
        if self._reset_next_step:
            self._reset_next_step = False
            return self.reset()

        self._hooks.before_step(self._physics_proxy, action, self._random_state)
        self._observation_updater.prepare_for_next_control_step()

        try:
            for i in range(self._n_sub_steps):
                self._substep(action)
                # The final observation update must happen after all the hooks in
                # `self._hooks.after_step` is called. Otherwise, if any of these hooks
                # modify the physics state then we might capture an observation that is
                # inconsistent with the final physics state.
                if i < self._n_sub_steps - 1:
                    self._observation_updater.update()
            physics_is_divergent = False
        except control.PhysicsError as e:
            if not self._raise_exception_on_physics_error:
                logging.warning(e)
                physics_is_divergent = True
            else:
                raise

        self._hooks.after_step(self._physics_proxy, self._random_state)
        self._observation_updater.update()

        if not physics_is_divergent:
            reward = self._task.get_reward(self._physics_proxy)
            cost = self._task.get_cost(self._physics_proxy)
            discount = self._task.get_discount(self._physics_proxy)
            terminating = (
                self._task.should_terminate_episode(self._physics_proxy)
                or self._physics.time() >= self._time_limit
            )
        else:
            reward = 0.0
            discount = 0.0
            cost = 0.0
            terminating = True

        obs = self._observation_updater.get_observation()

        if not terminating:
            return SafeTimeStep(dm_env.StepType.MID, reward, cost, discount, obs)
        else:
            self._reset_next_step = True
            return SafeTimeStep(dm_env.StepType.LAST, reward, cost, discount, obs)


class SafeEnvironmentWrapper(CMDPEnvironment):
    """A composer environment with functionality to skip physics recompilation."""

    def __init__(self, recompile_physics: bool = True, **base_kwargs) -> None:
        """Constructor.

        Args:
            recompile_physics: Whether to recompile the physics between episodes. When
                set to False, `initialize_episode_mjcf` and `after_compile` steps are
                skipped. This can be useful for speeding up training when the physics
                are not changing between episodes.
            **base_kwargs: `composer.Environment` kwargs.
        """
        super().__init__(**base_kwargs)

        self._recompile_physics_active = recompile_physics
        self._physics_recompiled_once = False

    def _reset_attempt(self) -> dm_env.TimeStep:
        if self._recompile_physics_active or not self._physics_recompiled_once:
            self._hooks.initialize_episode_mjcf(self._random_state)
            self._recompile_physics_and_update_observables()
            self._physics_recompiled_once = True

        with self._physics.reset_context():
            self._hooks.initialize_episode(self._physics_proxy, self._random_state)

        self._observation_updater.reset(self._physics_proxy, self._random_state)
        self._reset_next_step = False

        return SafeTimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            cost=None,
            discount=None,
            observation=self._observation_updater.get_observation(),
        )
