import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from server.models import LNGAction, LNGObservation
from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config


DEFAULT_CONFIG = {
    "max_steps": 10,
    "reward": {
        "w_cost": 1.0,
        "w_shortage": 6.0,
        "w_delay": 1.0,
        "w_risk": 3.0,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 2.0,
    },
}


class LNGEnvironment(Environment):


    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._env = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name = "stable"  # default task
        self._last_observation = None

    def reset(
        self,
        seed: Optional[int] = 42,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LNGObservation:
        """Reset the environment and return initial observation."""
        # Allow task selection via kwargs
        task_name = kwargs.get("task", self._task_name)
        if task_name not in ("stable", "volatile", "war"):
            task_name = "stable"
        self._task_name = task_name

        task_config = get_task_config(task_name)
        self._env = LNGEnv(config=DEFAULT_CONFIG, task_config=task_config)
        obs = self._env.reset(seed=seed or 42)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        observation = LNGObservation(
            time_step=obs.time_step,
            ships=[s.model_dump() for s in obs.ships],
            blocked_routes=obs.blocked_routes,
            storage=obs.storage.model_dump(),
            demand_forecast=obs.demand_forecast,
            price=obs.price,
            budget=obs.budget,
            goal=f"Manage LNG supply chain under '{task_name}' scenario. "
            f"Minimize shortage, cost, delay, and risk over {DEFAULT_CONFIG['max_steps']} steps.",
            done=False,
            reward=0.0,
        )
        self._last_observation = observation
        return observation

    def step(
        self,
        action: LNGAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LNGObservation:
        if self._env is None:
            self.reset(seed=42)

        # Build action dict for the internal env
        from src.lng_geoenv.models import Action as InternalAction

        internal_action = InternalAction(
            action_type=action.action_type,
            amount=action.amount,
            ship_id=action.ship_id,
            new_route=action.new_route,
        )

        obs, reward_obj, done, info = self._env.step(internal_action)

        self._state.step_count += 1

        observation = LNGObservation(
            time_step=obs.time_step,
            ships=[s.model_dump() for s in obs.ships],
            blocked_routes=obs.blocked_routes,
            storage=obs.storage.model_dump(),
            demand_forecast=obs.demand_forecast,
            price=obs.price,
            budget=obs.budget,
            goal=self._last_observation.goal if self._last_observation else "",
            done=done,
            reward=reward_obj.value,
            metadata={"metrics": info.get("metrics", {})},
        )
        self._last_observation = observation
        return observation

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="LNG-GeoEnv",
            description="Multi-agent LNG supply chain optimization with demand forecasting, "
            "route management, and dynamic reward computation under geopolitical disruptions.",
            version="0.1.0",
        )

    def close(self) -> None:
        """Clean up resources."""
        self._env = None
