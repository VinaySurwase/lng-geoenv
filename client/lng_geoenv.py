from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from server.models import LNGAction, LNGObservation


class LNGGeoEnv(EnvClient[LNGAction, LNGObservation, State]):


    def _step_payload(self, action: LNGAction) -> Dict:

        payload = {
            "action_type": action.action_type,
            "amount": action.amount,
        }
        if action.ship_id is not None:
            payload["ship_id"] = action.ship_id
        if action.new_route is not None:
            payload["new_route"] = action.new_route
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[LNGObservation]:

        obs_data = payload.get("observation", {})
        observation = LNGObservation(
            time_step=obs_data.get("time_step", 0),
            ships=obs_data.get("ships", []),
            blocked_routes=obs_data.get("blocked_routes", []),
            storage=obs_data.get("storage", {}),
            demand_forecast=obs_data.get("demand_forecast", []),
            price=obs_data.get("price", 0.0),
            budget=obs_data.get("budget", 0.0),
            goal=obs_data.get("goal", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:

        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
