"""
EvalForge — client.py
Typed EnvClient for WebSocket communication with the EvalForge server.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import EvalAction, EvalObservation, EvalState


class EvalForgeEnv(EnvClient[EvalAction, EvalObservation, EvalState]):
    """Client for the EvalForge environment."""

    def _step_payload(self, action: EvalAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[EvalObservation]:
        obs_data = payload.get("observation", payload)
        obs = EvalObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: dict) -> EvalState:
        return EvalState(**payload)
