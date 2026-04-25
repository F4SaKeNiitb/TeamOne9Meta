"""PROTOCOL-ARENA — Typed OpenEnv WebSocket client."""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    OrchestratorAction,
    OrchestratorObservation,
    OrchestratorState,
)


class ProtocolArenaEnv(EnvClient[OrchestratorAction, OrchestratorObservation, OrchestratorState]):
    """Client for the PROTOCOL-ARENA environment."""

    def _step_payload(self, action: OrchestratorAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[OrchestratorObservation]:
        obs_data = payload.get("observation", payload)
        obs = OrchestratorObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: dict) -> OrchestratorState:
        return OrchestratorState(**payload)
