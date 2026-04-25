"""PROTOCOL-ARENA — FastAPI application.

Serves the ProtocolArenaEnvironment via OpenEnv's create_app factory. Runs
on port 7860 by default (HF Spaces standard) and is importable both as a
module (`uvicorn arena.server.app:app`) and as a script.
"""

import os
import sys

# Ensure project root is on PYTHONPATH when invoked directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openenv.core.env_server.http_server import create_app

from arena.models import OrchestratorAction, OrchestratorObservation
from arena.server.arena_env import ProtocolArenaEnvironment


app = create_app(
    env=ProtocolArenaEnvironment,
    action_cls=OrchestratorAction,
    observation_cls=OrchestratorObservation,
    env_name="protocol_arena",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
