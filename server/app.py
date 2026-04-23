"""
EvalForge — server/app.py
FastAPI application using OpenEnv's create_app factory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

try:
    from ..models import EvalAction, EvalObservation
except ImportError:
    from models import EvalAction, EvalObservation

try:
    from .evalforge_environment import EvalForgeEnvironment
except ImportError:
    from evalforge_environment import EvalForgeEnvironment


app = create_app(
    env=EvalForgeEnvironment,
    action_cls=EvalAction,
    observation_cls=EvalObservation,
    env_name="evalforge",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
