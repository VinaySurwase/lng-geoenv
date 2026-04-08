
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: pip install openenv-core") from e

from server.models import LNGAction, LNGObservation
from server.lng_geoenv_environment import LNGEnvironment


# Create the OpenEnv-compliant FastAPI app
app = create_app(
    LNGEnvironment,
    LNGAction,
    LNGObservation,
    env_name="lng-geoenv",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
