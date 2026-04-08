from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class LNGAction(Action):


    action_type: str = Field(
        default="wait",
        description="Action type: wait | store | release | reroute | hedge",
    )
    amount: float = Field(
        default=0.0,
        description="Amount of LNG to store or release (for store/release actions)",
    )
    ship_id: Optional[int] = Field(
        default=None,
        description="Ship ID to reroute (1 or 2, for reroute action)",
    )
    new_route: Optional[str] = Field(
        default=None,
        description="New route for ship: Suez | Panama | Atlantic | Hormuz (for reroute action)",
    )


class LNGObservation(Observation):


    time_step: int = Field(default=0, description="Current simulation step")
    ships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Active LNG carrier vessels"
    )
    blocked_routes: List[str] = Field(
        default_factory=list, description="Currently disrupted shipping corridors"
    )
    storage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current LNG storage state (level, capacity)",
    )
    demand_forecast: List[float] = Field(
        default_factory=list, description="Forecasted demand per timestep"
    )
    price: float = Field(default=0.0, description="Current spot LNG price ($/unit)")
    budget: float = Field(default=0.0, description="Remaining hedging capital ($)")
    goal: str = Field(
        default="",
        description="Episode goal description",
    )
