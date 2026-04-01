
from google import genai


class LNGAgent:
    def __init__(self, model_name: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # -----------------------------
    # LLM ACTION
    # -----------------------------
    def get_llm_action(self, state: dict) -> dict:
        print("🔥 LLM ACTION CALL")

        try:
            t = state["time_step"]
            demand = state["demand_forecast"][t]
            storage = state["storage"]["level"]
            capacity = state["storage"]["capacity"]
            budget = state["budget"]

            ships = state.get("ships", [])
            blocked = state.get("blocked_routes", [])

            incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

            prompt = f"""
You are managing LNG supply.

GOAL:
- Avoid shortage (MOST IMPORTANT)
- Minimize cost

STATE:
Demand: {demand}
Storage: {storage}/{capacity}
Incoming: {incoming}
Budget: {budget}
Blocked Routes: {blocked}

IMPORTANT:
- release reduces storage
- store/hedge increases supply
- avoid shortage

Choose ONE:
wait / store / hedge / release_20 / release_50 / reroute

ONLY output action.
"""

            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            text = res.text.strip().lower()
            print(f"✅ LLM RAW: {text}")

            if "store" in text:
                return {"type": "store", "parameters": {"amount": 20}}
            if "hedge" in text:
                return {"type": "hedge", "parameters": {}}
            if "reroute" in text:
                return {"type": "reroute", "parameters": {"ship_id": 1, "new_route": "Atlantic"}}
            if "50" in text:
                return {"type": "release", "parameters": {"amount": 50}}
            if "20" in text:
                return {"type": "release", "parameters": {"amount": 20}}

            return {"type": "wait", "parameters": {}}

        except Exception as e:
            print(f"❌ LLM ERROR: {e}")
            return {"type": "wait", "parameters": {}}

    # -----------------------------
    # BASELINE (SAFE)
    # -----------------------------
    def baseline(self, state: dict) -> dict:
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]
        capacity = state["storage"]["capacity"]
        budget = state["budget"]

        ships = state.get("ships", [])
        blocked = state.get("blocked_routes", [])

        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        supply = storage + incoming
        deficit = demand - supply

        # deficit handling
        if deficit > 0:
            if budget >= 20:
                return {"type": "store", "parameters": {"amount": 20}}
            return {"type": "hedge", "parameters": {}}

        # reroute blocked ships
        for ship in ships:
            if ship["route"] in blocked:
                return {
                    "type": "reroute",
                    "parameters": {
                        "ship_id": ship["id"],
                        "new_route": "Atlantic"
                    }
                }

        # safe release
        if storage > 0.85 * capacity and deficit <= 0:
            return {"type": "release", "parameters": {"amount": 20}}

        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # SAFETY FILTER
    # -----------------------------
    def safe(self, state: dict, action: dict) -> dict:
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]
        capacity = state["storage"]["capacity"]

        ships = state.get("ships", [])
        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        supply = storage + incoming
        deficit = demand - supply

        # 🚨 NEVER release if deficit
        if deficit > 0 and action["type"] == "release":
            return self.baseline(state)

        # 🚨 NEVER reroute at start
        if t == 0 and action["type"] == "reroute":
            return self.baseline(state)

        # 🚨 ignore LLM when stable & safe
        if deficit <= 0 and storage < 0.5 * capacity:
            return self.baseline(state)

        return action

    # -----------------------------
    # FINAL DECISION
    # -----------------------------
    def act(self, state: dict, use_llm=False) -> dict:
        base = self.baseline(state)

        if use_llm:
            llm_action = self.get_llm_action(state)
            llm_action = self.safe(state, llm_action)
            return llm_action

        return base

    

# agent.py v2
# the following code calls the llm for every step (if we have no rate limit optimal code better results)
'''

from google import genai

class LNGAgent:
    def __init__(self, model_name: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # -----------------------------
    # 🔥 STRONG PROMPT (V2)
    # -----------------------------
    def _build_prompt(self, state):
        t = state["time_step"]

        return f"""
You are an expert LNG supply optimizer.

CRITICAL:
- Release REDUCES storage
- Store/Hedge INCREASE supply
- Avoid shortage (highest priority)

STATE:
Demand: {state['demand_forecast'][t]}
Storage: {state['storage']['level']}/{state['storage']['capacity']}
Budget: {state['budget']}
Ships: {state['ships']}
Blocked Routes: {state['blocked_routes']}

Think step-by-step:
1. Will shortage occur?
2. Do we need more supply?
3. Is storage too high?

Choose ONE action.

ACTIONS:
wait, store, hedge, release_25, release_50, reroute

ONLY OUTPUT ACTION.
"""

    # -----------------------------
    # LLM CALL
    # -----------------------------
    def _call_llm(self, state):
        try:
            prompt = self._build_prompt(state)

            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            return res.text.strip().lower()
        except:
            return "wait"

    # -----------------------------
    # PARSER
    # -----------------------------
    def _parse(self, text):
        if "store" in text:
            return {"type": "store", "parameters": {"amount": 20}}
        if "hedge" in text:
            return {"type": "hedge", "parameters": {}}
        if "reroute" in text:
            return {"type": "reroute", "parameters": {"ship_id": 1, "new_route": "Atlantic"}}
        if "50" in text:
            return {"type": "release", "parameters": {"amount": 50}}
        if "25" in text:
            return {"type": "release", "parameters": {"amount": 25}}
        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # SIMULATION (safety)
    # -----------------------------
    def _simulate(self, state, action):
        storage = state["storage"]["level"]
        capacity = state["storage"]["capacity"]

        if action["type"] == "release":
            storage = max(0, storage - action["parameters"].get("amount", 0))

        elif action["type"] == "store":
            storage = min(capacity, storage + action["parameters"].get("amount", 0))

        elif action["type"] == "hedge":
            storage += 20

        t = state["time_step"]
        demand = state["demand_forecast"][t]

        ships = state.get("ships", [])
        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        supply = storage + incoming
        deficit = max(0, demand - supply)

        return deficit

    # -----------------------------
    # FALLBACK (safe heuristic)
    # -----------------------------
    def _fallback(self, state):
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]

        ships = state.get("ships", [])
        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        if demand > storage + incoming:
            return {"type": "store", "parameters": {"amount": 20}}

        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # FINAL ACT
    # -----------------------------
    def act(self, state):
        llm_text = self._call_llm(state)
        llm_action = self._parse(llm_text)

        fallback = self._fallback(state)

        # simulate both
        llm_deficit = self._simulate(state, llm_action)
        fb_deficit = self._simulate(state, fallback)

        # pick safer
        if llm_deficit <= fb_deficit:
            return llm_action

        return fallback

'''