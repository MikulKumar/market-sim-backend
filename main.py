from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from simulation import (simulation, MarketMaker, NoiseTrader, Trendfollower,
                        RetailTrader, Whale, MeanReversion, OrderFlow)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-url.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
AGENT_MAP = {
    "MarketMaker": MarketMaker,
    "NoiseTrader": NoiseTrader,
    "Trendfollower": Trendfollower,
    "RetailTrader": RetailTrader,
    "Whale": Whale,
    "MeanReversion": MeanReversion,
    "OrderFlow": OrderFlow,
}

from typing import Any, Union

class AgentConfig(BaseModel):
    type: str
    num: int
    params: dict[str, Union[list, float, int]]

class SimConfig(BaseModel):
    start_price: float = 100.0
    ticks: int = 500
    agents: list[AgentConfig]

@app.post("/run")
def run_simulation(config: SimConfig):
    sim = simulation(start_price=config.start_price)

    for agent_config in config.agents:
        if agent_config.num == 0:
            continue
        agent_class = AGENT_MAP[agent_config.type]
        
        # convert lists back to tuples
        params = {}
        for key, val in agent_config.params.items():
            params[key] = tuple(val) if isinstance(val, list) else val

        sim.add_population(agent_class, num=agent_config.num, params=params)

    sim.run(ticks=config.ticks)
    return {
        "prices": sim.price_series(),
        "volumes": sim.volume_series()
    }