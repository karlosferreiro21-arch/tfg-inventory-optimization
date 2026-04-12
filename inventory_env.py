import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class InventoryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Environment parameters
        self.initial_stock = 50
        self.max_capacity = 200
        self.demand_lambda = 20
        self.lead_time_low = 2
        self.lead_time_high = 8
        self.holding_cost = 1.0
        self.stockout_cost = 10.0
        self.fixed_order_cost = 10.0
        self.episode_length = 52

        # Action space: 0, 5, 10, ..., 100 (21 discrete actions)
        self.action_space = spaces.Discrete(21)

        # Observation space: [stock_norm, in_transit_norm, demand_t-4..t-1 (4 values), week_norm]
        # All values in [0, 1]
        low = np.zeros(7, dtype=np.float32)
        high = np.ones(7, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state (initialized in reset)
        self.stock = None
        self.in_transit = None       # list of (units, remaining_weeks)
        self.demand_history = None   # deque of last 4 demands
        self.week = None
        self.np_random = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        stock_norm = self.stock / self.max_capacity
        in_transit_total = sum(units for units, _ in self.in_transit)
        # Normalize in-transit by the maximum possible: 100 units * 8 weeks of lead time
        in_transit_norm = np.clip(in_transit_total / (100 * self.lead_time_high), 0.0, 1.0)
        demand_norm = np.array(self.demand_history, dtype=np.float32) / self.demand_lambda
        week_norm = self.week / self.episode_length

        obs = np.array(
            [stock_norm, in_transit_norm, *demand_norm, week_norm],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.stock = float(self.initial_stock)
        self.in_transit = []                          # [(units, remaining_weeks), ...]
        self.demand_history = deque([0.0] * 4, maxlen=4)
        self.week = 0

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.stock is not None, "Call reset() before step()."

        order_qty = int(action) * 5   # map action index → units (0, 5, 10, ..., 100)

        # 1. Place order (if any)
        fixed_cost_this_step = 0.0
        if order_qty > 0:
            lead_time = int(self.np_random.integers(self.lead_time_low, self.lead_time_high + 1))
            self.in_transit.append([order_qty, lead_time])
            fixed_cost_this_step = self.fixed_order_cost

        # 2. Advance in-transit orders; collect arrivals
        units_received = 0.0
        remaining = []
        for entry in self.in_transit:
            entry[1] -= 1
            if entry[1] <= 0:
                units_received += entry[0]
            else:
                remaining.append(entry)
        self.in_transit = remaining

        # 3. Add received stock (cap at max capacity)
        self.stock = min(self.stock + units_received, float(self.max_capacity))

        # 4. Generate demand
        demand = float(self.np_random.poisson(self.demand_lambda))
        self.demand_history.append(demand)

        # 5. Update stock; compute stockout
        fulfilled = min(self.stock, demand)
        units_short = max(demand - self.stock, 0.0)
        self.stock = max(self.stock - demand, 0.0)

        # 6. Compute reward (negative cost)
        holding = self.holding_cost * self.stock
        stockout = self.stockout_cost * units_short
        cost = holding + stockout + fixed_cost_this_step
        reward = -cost

        # 7. Advance week
        self.week += 1
        terminated = self.week >= self.episode_length
        truncated = False

        info = {
            "stock": self.stock,
            "units_short": units_short,
            "cost": cost,
            "units_received": units_received,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        in_transit_total = sum(units for units, _ in self.in_transit)
        print(
            f"Week {self.week:>2}/{self.episode_length} | "
            f"Stock: {self.stock:>6.1f} | "
            f"In-transit: {in_transit_total:>5.0f} | "
            f"Last demand: {self.demand_history[-1]:>5.1f}"
        )

    def close(self):
        pass


# ----------------------------------------------------------------------

if __name__ == "__main__":
    env = InventoryEnv()
    obs, _ = env.reset(seed=42)

    total_cost = 0.0
    done = False

    print("Running full episode with random policy...\n")
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_cost += info["cost"]
        env.render()
        done = terminated or truncated

    print(f"\nTotal accumulated cost over {env.episode_length} weeks: {total_cost:.2f} €")
