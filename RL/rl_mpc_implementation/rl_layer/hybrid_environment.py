import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import sys
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from config.mpc_config import get_mpc_config, MPCConfig
from mpc_core.physics_model import DistrictHeatingPhysics
from mpc_core.mpc_controller import DistrictHeatingMPC
from integration.forecasting_bridge import SVRForecastingBridge
from rl_layer.mpc_parameter_agent import MPCParameterAgent

class HybridRLMPCEnvironment(gym.Env):

    def __init__(self,
                 config: Optional[MPCConfig] = None,
                 episode_length: int = 24,
                 start_date: str = '2024-01-01',
                 data_split: str = 'train'):

        super(HybridRLMPCEnvironment, self).__init__()

        self.config = config if config is not None else get_mpc_config()

        self.physics = DistrictHeatingPhysics(self.config)
        self.forecaster = SVRForecastingBridge(self.config)
        self.mpc = DistrictHeatingMPC(self.config, self.physics, self.forecaster)

        self.episode_length = episode_length
        self.current_step = 0
        self.episode_start_idx = 0

        self.data = self.forecaster.data
        self.data_split = data_split
        self._setup_data_splits()

        self.zones = self.config.zones
        self.n_zones = len(self.zones)

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        obs_dim = 59

        self.observation_space = spaces.Box(
            low=-100.0,
            high=200.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.episode_stats = {
            'costs': [],
            'demand_satisfactions': [],
            'efficiencies': [],
            'mpc_solve_times': [],
            'constraint_violations': [],
            'parameter_history': []
        }

        self.current_time = None
        self.outdoor_temp = 10.0
        self.hour = 12

        self.rng = np.random.default_rng()

        print("Hybrid RL-MPC Environment initialized")
        print(f"Episode length: {episode_length}h")
        print(f"Action space: {self.action_space.shape} (MPC parameters)")
        print(f"Observation space: {self.observation_space.shape}")
        print(f"Data split: {data_split}")

    def _setup_data_splits(self):
        if self.data is None:
            print("Warning: No data available, using synthetic time")
            return

        total_length = len(self.data)

        train_end = int(0.7 * total_length)
        val_end = int(0.85 * total_length)

        if self.data_split == 'train':
            self.data_start = 0
            self.data_end = train_end
        elif self.data_split == 'val':
            self.data_start = train_end
            self.data_end = val_end
        else:
            self.data_start = val_end
            self.data_end = total_length

        print(f"Data split '{self.data_split}': indices {self.data_start} to {self.data_end}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.episode_stats = {
            'costs': [],
            'demand_satisfactions': [],
            'efficiencies': [],
            'mpc_solve_times': [],
            'constraint_violations': [],
            'parameter_history': []
        }

        if self.data is not None:
            max_start = self.data_end - self.episode_length - 1
            if max_start <= self.data_start:
                max_start = self.data_start + 1

            episode_idx = self.rng.integers(self.data_start, max_start)
            self.current_time = self.data.index[episode_idx]

            weather_row = self.data.iloc[episode_idx]
            self.outdoor_temp = self._safe_get_value(weather_row, 'temp', 10.0)
            self.hour = self.current_time.hour
        else:
            self.current_time = pd.Timestamp('2024-01-01 12:00:00')
            self.outdoor_temp = 10.0
            self.hour = 12

        self.physics.reset_state()
        self.mpc.last_solution = None

        obs = self._get_observation()

        info = {
            'episode_start_time': str(self.current_time),
            'outdoor_temp': self.outdoor_temp,
            'hour': self.hour,
            'physics_state': self.physics.get_current_state(),
            'data_split': self.data_split
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        mpc_weights = self._action_to_mpc_weights(action)
        self.mpc.update_weights(mpc_weights)

        current_state = self.physics.get_current_state()

        mpc_solution = self.mpc.solve_mpc(
            self.current_time,
            current_state,
            self.outdoor_temp,
            self.hour
        )

        optimal_heat_production = mpc_solution['optimal_action']

        demand_forecast = mpc_solution['forecast_used']
        current_demand = demand_forecast[0, :] if demand_forecast.shape[0] > 0 else np.ones(self.n_zones) * 5.0

        physics_results = self.physics.update_thermal_dynamics(
            optimal_heat_production,
            self.outdoor_temp,
            current_demand
        )

        reward = self._calculate_rl_reward(mpc_solution, physics_results)

        self._update_episode_stats(mpc_solution, action)

        self.current_step += 1
        self._update_time()

        terminated = self.current_step >= self.episode_length
        truncated = False

        obs = self._get_observation()

        info = {
            'mpc_solution': mpc_solution,
            'physics_results': physics_results,
            'optimal_heat_production': optimal_heat_production,
            'current_demand': current_demand,
            'mpc_weights_used': mpc_weights,
            'step': self.current_step,
            'outdoor_temp': self.outdoor_temp,
            'hour': self.hour,
            'daily_cost_estimate': mpc_solution['daily_cost_estimate'],
            'demand_satisfaction': mpc_solution['demand_satisfaction'],
            'efficiency': mpc_solution['efficiency'],
            'constraint_violations': mpc_solution['constraint_violations']['total_violations']
        }

        return obs, float(reward), terminated, truncated, info

    def _action_to_mpc_weights(self, action: np.ndarray) -> Dict[str, float]:
        bounds = {
            'cost_weight': (0.1, 3.0),
            'comfort_weight': (0.5, 4.0),
            'efficiency_weight': (0.1, 2.0),
            'stability_weight': (0.05, 1.0)
        }

        mpc_weights = {}
        param_names = ['cost_weight', 'comfort_weight', 'efficiency_weight', 'stability_weight']

        for i, param_name in enumerate(param_names):
            min_val, max_val = bounds[param_name]
            param_value = min_val + action[i] * (max_val - min_val)
            mpc_weights[param_name] = float(param_value)

        self.current_prediction_horizon_scale = 0.5 + action[4] * 1.0
        self.current_risk_factor = 0.1 + action[5] * 0.9

        return mpc_weights

    def _get_observation(self) -> np.ndarray:
        physics_state = self.physics.get_current_state()
        zone_temps = physics_state['temperatures'] / 100.0
        zone_flows = physics_state['flows'] / 100.0
        zone_pressures = physics_state['pressures'] / 30.0

        physics_obs = np.concatenate([zone_temps, zone_flows, zone_pressures])

        weather_obs = np.array([
            self.outdoor_temp / 40.0 + 0.5,
            0.6,
            1013.0 / 1100.0,
            5.0 / 20.0,
            200.0 / 1000.0 if 6 <= self.hour <= 18 else 0.0
        ], dtype=np.float32)

        try:
            forecast = self.forecaster.get_demand_forecast(
                self.current_time, self.outdoor_temp, self.hour, 24
            )
            forecast_obs = forecast['zone_demands'] / 100.0
        except:
            fallback_total = max(5.0, (15.5 - self.outdoor_temp) * 3.5)
            zone_weights = np.array([self.config.zone_parameters[zone]['zone_weight'] for zone in self.zones])
            forecast_obs = (fallback_total * zone_weights) / 100.0

        if self.mpc.last_solution is not None:
            last_solution = self.mpc.last_solution
            mpc_obs = np.array([
                last_solution['daily_cost_estimate'] / 100.0,
                last_solution['demand_satisfaction'],
                last_solution['efficiency'],
                min(1.0, len(self.mpc.solve_times) / 100.0) if self.mpc.solve_times else 0.0,
                min(1.0, last_solution['constraint_violations']['total_violations'] / 10.0),
                min(1.0, abs(last_solution['objective_value']) / 1000.0) if last_solution['objective_value'] != float('inf') else 1.0
            ], dtype=np.float32)
        else:
            mpc_obs = np.zeros(6, dtype=np.float32)

        time_obs = np.array([
            self.hour / 24.0,
            self.current_time.dayofweek / 7.0 if hasattr(self.current_time, 'dayofweek') else 0.0,
            self.current_time.month / 12.0 if hasattr(self.current_time, 'month') else 0.5,
            ((self.current_time.month - 1) // 3) / 4.0 if hasattr(self.current_time, 'month') else 0.0
        ], dtype=np.float32)

        observation = np.concatenate([
            physics_obs,
            weather_obs,
            forecast_obs,
            mpc_obs,
            time_obs
        ])

        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)
        observation = observation.astype(np.float32)

        if observation.shape[0] < 59:
            observation = np.pad(observation, (0, 59 - observation.shape[0]), 'constant')
        elif observation.shape[0] > 59:
            observation = observation[:59]

        return observation

    def _calculate_rl_reward(self, mpc_solution: Dict[str, Any],
                             physics_results: Dict[str, np.ndarray]) -> float:
        daily_cost = mpc_solution['daily_cost_estimate']
        demand_satisfaction = mpc_solution['demand_satisfaction']
        efficiency = mpc_solution['efficiency']
        constraint_violations = mpc_solution['constraint_violations']['total_violations']

        target_cost = self.config.cost_params['target_daily_cost']

        cost_ratio = daily_cost / target_cost
        if cost_ratio <= 1.0:
            cost_reward = 25.0 * (1.0 - cost_ratio)
        else:
            cost_reward = -15.0 * (cost_ratio - 1.0)

        min_demand = self.config.constraints['hard_constraints']['min_demand_satisfaction']
        if demand_satisfaction >= min_demand:
            demand_reward = 15.0 + 5.0 * (demand_satisfaction - min_demand) / (1.0 - min_demand)
        else:
            demand_reward = -100.0 * (min_demand - demand_satisfaction)

        efficiency_reward = 8.0 * efficiency
        violation_penalty = -30.0 * constraint_violations
        solver_bonus = 2.0 if mpc_solution['solver_status'] == 'optimal' else -5.0
        stability_bonus = 3.0

        total_reward = (
                cost_reward +
                demand_reward +
                efficiency_reward +
                violation_penalty +
                solver_bonus +
                stability_bonus
        )

        return total_reward

    def _update_episode_stats(self, mpc_solution: Dict[str, Any], action: np.ndarray):
        self.episode_stats['costs'].append(mpc_solution['daily_cost_estimate'])
        self.episode_stats['demand_satisfactions'].append(mpc_solution['demand_satisfaction'])
        self.episode_stats['efficiencies'].append(mpc_solution['efficiency'])
        self.episode_stats['constraint_violations'].append(mpc_solution['constraint_violations']['total_violations'])

        if self.mpc.solve_times:
            self.episode_stats['mpc_solve_times'].append(self.mpc.solve_times[-1])
        else:
            self.episode_stats['mpc_solve_times'].append(0.0)

        self.episode_stats['parameter_history'].append(action.copy())

    def _update_time(self):
        if self.data is not None:
            try:
                current_idx = self.data.index.get_loc(self.current_time)
                if current_idx + 1 < len(self.data):
                    self.current_time = self.data.index[current_idx + 1]
                    weather_row = self.data.iloc[current_idx + 1]
                    self.outdoor_temp = self._safe_get_value(weather_row, 'temp', 10.0)
                    self.hour = self.current_time.hour
                else:
                    self.hour = (self.hour + 1) % 24
            except:
                self.hour = (self.hour + 1) % 24
                self.current_time = self.current_time + pd.Timedelta(hours=1)
        else:
            self.hour = (self.hour + 1) % 24
            self.current_time = self.current_time + pd.Timedelta(hours=1)
            self.outdoor_temp = 10.0 + 5.0 * np.sin(2 * np.pi * self.hour / 24)

    def _safe_get_value(self, row: pd.Series, column: str, default: float) -> float:
        if column in row.index:
            try:
                value = float(row[column])
                return value if not np.isnan(value) else default
            except (ValueError, TypeError):
                return default
        return default

    def get_episode_summary(self) -> Dict[str, Any]:
        if not self.episode_stats['costs']:
            return {'error': 'No episode data available'}

        costs = np.array(self.episode_stats['costs'])
        demand_sats = np.array(self.episode_stats['demand_satisfactions'])
        efficiencies = np.array(self.episode_stats['efficiencies'])
        violations = np.array(self.episode_stats['constraint_violations'])
        solve_times = np.array(self.episode_stats['mpc_solve_times'])

        return {
            'episode_length': len(costs),
            'avg_daily_cost': float(np.mean(costs)),
            'min_daily_cost': float(np.min(costs)),
            'max_daily_cost': float(np.max(costs)),
            'cost_std': float(np.std(costs)),
            'avg_demand_satisfaction': float(np.mean(demand_sats)),
            'min_demand_satisfaction': float(np.min(demand_sats)),
            'avg_efficiency': float(np.mean(efficiencies)),
            'total_violations': int(np.sum(violations)),
            'avg_solve_time': float(np.mean(solve_times)),
            'target_achievement': {
                'cost_target_met': np.mean(costs) <= self.config.cost_params['target_daily_cost'],
                'demand_target_met': np.min(demand_sats) >= self.config.constraints['hard_constraints']['min_demand_satisfaction'],
                'combined_success': (np.mean(costs) <= self.config.cost_params['target_daily_cost'] and
                                     np.min(demand_sats) >= self.config.constraints['hard_constraints']['min_demand_satisfaction'])
            }
        }

def test_hybrid_environment():
    print("Testing Hybrid RL-MPC Environment...")
    print("=" * 50)

    try:
        env = HybridRLMPCEnvironment(episode_length=12, data_split='train')

        print(f"Environment created successfully!")
        print(f"Action space: {env.action_space.shape}")
        print(f"Observation space: {env.observation_space.shape}")

        obs, info = env.reset(seed=42)
        print(f"\nReset successful:")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Start time: {info['episode_start_time']}")
        print(f"  Outdoor temp: {info['outdoor_temp']:.1f}°C")

        print(f"\nTesting environment steps...")
        total_reward = 0

        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step {step+1}:")
            print(f"  Action (first 4): {action[:4]}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Daily cost: {info['daily_cost_estimate']:.1f}€")
            print(f"  Demand satisfaction: {info['demand_satisfaction']:.1%}")
            print(f"  Constraint violations: {info['constraint_violations']}")

            if terminated or truncated:
                break

        summary = env.get_episode_summary()
        print(f"\nEpisode Summary:")
        print(f"  Average daily cost: {summary['avg_daily_cost']:.1f}€")
        print(f"  Average demand satisfaction: {summary['avg_demand_satisfaction']:.1%}")
        print(f"  Total violations: {summary['total_violations']}")
        print(f"  Cost target met: {summary['target_achievement']['cost_target_met']}")
        print(f"  Demand target met: {summary['target_achievement']['demand_target_met']}")
        print(f"  Combined success: {summary['target_achievement']['combined_success']}")

        print(f"\nHybrid RL-MPC environment test completed successfully!")
        return True

    except Exception as e:
        print(f"Hybrid environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hybrid_environment()

    if success:
        print(f"\nHybrid RL-MPC Environment ready!")
        print(f"Key capabilities:")
        print(f"  - RL agent tunes MPC parameters (6D action space)")
        print(f"  - MPC enforces constraints and optimizes heat production")
        print(f"  - Physics model ensures realistic thermal dynamics")
        print(f"  - SVR forecasting provides demand predictions")
        print(f"  - Comprehensive performance tracking")
        print(f"  - Target: 25€/day cost with 85% demand satisfaction")
    else:
        print(f"\nHybrid environment test failed!")
        print(f"Check component integration and dependencies.")