import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import warnings

from common.base_environment import DistrictHeatingEnv
from common.utils import ValidationUtils, HeatDemandUtils
from td3_config import get_environment_config, get_td3_hyperparameters

class TD3DistrictHeatingEnv(DistrictHeatingEnv):

    def __init__(self,
                 data_file: str = None,
                 svr_model_path: str = None,
                 episode_length: int = 24,
                 start_date: str = '2024-01-01',
                 td3_optimizations: bool = True):

        env_config = get_environment_config()

        if data_file is None:
            data_file = env_config['data_file']
        if svr_model_path is None:
            svr_model_path = env_config['svr_model_path']

        super().__init__(
            data_file=data_file,
            svr_model_path=svr_model_path,
            episode_length=episode_length,
            start_date=start_date
        )

        self.td3_optimizations = td3_optimizations
        self.td3_config = get_td3_hyperparameters()

        if td3_optimizations:
            self._apply_td3_optimizations()

        self.episode_stats = {
            'production_history': [],
            'demand_history': [],
            'reward_components': [],
            'efficiency_scores': [],
            'temperature_deviations': [],
            'action_consistency': []
        }

        print("TD3 District Heating Environment initialized successfully!")
        if td3_optimizations:
            print("TD3-specific optimizations enabled")

    def _apply_td3_optimizations(self):
        self.reward_scale = 8.0
        self.base_cost = 42.0
        self.peak_multiplier = 1.4
        self.efficiency_bonus = 12.0
        self.temp_target = 85.0
        self.temp_tolerance = 12.0
        self.action_consistency_weight = 0.05
        self.previous_action = np.zeros(self.n_zones, dtype=np.float32)
        self.exploration_noise_std = 0.1
        self.noise_clip = 0.5

        print("TD3 optimizations applied:")
        print(f"  - Reward scaling: {self.reward_scale}")
        print(f"  - Efficiency bonus: {self.efficiency_bonus}")
        print(f"  - Action consistency weight: {self.action_consistency_weight}")
        print(f"  - Exploration noise std: {self.exploration_noise_std}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed, options)

        self.episode_stats = {
            'production_history': [],
            'demand_history': [],
            'reward_components': [],
            'efficiency_scores': [],
            'temperature_deviations': [],
            'action_consistency': []
        }

        if hasattr(self, 'action_consistency_weight'):
            self.previous_action = np.zeros(self.n_zones, dtype=np.float32)

        info.update({
            'td3_optimizations': self.td3_optimizations,
            'reward_scale': getattr(self, 'reward_scale', 1.0),
            'episode_stats_initialized': True,
            'exploration_noise_std': getattr(self, 'exploration_noise_std', 0.0)
        })

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        original_action = action.copy()

        obs, base_reward, terminated, truncated, info = super().step(action)

        if self.td3_optimizations:
            enhanced_reward = self._calculate_td3_reward(original_action, info, base_reward)
        else:
            enhanced_reward = base_reward

        if hasattr(self, 'reward_scale'):
            enhanced_reward *= self.reward_scale

        self._update_episode_stats(original_action, info, enhanced_reward)

        info.update({
            'base_reward': float(base_reward),
            'enhanced_reward': float(enhanced_reward),
            'reward_components': self._get_reward_components(original_action, info),
            'action_consistency': self._calculate_action_consistency(original_action)
        })

        if hasattr(self, 'action_consistency_weight'):
            self.previous_action = original_action.copy()

        return obs, float(enhanced_reward), terminated, truncated, info

    def _calculate_td3_reward(self, action: np.ndarray, info: Dict, base_reward: float) -> float:
        total_production = info['total_production']
        total_demand = info['total_demand']
        cost = info['cost']
        efficiency = info['efficiency']
        zone_temps = info['zone_temps']

        efficiency_reward = efficiency * self.efficiency_bonus

        demand_ratio = total_production / max(total_demand, 1.0)
        if 0.8 <= demand_ratio <= 1.2:
            demand_reward = 10.0 - 2.0 * abs(demand_ratio - 1.0)
        elif demand_ratio < 0.8:
            demand_reward = -8.0 * (0.8 - demand_ratio) ** 2
        else:
            demand_reward = -3.0 * (demand_ratio - 1.2) ** 2

        temp_deviations = np.abs(zone_temps - self.temp_target)
        avg_deviation = np.mean(temp_deviations)
        temp_reward = max(0, (self.temp_tolerance - avg_deviation) / self.temp_tolerance) * 6.0

        action_consistency = self._calculate_action_consistency(action)
        consistency_reward = action_consistency * self.action_consistency_weight * 20.0

        cost_penalty = -cost / 120.0

        total_reward = (
                efficiency_reward +
                demand_reward +
                temp_reward +
                consistency_reward +
                cost_penalty
        )

        return total_reward

    def _calculate_action_consistency(self, action: np.ndarray) -> float:
        if not hasattr(self, 'previous_action') or np.allclose(self.previous_action, 0):
            return 0.5

        action_diff = np.mean(np.abs(action - self.previous_action))
        consistency = max(0, 1.0 - action_diff * 2.0)

        return consistency

    def _get_reward_components(self, action: np.ndarray, info: Dict) -> Dict[str, float]:
        if not self.td3_optimizations:
            return {'total_reward': info.get('enhanced_reward', 0)}

        total_production = info['total_production']
        total_demand = info['total_demand']
        efficiency = info['efficiency']
        zone_temps = info['zone_temps']
        cost = info['cost']

        efficiency_reward = efficiency * self.efficiency_bonus

        demand_ratio = total_production / max(total_demand, 1.0)
        if 0.8 <= demand_ratio <= 1.2:
            demand_reward = 10.0 - 2.0 * abs(demand_ratio - 1.0)
        elif demand_ratio < 0.8:
            demand_reward = -8.0 * (0.8 - demand_ratio) ** 2
        else:
            demand_reward = -3.0 * (demand_ratio - 1.2) ** 2

        temp_deviations = np.abs(zone_temps - self.temp_target)
        avg_deviation = np.mean(temp_deviations)
        temp_reward = max(0, (self.temp_tolerance - avg_deviation) / self.temp_tolerance) * 6.0

        action_consistency = self._calculate_action_consistency(action)
        consistency_reward = action_consistency * self.action_consistency_weight * 20.0

        cost_penalty = -cost / 120.0

        return {
            'efficiency_reward': float(efficiency_reward),
            'demand_reward': float(demand_reward),
            'temperature_reward': float(temp_reward),
            'consistency_reward': float(consistency_reward),
            'cost_penalty': float(cost_penalty),
            'action_consistency': float(action_consistency),
            'total_reward': float(efficiency_reward + demand_reward + temp_reward +
                                  consistency_reward + cost_penalty)
        }

    def _update_episode_stats(self, action: np.ndarray, info: Dict, reward: float):
        self.episode_stats['production_history'].append(info['total_production'])
        self.episode_stats['demand_history'].append(info['total_demand'])
        self.episode_stats['reward_components'].append(self._get_reward_components(action, info))
        self.episode_stats['efficiency_scores'].append(info['efficiency'])

        if 'zone_temps' in info:
            temp_deviation = np.mean(np.abs(info['zone_temps'] - self.temp_target))
            self.episode_stats['temperature_deviations'].append(temp_deviation)

        action_consistency = self._calculate_action_consistency(action)
        self.episode_stats['action_consistency'].append(action_consistency)

    def get_episode_summary(self) -> Dict[str, Any]:
        if not self.episode_stats['production_history']:
            return {'error': 'No episode data available'}

        production_history = np.array(self.episode_stats['production_history'])
        demand_history = np.array(self.episode_stats['demand_history'])
        efficiency_scores = np.array(self.episode_stats['efficiency_scores'])
        temp_deviations = np.array(self.episode_stats['temperature_deviations'])
        action_consistency = np.array(self.episode_stats['action_consistency'])

        avg_efficiency = np.mean(efficiency_scores)
        production_demand_ratio = np.sum(production_history) / np.sum(demand_history)
        avg_temp_deviation = np.mean(temp_deviations)
        avg_action_consistency = np.mean(action_consistency)

        if self.episode_stats['reward_components']:
            reward_components = {}
            for component in ['efficiency_reward', 'demand_reward', 'temperature_reward',
                              'consistency_reward', 'cost_penalty']:
                values = [rc.get(component, 0) for rc in self.episode_stats['reward_components']]
                reward_components[f'avg_{component}'] = np.mean(values)
                reward_components[f'total_{component}'] = np.sum(values)
        else:
            reward_components = {}

        return {
            'episode_length': len(production_history),
            'avg_efficiency': float(avg_efficiency),
            'production_demand_ratio': float(production_demand_ratio),
            'avg_temperature_deviation': float(avg_temp_deviation),
            'avg_action_consistency': float(avg_action_consistency),
            'total_production': float(np.sum(production_history)),
            'total_demand': float(np.sum(demand_history)),
            'reward_components': reward_components,
            'td3_optimizations_used': self.td3_optimizations
        }

    def set_evaluation_mode(self, eval_mode: bool = True):
        if eval_mode:
            self.exploration_noise_std = 0.0
            print("Environment set to evaluation mode")
        else:
            if self.td3_optimizations:
                self.exploration_noise_std = 0.1
            print("Environment set to training mode")

    def add_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'exploration_noise_std') or self.exploration_noise_std == 0:
            return action

        noise = self.rng.normal(0, self.exploration_noise_std, action.shape)
        noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        noisy_action = np.clip(action + noise, 0.0, 1.0)

        return noisy_action

def create_td3_environment(training: bool = True, **kwargs) -> TD3DistrictHeatingEnv:
    env_config = get_environment_config()

    if training:
        start_date = env_config['train_start_date']
        td3_optimizations = True
    else:
        start_date = env_config['eval_start_date']
        td3_optimizations = True

    config = {
        'data_file': env_config['data_file'],
        'svr_model_path': env_config['svr_model_path'],
        'episode_length': env_config['episode_length'],
        'start_date': start_date,
        'td3_optimizations': td3_optimizations
    }
    config.update(kwargs)

    return TD3DistrictHeatingEnv(**config)

def test_td3_environment():
    print("Testing TD3 District Heating Environment...")
    print("=" * 50)

    try:
        env_config = get_environment_config()
        is_valid = ValidationUtils.validate_environment_setup(
            env_config['data_file'],
            env_config['svr_model_path']
        )

        if not is_valid:
            print("Environment setup validation failed!")
            return False

        env = create_td3_environment(training=True)

        print(f"Environment created successfully!")
        print(f"  - TD3 optimizations: {env.td3_optimizations}")
        print(f"  - Observation space: {env.observation_space.shape}")
        print(f"  - Action space: {env.action_space.shape}")

        obs, info = env.reset()
        print(f"  - Reset successful: obs shape {obs.shape}")
        print(f"  - Info keys: {list(info.keys())}")

        print(f"\nTesting environment steps...")
        total_reward = 0

        for step in range(5):
            action = env.action_space.sample()

            if hasattr(env, 'add_exploration_noise'):
                noisy_action = env.add_exploration_noise(action)
                print(f"Step {step+1}: action noise applied, max diff: {np.max(np.abs(action - noisy_action)):.3f}")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step {step+1}: reward={reward:.2f}, consistency={info.get('action_consistency', 0):.3f}")

            if terminated or truncated:
                break

        summary = env.get_episode_summary()
        print(f"\nEpisode summary keys: {list(summary.keys())}")
        print(f"Average action consistency: {summary.get('avg_action_consistency', 0):.3f}")

        env.set_evaluation_mode(True)
        env.set_evaluation_mode(False)

        print(f"\nTD3 environment test completed successfully!")
        print(f"Average reward: {total_reward/5:.2f}")

        return True

    except Exception as e:
        print(f"TD3 environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TD3 District Heating Environment")
    print("=" * 40)

    success = test_td3_environment()

    if success:
        print(f"\nTD3 environment ready for training!")
        print(f"\nNext steps:")
        print(f"1. Test TD3 setup: python td3_implementation/test_td3.py")
        print(f"2. Start TD3 training: python td3_implementation/td3_training.py")
    else:
        print(f"\nTD3 environment test failed!")
        print(f"Please check the error messages and fix issues before proceeding.")