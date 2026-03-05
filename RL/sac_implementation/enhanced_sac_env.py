import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import warnings

from common.base_environment import DistrictHeatingEnv
from common.utils import ValidationUtils, HeatDemandUtils
from enhanced_sac_config import get_enhanced_environment_config, get_enhanced_sac_hyperparameters

class EnhancedSACDistrictHeatingEnv(DistrictHeatingEnv):

    def __init__(self,
                 data_file: str = None,
                 svr_model_path: str = None,
                 episode_length: int = 24,
                 start_date: str = '2024-01-01',
                 cost_optimization: bool = True):

        env_config = get_enhanced_environment_config()

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

        self.cost_optimization = cost_optimization
        self.enhanced_config = get_enhanced_sac_hyperparameters()
        self.env_config = env_config

        if cost_optimization:
            self._apply_cost_optimizations()

        self.cost_history = []
        self.efficiency_history = []
        self.demand_satisfaction_history = []
        self.episode_stats = {
            'cost_components': [],
            'efficiency_scores': [],
            'demand_ratios': [],
            'reward_breakdown': [],
            'action_consistency': []
        }

        print("SAC District Heating Environment initialized!")
        if cost_optimization:
            print("Cost optimization enhancements enabled")

    def _apply_cost_optimizations(self):
        self.base_cost_per_mwh = 30.0
        self.peak_cost_multiplier = 1.1
        self.efficiency_bonus_scale = 25.0

        self.cost_weight = self.env_config['cost_weight']
        self.efficiency_weight = self.env_config['efficiency_weight']
        self.demand_weight = self.env_config['demand_weight']
        self.stability_weight = self.env_config['stability_weight']

        self.reward_scale = 8.0

        self.action_smoothing = 0.12
        self.previous_action = np.zeros(self.n_zones, dtype=np.float32)

        self.target_daily_cost = 25.0
        self.cost_penalty_threshold = 30.0

        print("Cost optimization parameters:")
        print(f"  - Cost weight: {self.cost_weight:.1f}")
        print(f"  - Base cost: {self.base_cost_per_mwh}€/MWh")
        print(f"  - Target daily cost: {self.target_daily_cost}€")
        print(f"  - Reward scaling: {self.reward_scale}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed, options)

        self.cost_history = []
        self.efficiency_history = []
        self.demand_satisfaction_history = []
        self.episode_stats = {
            'cost_components': [],
            'efficiency_scores': [],
            'demand_ratios': [],
            'reward_breakdown': [],
            'action_consistency': []
        }

        if hasattr(self, 'action_smoothing'):
            self.previous_action = np.zeros(self.n_zones, dtype=np.float32)

        info.update({
            'cost_optimization_enabled': self.cost_optimization,
            'target_daily_cost': getattr(self, 'target_daily_cost', None),
            'cost_weight': getattr(self, 'cost_weight', None),
            'reward_scale': getattr(self, 'reward_scale', 1.0)
        })

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if hasattr(self, 'action_smoothing') and self.action_smoothing > 0:
            smoothed_action = (1 - self.action_smoothing) * action + self.action_smoothing * self.previous_action
            self.previous_action = action.copy()
            action = smoothed_action

        obs, base_reward, terminated, truncated, info = super().step(action)

        if self.cost_optimization:
            enhanced_reward = self._calculate_cost_optimized_reward(action, info, base_reward)
        else:
            enhanced_reward = base_reward

        if hasattr(self, 'reward_scale'):
            enhanced_reward *= self.reward_scale

        self._update_cost_tracking(action, info, enhanced_reward)

        info.update({
            'base_reward': float(base_reward),
            'enhanced_reward': float(enhanced_reward),
            'reward_breakdown': self._get_reward_breakdown(action, info),
            'cost_analysis': self._get_cost_analysis(info),
            'action_smoothed': hasattr(self, 'action_smoothing') and self.action_smoothing > 0
        })

        return obs, float(enhanced_reward), terminated, truncated, info

    def _calculate_cost_optimized_reward(self, action: np.ndarray, info: Dict, base_reward: float) -> float:
        total_production = info['total_production']
        total_demand = info['total_demand']
        cost = info['cost']
        efficiency = info['efficiency']
        zone_temps = info.get('zone_temps', np.zeros(self.n_zones))

        daily_cost_estimate = cost * 24
        if daily_cost_estimate <= self.target_daily_cost:
            cost_reward = 20.0 * (1.0 - daily_cost_estimate / self.target_daily_cost)
        else:
            excess_ratio = (daily_cost_estimate - self.target_daily_cost) / self.target_daily_cost
            cost_reward = -10.0 * (1.0 + excess_ratio ** 2)

        cost_component = self.cost_weight * cost_reward

        efficiency_reward = efficiency * 15.0
        if efficiency > 0.7:
            efficiency_reward += (efficiency - 0.7) * 10.0

        efficiency_component = self.efficiency_weight * efficiency_reward

        demand_ratio = total_production / max(total_demand, 1.0)
        if 0.9 <= demand_ratio <= 1.1:
            demand_reward = 8.0
        elif demand_ratio < 0.9:
            demand_penalty = (0.9 - demand_ratio) * 15.0
            demand_reward = -demand_penalty
        else:
            overproduction_penalty = (demand_ratio - 1.1) * 8.0
            demand_reward = -overproduction_penalty

        demand_component = self.demand_weight * demand_reward

        temp_deviations = np.abs(zone_temps - 85.0)
        temp_stability_score = 1.0 - np.mean(temp_deviations) / 85.0
        stability_reward = temp_stability_score * 5.0

        stability_component = self.stability_weight * stability_reward

        total_reward = cost_component + efficiency_component + demand_component + stability_component

        return float(total_reward)

    def _get_reward_breakdown(self, action: np.ndarray, info: Dict) -> Dict[str, float]:
        total_production = info['total_production']
        total_demand = info['total_demand']
        cost = info['cost']
        efficiency = info['efficiency']
        zone_temps = info.get('zone_temps', np.zeros(self.n_zones))

        daily_cost_estimate = cost * 24
        if daily_cost_estimate <= self.target_daily_cost:
            cost_reward = 20.0 * (1.0 - daily_cost_estimate / self.target_daily_cost)
        else:
            excess_ratio = (daily_cost_estimate - self.target_daily_cost) / self.target_daily_cost
            cost_reward = -10.0 * (1.0 + excess_ratio ** 2)

        efficiency_reward = efficiency * 15.0
        if efficiency > 0.7:
            efficiency_reward += (efficiency - 0.7) * 10.0

        demand_ratio = total_production / max(total_demand, 1.0)
        if 0.9 <= demand_ratio <= 1.1:
            demand_reward = 8.0
        elif demand_ratio < 0.9:
            demand_penalty = (0.9 - demand_ratio) * 15.0
            demand_reward = -demand_penalty
        else:
            overproduction_penalty = (demand_ratio - 1.1) * 8.0
            demand_reward = -overproduction_penalty

        temp_deviations = np.abs(zone_temps - 85.0)
        temp_stability_score = 1.0 - np.mean(temp_deviations) / 85.0
        stability_reward = temp_stability_score * 5.0

        return {
            'cost_reward': float(cost_reward),
            'cost_component': float(self.cost_weight * cost_reward),
            'efficiency_reward': float(efficiency_reward),
            'efficiency_component': float(self.efficiency_weight * efficiency_reward),
            'demand_reward': float(demand_reward),
            'demand_component': float(self.demand_weight * demand_reward),
            'stability_reward': float(stability_reward),
            'stability_component': float(self.stability_weight * stability_reward),
            'total_reward_unscaled': float(self.cost_weight * cost_reward +
                                           self.efficiency_weight * efficiency_reward +
                                           self.demand_weight * demand_reward +
                                           self.stability_weight * stability_reward)
        }

    def _get_cost_analysis(self, info: Dict) -> Dict[str, Any]:
        cost = info['cost']
        total_production = info['total_production']
        total_demand = info['total_demand']

        daily_cost_estimate = cost * 24
        hourly_cost_per_mw = cost / max(total_production, 0.1)

        vs_target = daily_cost_estimate - self.target_daily_cost
        cost_efficiency = cost / max(info['efficiency'], 0.01)

        return {
            'hourly_cost': float(cost),
            'daily_cost_estimate': float(daily_cost_estimate),
            'vs_target': float(vs_target),
            'target_achieved': daily_cost_estimate <= self.target_daily_cost,
            'hourly_cost_per_mw': float(hourly_cost_per_mw),
            'cost_efficiency_ratio': float(cost_efficiency),
            'cost_optimization_active': self.cost_optimization
        }

    def _update_cost_tracking(self, action: np.ndarray, info: Dict, reward: float):
        self.cost_history.append(info['cost'])
        self.efficiency_history.append(info['efficiency'])

        demand_ratio = info['total_production'] / max(info['total_demand'], 1.0)
        self.demand_satisfaction_history.append(demand_ratio)

        self.episode_stats['cost_components'].append(self._get_cost_analysis(info))
        self.episode_stats['efficiency_scores'].append(info['efficiency'])
        self.episode_stats['demand_ratios'].append(demand_ratio)
        self.episode_stats['reward_breakdown'].append(self._get_reward_breakdown(action, info))

        if hasattr(self, 'previous_action'):
            action_consistency = 1.0 - np.mean(np.abs(action - self.previous_action))
            self.episode_stats['action_consistency'].append(action_consistency)

    def get_cost_performance_summary(self) -> Dict[str, Any]:
        if not self.cost_history:
            return {'error': 'No episode data available'}

        cost_array = np.array(self.cost_history)
        efficiency_array = np.array(self.efficiency_history)
        demand_array = np.array(self.demand_satisfaction_history)

        avg_hourly_cost = np.mean(cost_array)
        estimated_daily_cost = avg_hourly_cost * 24
        cost_std = np.std(cost_array)

        avg_efficiency = np.mean(efficiency_array)
        avg_demand_satisfaction = np.mean(demand_array)

        cost_efficiency_ratio = estimated_daily_cost / max(avg_efficiency, 0.01)

        vs_target = estimated_daily_cost - self.target_daily_cost

        return {
            'episode_length': len(self.cost_history),
            'avg_hourly_cost': float(avg_hourly_cost),
            'estimated_daily_cost': float(estimated_daily_cost),
            'cost_std': float(cost_std),
            'avg_efficiency': float(avg_efficiency),
            'avg_demand_satisfaction': float(avg_demand_satisfaction),
            'cost_efficiency_ratio': float(cost_efficiency_ratio),
            'vs_target_cost': float(vs_target),
            'target_achieved': estimated_daily_cost <= self.target_daily_cost,
            'cost_optimization_enabled': self.cost_optimization,
            'cost_weight': self.cost_weight if hasattr(self, 'cost_weight') else None
        }

    def set_evaluation_mode(self, eval_mode: bool = True):
        if eval_mode:
            self.action_smoothing = 0.05
            print("Environment set to evaluation mode (reduced action smoothing)")
        else:
            if self.cost_optimization:
                self.action_smoothing = 0.12
            print("Environment set to training mode")

def create_enhanced_sac_environment(training: bool = True, **kwargs) -> EnhancedSACDistrictHeatingEnv:
    env_config = get_enhanced_environment_config()

    if training:
        start_date = env_config['train_start_date']
        cost_optimization = True
    else:
        start_date = env_config['eval_start_date']
        cost_optimization = True

    parent_dir = Path(__file__).parent.parent
    data_file = str(parent_dir / env_config['data_file'])
    svr_model_path = str(parent_dir / env_config['svr_model_path'])

    config = {
        'data_file': data_file,
        'svr_model_path': svr_model_path,
        'episode_length': env_config['episode_length'],
        'start_date': start_date,
        'cost_optimization': cost_optimization
    }
    config.update(kwargs)

    return EnhancedSACDistrictHeatingEnv(**config)

def test_enhanced_sac_environment():
    print("Testing SAC District Heating Environment...")
    print("=" * 60)

    try:
        env_config = get_enhanced_environment_config()
        parent_dir = Path(__file__).parent.parent
        data_file = parent_dir / env_config['data_file']
        svr_file = parent_dir / env_config['svr_model_path']

        is_valid = ValidationUtils.validate_environment_setup(
            str(data_file),
            str(svr_file)
        )

        if not is_valid:
            print("Environment setup validation failed!")
            return False

        env = create_enhanced_sac_environment(training=True)

        print("Environment created successfully!")
        print(f"  - Cost optimization: {env.cost_optimization}")
        print(f"  - Target daily cost: {env.target_daily_cost}€")
        print(f"  - Cost weight: {env.cost_weight}")
        print(f"  - Observation space: {env.observation_space.shape}")
        print(f"  - Action space: {env.action_space.shape}")

        obs, info = env.reset()
        print(f"  - Reset successful: obs shape {obs.shape}")
        print(f"  - Info keys: {list(info.keys())}")

        print(f"\nTesting cost optimization steps...")
        total_reward = 0
        cost_history = []

        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            daily_cost_est = info['cost_analysis']['daily_cost_estimate']
            cost_history.append(daily_cost_est)

            print(f"Step {step+1}: reward={reward:.2f}, daily_cost_est={daily_cost_est:.1f}€, "
                  f"efficiency={info['efficiency']:.2f}")

            if terminated or truncated:
                break

        summary = env.get_cost_performance_summary()
        print(f"\nCost Performance Summary:")
        print(f"  - Estimated daily cost: {summary['estimated_daily_cost']:.2f}€")
        print(f"  - vs Target (25€): {summary['vs_target_cost']:+.2f}€")
        print(f"  - Target achieved: {summary['target_achieved']}")

        print(f"\nSAC environment test completed successfully!")
        print(f"Average reward: {total_reward/5:.2f}")
        print(f"Average daily cost estimate: {np.mean(cost_history):.2f}€")

        return True

    except Exception as e:
        print(f"SAC environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("SAC District Heating Environment")
    print("Cost Optimization Focus")
    print("=" * 50)

    success = test_enhanced_sac_environment()

    if success:
        print(f"\n✓ SAC environment ready for training!")
        print(f"\nKey Cost Optimizations:")
        print(f"1. Cost weight: 60% (vs typical 40%)")
        print(f"2. Target daily cost: <25€")
        print(f"3. Enhanced cost penalty for exceeding targets")
        print(f"4. Efficiency bonuses for cost-effective operation")

        print(f"\nNext steps:")
        print(f"1. Test setup: enhanced_test_sac.py")
        print(f"2. Start training: enhanced_sac_training.py")
    else:
        print(f"\n✗ SAC environment test failed!")
        print(f"Please check the error messages and fix issues before proceeding.")