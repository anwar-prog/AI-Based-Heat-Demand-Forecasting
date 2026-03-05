import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.mpc_config import get_mpc_config, MPCConfig

class MPCParameterExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim: int = 256):
        super(MPCParameterExtractor, self).__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

class MPCParameterPolicy(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn = nn.ReLU, *args, **kwargs):

        if net_arch is None:
            net_arch = [
                dict(pi=[256, 128, 64], vf=[256, 128, 64])
            ]

        super(MPCParameterPolicy, self).__init__(
            observation_space, action_space, lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=MPCParameterExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            *args, **kwargs
        )

class MPCParameterAgent:

    def __init__(self, config: Optional[MPCConfig] = None,
                 learning_rate: float = 3e-4,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 target_update_interval: int = 1,
                 ent_coef: str = 'auto',
                 seed: Optional[int] = None):

        self.config = config if config is not None else get_mpc_config()

        self.parameter_bounds = {
            'cost_weight': (0.1, 3.0),
            'comfort_weight': (0.5, 4.0),
            'efficiency_weight': (0.1, 2.0),
            'stability_weight': (0.05, 1.0),
            'prediction_horizon_scale': (0.5, 1.5),
            'risk_factor': (0.1, 1.0)
        }

        self.action_dim = len(self.parameter_bounds)
        self.state_dim = None

        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.ent_coef = ent_coef
        self.seed = seed

        self.agent = None

        self.training_stats = {
            'episode_rewards': [],
            'episode_costs': [],
            'episode_demand_satisfaction': [],
            'parameter_history': [],
            'exploration_noise': []
        }

        self.best_performance = {
            'cost': float('inf'),
            'parameters': None,
            'episode': 0
        }

        print("MPC Parameter Agent initialized")
        print(f"Parameter bounds: {list(self.parameter_bounds.keys())}")
        print(f"Action dimension: {self.action_dim}")

    def initialize_agent(self, observation_space, action_space):
        self.state_dim = observation_space.shape[0]

        self.agent = SAC(
            policy=MPCParameterPolicy,
            env=None,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=1000,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            target_update_interval=self.target_update_interval,
            ent_coef=self.ent_coef,
            seed=self.seed,
            verbose=1
        )

        print(f"SAC agent initialized")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {action_space.shape[0]}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Buffer size: {self.buffer_size}")

    def denormalize_action(self, normalized_action: np.ndarray) -> Dict[str, float]:
        parameters = {}
        param_names = list(self.parameter_bounds.keys())

        for i, param_name in enumerate(param_names):
            min_val, max_val = self.parameter_bounds[param_name]
            param_value = min_val + normalized_action[i] * (max_val - min_val)
            parameters[param_name] = float(param_value)

        return parameters

    def normalize_action(self, parameters: Dict[str, float]) -> np.ndarray:
        normalized = np.zeros(self.action_dim)
        param_names = list(self.parameter_bounds.keys())

        for i, param_name in enumerate(param_names):
            min_val, max_val = self.parameter_bounds[param_name]
            param_value = parameters.get(param_name, (min_val + max_val) / 2)
            normalized[i] = (param_value - min_val) / (max_val - min_val)
            normalized[i] = np.clip(normalized[i], 0.0, 1.0)

        return normalized

    def action_to_mpc_weights(self, action: np.ndarray) -> Dict[str, float]:
        parameters = self.denormalize_action(action)

        mpc_weights = {
            'cost_weight': parameters['cost_weight'],
            'comfort_weight': parameters['comfort_weight'],
            'efficiency_weight': parameters['efficiency_weight'],
            'stability_weight': parameters['stability_weight']
        }

        self.current_parameters = parameters

        return mpc_weights

    def predict_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.agent is None:
            return np.ones(self.action_dim) * 0.5

        action, _ = self.agent.predict(observation, deterministic=deterministic)
        action = np.clip(action, 0.0, 1.0)

        return action

    def update_training_stats(self, episode_reward: float, episode_cost: float,
                              demand_satisfaction: float, parameters: Dict[str, float]):
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_costs'].append(episode_cost)
        self.training_stats['episode_demand_satisfaction'].append(demand_satisfaction)
        self.training_stats['parameter_history'].append(parameters.copy())

        if episode_cost < self.best_performance['cost']:
            self.best_performance['cost'] = episode_cost
            self.best_performance['parameters'] = parameters.copy()
            self.best_performance['episode'] = len(self.training_stats['episode_costs'])

        max_history = 1000
        for key in self.training_stats:
            if len(self.training_stats[key]) > max_history:
                self.training_stats[key] = self.training_stats[key][-max_history:]

    def get_training_summary(self) -> Dict[str, Any]:
        if not self.training_stats['episode_costs']:
            return {'error': 'No training data available'}

        recent_episodes = min(100, len(self.training_stats['episode_costs']))
        recent_costs = self.training_stats['episode_costs'][-recent_episodes:]
        recent_rewards = self.training_stats['episode_rewards'][-recent_episodes:]
        recent_demand = self.training_stats['episode_demand_satisfaction'][-recent_episodes:]

        return {
            'total_episodes': len(self.training_stats['episode_costs']),
            'recent_avg_cost': float(np.mean(recent_costs)),
            'recent_avg_reward': float(np.mean(recent_rewards)),
            'recent_avg_demand_satisfaction': float(np.mean(recent_demand)),
            'best_cost': float(self.best_performance['cost']),
            'best_cost_episode': self.best_performance['episode'],
            'best_parameters': self.best_performance['parameters'],
            'cost_improvement': float(recent_costs[0] - recent_costs[-1]) if len(recent_costs) > 1 else 0.0,
            'recent_cost_std': float(np.std(recent_costs)),
            'parameter_stability': self._calculate_parameter_stability()
        }

    def _calculate_parameter_stability(self) -> Dict[str, float]:
        if len(self.training_stats['parameter_history']) < 10:
            return {}

        recent_params = self.training_stats['parameter_history'][-50:]
        stability = {}

        for param_name in self.parameter_bounds.keys():
            values = [params[param_name] for params in recent_params]
            stability[param_name] = float(np.std(values))

        return stability

    def save_agent(self, save_path: str):
        if self.agent is not None:
            self.agent.save(save_path)
            print(f"Agent saved to {save_path}")
        else:
            print("No agent to save")

    def load_agent(self, load_path: str):
        if Path(load_path).exists():
            self.agent = SAC.load(load_path)
            print(f"Agent loaded from {load_path}")
        else:
            print(f"Agent file not found: {load_path}")

    def get_exploration_action(self, observation: np.ndarray,
                               exploration_noise: float = 0.1) -> np.ndarray:
        action = self.predict_action(observation, deterministic=False)

        noise = np.random.normal(0, exploration_noise, action.shape)
        action_with_noise = action + noise
        action_with_noise = np.clip(action_with_noise, 0.0, 1.0)

        self.training_stats['exploration_noise'].append(float(np.mean(np.abs(noise))))

        return action_with_noise

    def calculate_parameter_reward(self, mpc_results: Dict[str, Any]) -> float:
        daily_cost = mpc_results.get('daily_cost_estimate', 100.0)
        demand_satisfaction = mpc_results.get('demand_satisfaction', 0.0)
        efficiency = mpc_results.get('efficiency', 0.0)
        constraint_violations = mpc_results.get('constraint_violations', {}).get('total_violations', 0)

        target_cost = self.config.cost_params['target_daily_cost']
        cost_ratio = daily_cost / target_cost

        if cost_ratio <= 1.0:
            cost_reward = 20.0 * (1.0 - cost_ratio)
        else:
            cost_reward = -10.0 * (cost_ratio - 1.0)

        min_demand = self.config.constraints['hard_constraints']['min_demand_satisfaction']
        if demand_satisfaction >= min_demand:
            demand_reward = 10.0 + 5.0 * (demand_satisfaction - min_demand) / (1.0 - min_demand)
        else:
            demand_reward = -50.0 * (min_demand - demand_satisfaction)

        efficiency_reward = 5.0 * efficiency
        violation_penalty = -20.0 * constraint_violations
        stability_bonus = 2.0

        total_reward = (
                cost_reward +
                demand_reward +
                efficiency_reward +
                violation_penalty +
                stability_bonus
        )

        return float(total_reward)

    def get_parameter_info(self) -> Dict[str, Any]:
        if hasattr(self, 'current_parameters'):
            return {
                'current_parameters': self.current_parameters,
                'parameter_bounds': self.parameter_bounds,
                'parameter_ranges': {
                    name: bounds[1] - bounds[0]
                    for name, bounds in self.parameter_bounds.items()
                }
            }
        else:
            return {
                'parameter_bounds': self.parameter_bounds,
                'current_parameters': None
            }

def test_mpc_parameter_agent():
    print("Testing MPC Parameter Agent...")
    print("=" * 50)

    try:
        agent = MPCParameterAgent(
            learning_rate=1e-3,
            buffer_size=10000,
            batch_size=64
        )

        print(f"Parameter agent created successfully!")
        print(f"Action dimension: {agent.action_dim}")
        print(f"Parameter bounds: {list(agent.parameter_bounds.keys())}")

        test_params = {
            'cost_weight': 1.5,
            'comfort_weight': 2.0,
            'efficiency_weight': 0.8,
            'stability_weight': 0.4,
            'prediction_horizon_scale': 1.0,
            'risk_factor': 0.5
        }

        print(f"\nTesting parameter conversion...")
        print(f"Original parameters: {test_params}")

        normalized = agent.normalize_action(test_params)
        print(f"Normalized action: {normalized}")

        reconstructed = agent.denormalize_action(normalized)
        print(f"Reconstructed parameters: {reconstructed}")

        mpc_weights = agent.action_to_mpc_weights(normalized)
        print(f"MPC weights: {mpc_weights}")

        dummy_observation = np.random.randn(31)
        action = agent.predict_action(dummy_observation)
        print(f"\nDummy action prediction: {action}")

        exploration_action = agent.get_exploration_action(dummy_observation, exploration_noise=0.1)
        print(f"Action with exploration: {exploration_action}")

        mock_mpc_results = {
            'daily_cost_estimate': 28.5,
            'demand_satisfaction': 0.92,
            'efficiency': 0.85,
            'constraint_violations': {'total_violations': 0}
        }

        reward = agent.calculate_parameter_reward(mock_mpc_results)
        print(f"\nReward calculation test:")
        print(f"Mock MPC results: {mock_mpc_results}")
        print(f"Calculated reward: {reward:.2f}")

        agent.update_training_stats(
            episode_reward=reward,
            episode_cost=mock_mpc_results['daily_cost_estimate'],
            demand_satisfaction=mock_mpc_results['demand_satisfaction'],
            parameters=test_params
        )

        summary = agent.get_training_summary()
        print(f"\nTraining summary:")
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Best cost: {summary['best_cost']:.1f} €")
        print(f"Best parameters: {summary['best_parameters']}")

        param_info = agent.get_parameter_info()
        print(f"\nParameter info:")
        print(f"Current parameters set: {param_info['current_parameters'] is not None}")
        print(f"Parameter ranges: {param_info['parameter_ranges']}")

        print(f"\nMPC parameter agent test completed successfully!")
        return True

    except Exception as e:
        print(f"MPC parameter agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mpc_parameter_agent()

    if success:
        print(f"\nMPC Parameter Agent ready!")
        print(f"Key capabilities:")
        print(f"  - SAC-based parameter optimization")
        print(f"  - MPC weight tuning (cost, comfort, efficiency, stability)")
        print(f"  - Exploration with noise injection")
        print(f"  - Performance tracking and best parameter storage")
        print(f"  - Reward calculation based on cost and constraint satisfaction")
        print(f"  - Parameter stability analysis")
    else:
        print(f"\nMPC parameter agent test failed!")
        print(f"Check stable-baselines3 installation and dependencies.")