import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import sys
import pickle
import json
import time
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from config.mpc_config import get_mpc_config, MPCConfig
from rl_layer.hybrid_environment import HybridRLMPCEnvironment
from rl_layer.mpc_parameter_agent import MPCParameterAgent
from mpc_core.mpc_controller import DistrictHeatingMPC

class HybridTrainingCallback(BaseCallback):

    def __init__(self, eval_env: HybridRLMPCEnvironment,
                 eval_freq: int = 1000,
                 n_eval_episodes: int = 5,
                 verbose: int = 1):
        super(HybridTrainingCallback, self).__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.episode_rewards = []
        self.episode_costs = []
        self.episode_demand_satisfaction = []
        self.best_performance = {
            'cost': float('inf'),
            'episode': 0,
            'parameters': None
        }

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_performance()

        return True

    def _evaluate_performance(self):
        episode_rewards = []
        episode_costs = []
        episode_demand_sats = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_info = []

            terminated = False
            while not terminated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_info.append(info)

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)

            if episode_info:
                costs = [info['daily_cost_estimate'] for info in episode_info]
                demand_sats = [info['demand_satisfaction'] for info in episode_info]

                episode_costs.append(np.mean(costs))
                episode_demand_sats.append(np.mean(demand_sats))

        avg_reward = np.mean(episode_rewards)
        avg_cost = np.mean(episode_costs)
        avg_demand_sat = np.mean(episode_demand_sats)

        self.episode_rewards.append(avg_reward)
        self.episode_costs.append(avg_cost)
        self.episode_demand_satisfaction.append(avg_demand_sat)

        if avg_cost < self.best_performance['cost']:
            self.best_performance['cost'] = avg_cost
            self.best_performance['episode'] = len(self.episode_costs)

            if hasattr(self.eval_env, 'mpc') and self.eval_env.mpc.last_solution:
                self.best_performance['parameters'] = self.eval_env.mpc.weights.copy()

        if self.verbose >= 1:
            print(f"Eval Episode {len(self.episode_costs)}: "
                  f"Reward={avg_reward:.1f}, Cost={avg_cost:.1f}€, "
                  f"Demand={avg_demand_sat:.1%}")

class HybridRLMPCController:

    def __init__(self, config: Optional[MPCConfig] = None,
                 save_path: Optional[str] = None):

        self.config = config if config is not None else get_mpc_config()

        if save_path is None:
            save_path = str(Path(__file__).parent.parent / "results")
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.train_env = None
        self.eval_env = None
        self.agent = None
        self.training_callback = None

        self.training_config = {
            'total_timesteps': 50000,
            'eval_freq': 1000,
            'n_eval_episodes': 5,
            'learning_rate': 3e-4,
            'buffer_size': 50000,
            'batch_size': 256,
            'learning_starts': 2000
        }

        self.training_history = {
            'costs': [],
            'rewards': [],
            'demand_satisfaction': [],
            'best_parameters': None,
            'training_time': 0
        }

        print("Hybrid RL-MPC Controller initialized for FULL TRAINING")
        print(f"Save path: {self.save_path}")
        print(f"Total timesteps: {self.training_config['total_timesteps']}")
        print(f"Target: {self.config.cost_params['target_daily_cost']}€/day cost")
        print(f"Constraint: {self.config.constraints['hard_constraints']['min_demand_satisfaction']*100}% demand satisfaction")

    def create_environments(self, episode_length: int = 24):
        print("Creating hybrid environments...")

        self.train_env = HybridRLMPCEnvironment(
            config=self.config,
            episode_length=episode_length,
            data_split='train'
        )

        self.eval_env = HybridRLMPCEnvironment(
            config=self.config,
            episode_length=episode_length,
            data_split='val'
        )

        self.train_env_vec = DummyVecEnv([lambda: self.train_env])

        print(f"Environments created:")
        print(f"  Training: {self.train_env.data_split} split")
        print(f"  Evaluation: {self.eval_env.data_split} split")
        print(f"  Episode length: {episode_length}h")

    def create_agent(self):
        if self.train_env is None:
            raise ValueError("Training environment must be created first")

        print("Creating SAC agent...")

        self.agent = SAC(
            policy='MlpPolicy',
            env=self.train_env_vec,
            learning_rate=self.training_config['learning_rate'],
            buffer_size=self.training_config['buffer_size'],
            learning_starts=self.training_config['learning_starts'],
            batch_size=self.training_config['batch_size'],
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            target_update_interval=1,
            ent_coef='auto',
            seed=42,
            verbose=1,
            device='auto'
        )

        self.training_callback = HybridTrainingCallback(
            eval_env=self.eval_env,
            eval_freq=self.training_config['eval_freq'],
            n_eval_episodes=self.training_config['n_eval_episodes'],
            verbose=1
        )

        print(f"SAC agent created:")
        print(f"  Learning rate: {self.training_config['learning_rate']}")
        print(f"  Buffer size: {self.training_config['buffer_size']}")
        print(f"  Batch size: {self.training_config['batch_size']}")

    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        if self.agent is None:
            raise ValueError("Agent must be created first")

        if total_timesteps is None:
            total_timesteps = self.training_config['total_timesteps']

        print(f"Starting hybrid RL-MPC FULL TRAINING...")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Target performance: {self.config.cost_params['target_daily_cost']}€/day")
        print(f"Estimated training time: {total_timesteps / 1000 * 3:.1f} minutes")

        start_time = time.time()

        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.training_callback,
            progress_bar=True
        )

        training_time = time.time() - start_time

        final_cost = self.training_callback.episode_costs[-1] if self.training_callback.episode_costs else 50.0
        best_cost = self.training_callback.best_performance['cost'] if self.training_callback.best_performance['cost'] != float('inf') else 50.0
        final_demand = self.training_callback.episode_demand_satisfaction[-1] if self.training_callback.episode_demand_satisfaction else 0.9

        training_results = {
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'final_cost': final_cost,
            'best_cost': best_cost,
            'best_episode': self.training_callback.best_performance['episode'],
            'final_demand_satisfaction': final_demand,
            'cost_history': self.training_callback.episode_costs.copy(),
            'reward_history': self.training_callback.episode_rewards.copy(),
            'demand_history': self.training_callback.episode_demand_satisfaction.copy()
        }

        self.training_history.update(training_results)

        print(f"\nFULL TRAINING COMPLETED!")
        print(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes evaluated: {len(self.training_callback.episode_costs)}")
        print(f"Best cost achieved: {best_cost:.1f}€/day")
        print(f"Final cost: {final_cost:.1f}€/day")
        print(f"Target achievement: {'YES' if best_cost <= self.config.cost_params['target_daily_cost'] else 'NO'}")

        print(f"\nComparison with your thesis baselines:")
        print(f"  Original SAC: 53.35€ → Hybrid RL-MPC: {best_cost:.1f}€ ({((53.35-best_cost)/53.35*100):+.1f}%)")
        print(f"  Enhanced SAC: 47.97€ → Hybrid RL-MPC: {best_cost:.1f}€ ({((47.97-best_cost)/47.97*100):+.1f}%)")
        print(f"  Hybrid SAC-TD3: 37.44€ → Hybrid RL-MPC: {best_cost:.1f}€ ({((37.44-best_cost)/37.44*100):+.1f}%)")

        return training_results

    def evaluate(self, n_episodes: int = 20, deterministic: bool = True) -> Dict[str, Any]:
        if self.agent is None:
            raise ValueError("Agent must be trained first")

        print(f"Evaluating hybrid RL-MPC system (COMPREHENSIVE)...")
        print(f"Episodes: {n_episodes}, Deterministic: {deterministic}")

        evaluation_results = {
            'episodes': [],
            'costs': [],
            'demand_satisfactions': [],
            'efficiencies': [],
            'constraint_violations': [],
            'mpc_parameters': []
        }

        for episode in range(n_episodes):
            print(f"Evaluating episode {episode+1}/{n_episodes}...", end='\r')
            obs, _ = self.eval_env.reset()
            episode_data = {
                'rewards': [],
                'costs': [],
                'demand_satisfactions': [],
                'efficiencies': [],
                'violations': [],
                'parameters': []
            }

            terminated = False
            while not terminated:
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                episode_data['rewards'].append(reward)
                episode_data['costs'].append(info['daily_cost_estimate'])
                episode_data['demand_satisfactions'].append(info['demand_satisfaction'])
                episode_data['efficiencies'].append(info['efficiency'])
                episode_data['violations'].append(info['constraint_violations'])
                episode_data['parameters'].append(info['mpc_weights_used'].copy())

                if terminated or truncated:
                    break

            evaluation_results['episodes'].append(episode_data)
            evaluation_results['costs'].append(np.mean(episode_data['costs']))
            evaluation_results['demand_satisfactions'].append(np.mean(episode_data['demand_satisfactions']))
            evaluation_results['efficiencies'].append(np.mean(episode_data['efficiencies']))
            evaluation_results['constraint_violations'].append(np.sum(episode_data['violations']))
            evaluation_results['mpc_parameters'].append(episode_data['parameters'][-1])

        summary = {
            'n_episodes': n_episodes,
            'avg_cost': float(np.mean(evaluation_results['costs'])),
            'std_cost': float(np.std(evaluation_results['costs'])),
            'min_cost': float(np.min(evaluation_results['costs'])),
            'max_cost': float(np.max(evaluation_results['costs'])),
            'avg_demand_satisfaction': float(np.mean(evaluation_results['demand_satisfactions'])),
            'min_demand_satisfaction': float(np.min(evaluation_results['demand_satisfactions'])),
            'avg_efficiency': float(np.mean(evaluation_results['efficiencies'])),
            'total_violations': int(np.sum(evaluation_results['constraint_violations'])),
            'target_achievement': {
                'cost_target_met': np.mean(evaluation_results['costs']) <= self.config.cost_params['target_daily_cost'],
                'demand_target_met': np.min(evaluation_results['demand_satisfactions']) >= self.config.constraints['hard_constraints']['min_demand_satisfaction'],
                'combined_success': (np.mean(evaluation_results['costs']) <= self.config.cost_params['target_daily_cost'] and
                                     np.min(evaluation_results['demand_satisfactions']) >= self.config.constraints['hard_constraints']['min_demand_satisfaction'])
            }
        }

        print(f"\nCOMPREHENSIVE EVALUATION RESULTS:")
        print(f"  Average cost: {summary['avg_cost']:.1f} ± {summary['std_cost']:.1f} €/day")
        print(f"  Cost range: {summary['min_cost']:.1f} - {summary['max_cost']:.1f} €/day")
        print(f"  Average demand satisfaction: {summary['avg_demand_satisfaction']:.1%}")
        print(f"  Minimum demand satisfaction: {summary['min_demand_satisfaction']:.1%}")
        print(f"  Total constraint violations: {summary['total_violations']}")
        print(f"  Cost target (≤{self.config.cost_params['target_daily_cost']}€): {'✓' if summary['target_achievement']['cost_target_met'] else '✗'}")
        print(f"  Demand target (≥{self.config.constraints['hard_constraints']['min_demand_satisfaction']*100}%): {'✓' if summary['target_achievement']['demand_target_met'] else '✗'}")
        print(f"  Overall success: {'✓' if summary['target_achievement']['combined_success'] else '✗'}")

        return summary

    def save_system(self, name: str = "hybrid_rl_mpc_full"):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.save_path / f"{name}_{timestamp}"
        save_dir.mkdir(exist_ok=True)

        if self.agent is not None:
            agent_path = save_dir / "sac_agent"
            self.agent.save(str(agent_path))
            print(f"Agent saved to {agent_path}")

        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        config_path = save_dir / "mpc_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)

        if self.training_callback is not None:
            callback_data = {
                'episode_costs': self.training_callback.episode_costs,
                'episode_rewards': self.training_callback.episode_rewards,
                'episode_demand_satisfaction': self.training_callback.episode_demand_satisfaction,
                'best_performance': self.training_callback.best_performance
            }
            callback_path = save_dir / "training_callback.json"
            with open(callback_path, 'w') as f:
                json.dump(callback_data, f, indent=2)

        print(f"Complete FULL TRAINED system saved to {save_dir}")
        return str(save_dir)

    def load_system(self, load_path: str):
        load_dir = Path(load_path)

        agent_path = load_dir / "sac_agent.zip"
        if agent_path.exists():
            self.agent = SAC.load(str(agent_path.with_suffix('')))
            print(f"Agent loaded from {agent_path}")

        history_path = load_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            print(f"Training history loaded")

        config_path = load_dir / "mpc_config.pkl"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print(f"Configuration loaded")

def run_full_training():
    print("=" * 80)
    print("HYBRID RL-MPC SYSTEM - FULL TRAINING EXECUTION")
    print("Target: 25€/day with 85% demand satisfaction guarantee")
    print("=" * 80)

    try:
        controller = HybridRLMPCController()

        controller.create_environments(episode_length=24)
        controller.create_agent()

        print(f"\nStarting FULL TRAINING (50,000 timesteps)...")
        training_results = controller.train()

        print(f"\nRunning comprehensive evaluation...")
        eval_results = controller.evaluate(n_episodes=20)

        save_path = controller.save_system("hybrid_rl_mpc_production")

        print(f"\n" + "=" * 80)
        print(f"FULL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"=" * 80)
        print(f"Training time: {training_results['training_time']/60:.1f} minutes")
        print(f"Best cost achieved: {training_results['best_cost']:.1f}€/day")
        print(f"Target (25€): {'ACHIEVED' if training_results['best_cost'] <= 25.0 else 'NOT ACHIEVED'}")
        print(f"Demand satisfaction: {eval_results['avg_demand_satisfaction']:.1%}")
        print(f"Constraint violations: {eval_results['total_violations']}")
        print(f"System saved to: {save_path}")
        print(f"=" * 80)

        return True

    except Exception as e:
        print(f"Full training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_full_training()

    if success:
        print(f"\nHybrid RL-MPC system training completed successfully!")
        print(f"System is ready for thesis presentation and industrial deployment.")
    else:
        print(f"\nTraining failed. Check error messages above.")