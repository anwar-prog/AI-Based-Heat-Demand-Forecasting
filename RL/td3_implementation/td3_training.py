import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from typing import Dict, Tuple, Any
import warnings
import os
import time
from datetime import datetime
import json

from td3_env import create_td3_environment, TD3DistrictHeatingEnv
from td3_config import get_td3_hyperparameters, get_training_config, get_evaluation_config
from common.evaluation_metrics import DistrictHeatingEvaluator
from common.utils import ValidationUtils, ModelUtils, create_timestamp

warnings.filterwarnings("ignore", category=UserWarning)

class EnhancedTD3DistrictHeatingEnv(TD3DistrictHeatingEnv):

    def _calculate_td3_reward(self, action: np.ndarray, info: Dict, base_reward: float) -> float:
        total_production = info['total_production']
        total_demand = info['total_demand']
        cost = info['cost']
        efficiency = info['efficiency']
        zone_temps = info['zone_temps']

        if total_production > 1.0:
            production_bonus = 15.0 + min(total_production * 0.1, 10.0)
        else:
            production_bonus = -30.0

        efficiency_reward = efficiency * self.efficiency_bonus * 1.2

        demand_ratio = total_production / max(total_demand, 1.0)
        if 0.6 <= demand_ratio <= 1.4:
            demand_reward = 15.0 - 5.0 * abs(demand_ratio - 1.0)
        elif demand_ratio < 0.6:
            demand_reward = -10.0 * (0.6 - demand_ratio)
        else:
            demand_reward = -5.0 * (demand_ratio - 1.4)

        temp_deviations = np.abs(zone_temps - self.temp_target)
        avg_deviation = np.mean(temp_deviations)
        temp_reward = max(0, (self.temp_tolerance - avg_deviation) / self.temp_tolerance) * 8.0

        action_consistency = self._calculate_action_consistency(action)
        consistency_reward = action_consistency * self.action_consistency_weight * 25.0

        cost_penalty = -cost / 150.0

        action_diversity = np.std(action)
        diversity_bonus = min(action_diversity * 5.0, 3.0)

        total_reward = (
                production_bonus * 1.5 +
                efficiency_reward +
                demand_reward +
                temp_reward +
                consistency_reward +
                cost_penalty +
                diversity_bonus
        )

        return total_reward

def create_enhanced_td3_environment(training: bool = True, **kwargs):
    from td3_config import get_environment_config
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

    return EnhancedTD3DistrictHeatingEnv(**config)

class TD3Trainer:

    def __init__(self):
        self.timestamp = create_timestamp()
        self.td3_config = get_td3_hyperparameters()
        self.training_config = get_training_config()
        self.eval_config = get_evaluation_config()

        self.enhanced_td3_config = self.td3_config.copy()
        self.enhanced_td3_config.update({
            'target_policy_noise': 0.3,
            'target_noise_clip': 0.8,
            'learning_starts': 5000,
            'batch_size': 128,
            'tau': 0.01,
        })

        self.results_dir = ModelUtils.create_results_directory("td3_implementation", "td3")

        print(f"Enhanced TD3 Trainer initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Timestamp: {self.timestamp}")

    def validate_setup(self) -> bool:
        print("Validating TD3 training setup...")

        env_valid = ValidationUtils.validate_environment_setup(
            "data/schweinfurt_data.csv",
            "data/svr_model_33features.pkl"
        )

        if not env_valid:
            return False

        try:
            test_env = create_enhanced_td3_environment(training=True)
            env_compatible = ValidationUtils.test_environment_compatibility(
                EnhancedTD3DistrictHeatingEnv,
                {
                    'data_file': 'data/schweinfurt_data.csv',
                    'svr_model_path': 'data/svr_model_33features.pkl',
                    'start_date': '2021-01-01'
                }
            )
        except Exception as e:
            print(f"Environment compatibility test failed: {e}")
            return False

        if not env_compatible:
            return False

        print("Setup validation passed!")
        return True

    def create_environments(self):
        print("Creating enhanced TD3 environments...")

        train_env = create_enhanced_td3_environment(training=True, start_date='2021-01-01')
        train_env = Monitor(
            train_env,
            filename=str(self.results_dir / "logs" / f"train_log_{self.timestamp}")
        )

        eval_env = create_enhanced_td3_environment(training=False, start_date='2024-01-01')
        eval_env = Monitor(
            eval_env,
            filename=str(self.results_dir / "logs" / f"eval_log_{self.timestamp}")
        )

        print("Enhanced environments created successfully")
        print(f"  - Training environment: {train_env.env.__class__.__name__}")
        print(f"  - Evaluation environment: {eval_env.env.__class__.__name__}")
        print("  - Enhanced reward structure applied")

        return train_env, eval_env

    def create_td3_model(self, train_env):
        print("Creating enhanced TD3 model...")

        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.15 * np.ones(n_actions)
        )

        model = TD3(
            'MlpPolicy',
            train_env,
            action_noise=action_noise,
            **self.enhanced_td3_config,
            tensorboard_log=str(self.results_dir / "logs"),
            verbose=1
        )

        print("Enhanced TD3 model created successfully")
        print(f"  - Policy architecture: {self.enhanced_td3_config['policy_kwargs']['net_arch']}")
        print(f"  - Learning rate: {self.enhanced_td3_config['learning_rate']}")
        print(f"  - Buffer size: {self.enhanced_td3_config['buffer_size']:,}")
        print(f"  - Batch size: {self.enhanced_td3_config['batch_size']}")
        print(f"  - Policy delay: {self.enhanced_td3_config['policy_delay']}")
        print(f"  - Target policy noise: {self.enhanced_td3_config['target_policy_noise']}")
        print(f"  - Learning starts: {self.enhanced_td3_config['learning_starts']}")
        print(f"  - Action noise std: 0.15")
        print(f"  - Device: {model.device}")

        return model

    def setup_callbacks(self, eval_env):
        print("Setting up training callbacks...")

        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=200.0,
            verbose=1
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.results_dir / "models" / "best_model"),
            log_path=str(self.results_dir / "evaluations"),
            eval_freq=self.training_config['eval_freq'],
            n_eval_episodes=self.training_config['n_eval_episodes'],
            deterministic=self.training_config['eval_deterministic'],
            render=False,
            callback_on_new_best=stop_callback,
            verbose=1
        )

        callback_list = CallbackList([eval_callback])

        print("Callbacks configured:")
        print(f"  - Evaluation frequency: {self.training_config['eval_freq']:,} steps")
        print(f"  - Evaluation episodes: {self.training_config['n_eval_episodes']}")
        print(f"  - Reward threshold: 200.0 (increased for better learning)")

        return callback_list

    def train_td3_agent(self):
        print("Starting Enhanced TD3 Training Pipeline")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not self.validate_setup():
            print("Setup validation failed. Aborting training.")
            return None

        train_env, eval_env = self.create_environments()

        model = self.create_td3_model(train_env)

        callbacks = self.setup_callbacks(eval_env)

        print(f"\nEnhanced Training Configuration:")
        print(f"  - Total timesteps: {self.training_config['total_timesteps']:,}")
        print(f"  - Warmup timesteps: {self.enhanced_td3_config['learning_starts']:,}")
        print(f"  - Expected training time: ~{self.estimate_training_time():.1f} minutes")
        print(f"  - Enhanced reward structure: Active")
        print(f"  - Increased exploration noise: Active")
        print(f"  - Modified hyperparameters: Active")

        print(f"\nStarting enhanced TD3 training...")
        start_time = time.time()

        try:
            model.learn(
                total_timesteps=self.training_config['total_timesteps'],
                callback=callbacks,
                progress_bar=self.training_config['progress_bar']
            )

            training_time = time.time() - start_time
            print(f"Training completed in {training_time/60:.1f} minutes!")

        except KeyboardInterrupt:
            training_time = time.time() - start_time
            print(f"\nTraining interrupted after {training_time/60:.1f} minutes")
            print("Saving current model...")
        except Exception as e:
            print(f"Training failed: {e}")
            raise

        final_model_path = self.results_dir / "models" / f"final_td3_model_{self.timestamp}"
        model.save(str(final_model_path))
        print(f"Final model saved: {final_model_path}.zip")

        best_model_path = self.results_dir / "models" / "best_model" / "best_model.zip"
        if best_model_path.exists():
            model_path = str(best_model_path)
            print(f"Using best model: {model_path}")
        else:
            model_path = str(final_model_path) + ".zip"
            print(f"Using final model: {model_path}")

        return model_path, training_time

    def evaluate_trained_model(self, model_path: str):
        print(f"\nEvaluating Trained Enhanced TD3 Model")
        print("=" * 50)

        evaluator = DistrictHeatingEvaluator(results_dir=str(self.results_dir / "evaluations"))

        env_kwargs = {
            'data_file': 'data/schweinfurt_data.csv',
            'svr_model_path': 'data/svr_model_33features.pkl',
            'start_date': '2024-06-01'
        }

        print("Evaluating TD3 performance...")
        td3_results = evaluator.evaluate_model(
            model_path,
            EnhancedTD3DistrictHeatingEnv,
            env_kwargs,
            episodes=self.eval_config['n_eval_episodes'],
            algorithm='TD3'
        )

        print("Comparing with baseline...")
        comparison = evaluator.compare_with_baseline(
            model_path,
            EnhancedTD3DistrictHeatingEnv,
            env_kwargs,
            episodes=5,
            algorithm='TD3'
        )

        return td3_results, comparison

    def compare_with_sac(self, td3_results):
        print(f"\nComparing Enhanced TD3 with SAC Results")
        print("=" * 50)

        sac_baseline = {
            'avg_reward': 2593.13,
            'avg_cost': 53.35,
            'avg_efficiency': 0.581
        }

        td3_summary = td3_results['summary']

        print(f"Algorithm Performance Comparison:")
        print(f"{'Metric':<20} {'TD3':<12} {'SAC':<12} {'Difference':<12}")
        print("-" * 56)

        reward_diff = ((td3_summary['avg_reward'] - sac_baseline['avg_reward']) / abs(sac_baseline['avg_reward'])) * 100
        cost_diff = ((td3_summary['avg_cost'] - sac_baseline['avg_cost']) / max(sac_baseline['avg_cost'], 0.01)) * 100
        efficiency_diff = ((td3_summary['avg_efficiency'] - sac_baseline['avg_efficiency']) / max(sac_baseline['avg_efficiency'], 0.01)) * 100

        print(f"{'Average Reward':<20} {td3_summary['avg_reward']:>8.2f}   {sac_baseline['avg_reward']:>8.2f}   {reward_diff:>+8.1f}%")
        print(f"{'Daily Cost (€)':<20} {td3_summary['avg_cost']:>8.2f}   {sac_baseline['avg_cost']:>8.2f}   {cost_diff:>+8.1f}%")
        print(f"{'Efficiency':<20} {td3_summary['avg_efficiency']:>8.3f}   {sac_baseline['avg_efficiency']:>8.3f}   {efficiency_diff:>+8.1f}%")

        print(f"\nAlgorithm Analysis:")
        print(f"Enhanced TD3 Performance Assessment:")

        if td3_summary['avg_efficiency'] > 0.1:
            if abs(reward_diff) < 20:
                print("- TD3 demonstrates competitive performance with SAC")
            elif reward_diff > 0:
                print("- TD3's deterministic policy shows advantages in this domain")
            else:
                print("- SAC's stochastic exploration still provides benefits")

            if td3_summary['avg_cost'] < sac_baseline['avg_cost']:
                print("- TD3 achieves superior cost optimization")
            else:
                print("- SAC maintains cost efficiency advantages")

            if td3_summary['avg_efficiency'] > sac_baseline['avg_efficiency']:
                print("- TD3's consistent policy improves operational efficiency")
            else:
                print("- SAC's exploration strategy benefits overall efficiency")
        else:
            print("- TD3 still struggles with exploration despite enhancements")
            print("- District heating domain particularly benefits from stochastic policies")

        return {
            'reward_diff': reward_diff,
            'cost_diff': cost_diff,
            'efficiency_diff': efficiency_diff
        }

    def generate_training_report(self, model_path: str, training_time: float,
                                 td3_results: dict, comparison: dict, sac_comparison: dict):
        print(f"\nGenerating Enhanced TD3 Training Report")
        print("=" * 40)

        report = {
            'training_info': {
                'algorithm': 'Enhanced TD3',
                'timestamp': self.timestamp,
                'training_time_minutes': training_time / 60,
                'model_path': model_path,
                'enhancements': {
                    'enhanced_reward_structure': True,
                    'increased_exploration_noise': True,
                    'modified_hyperparameters': self.enhanced_td3_config,
                    'early_stopping_threshold': 200.0
                },
                'configuration': {
                    'td3_hyperparameters': self.enhanced_td3_config,
                    'training_config': self.training_config,
                    'evaluation_config': self.eval_config
                }
            },
            'results': {
                'td3_performance': td3_results,
                'baseline_comparison': comparison,
                'sac_comparison': sac_comparison
            },
            'summary': {
                'avg_reward': td3_results['summary']['avg_reward'],
                'avg_cost': td3_results['summary']['avg_cost'],
                'avg_efficiency': td3_results['summary']['avg_efficiency'],
                'cost_reduction_vs_baseline': comparison['improvements']['cost_reduction'],
                'reward_vs_sac': sac_comparison['reward_diff'],
                'training_successful': td3_results['summary']['avg_efficiency'] > 0.1
            }
        }

        report_path = self.results_dir / f"enhanced_td3_training_report_{self.timestamp}.json"
        ModelUtils.save_model_results(report, str(report_path))

        print(f"Training report saved: {report_path}")

        return report

    def print_final_summary(self, report: dict):
        print(f"\nENHANCED TD3 TRAINING COMPLETED")
        print("=" * 60)
        print(f"Training time: {report['training_info']['training_time_minutes']:.1f} minutes")
        print(f"Model saved: {report['training_info']['model_path']}")

        summary = report['summary']
        print(f"\nFinal Performance:")
        print(f"  - Average Reward: {summary['avg_reward']:.2f}")
        print(f"  - Average Cost: {summary['avg_cost']:.2f} €")
        print(f"  - Average Efficiency: {summary['avg_efficiency']:.3f}")
        print(f"  - Cost Reduction vs Baseline: {summary['cost_reduction_vs_baseline']:+.1f}%")
        print(f"  - Performance vs SAC: {summary['reward_vs_sac']:+.1f}%")

        print(f"\nEnhancements Applied:")
        enhancements = report['training_info']['enhancements']
        print(f"  - Enhanced reward structure: {enhancements['enhanced_reward_structure']}")
        print(f"  - Increased exploration noise: {enhancements['increased_exploration_noise']}")
        print(f"  - Modified hyperparameters: Active")

        print(f"\nResults saved to: {self.results_dir}")

        if summary.get('training_successful', False):
            print(f"\nEnhanced TD3 training completed successfully!")
            print(f"Next steps:")
            print(f"  1. Analyze enhanced results in {self.results_dir}")
            print(f"  2. Compare enhanced TD3 vs SAC performance")
            print(f"  3. Document algorithm improvements for thesis")
            print(f"  4. Generate comparative analysis charts")
        else:
            print(f"\nTD3 still showing challenges despite enhancements.")
            print(f"This provides valuable insights about algorithm-domain fit for thesis discussion.")

    def estimate_training_time(self) -> float:
        base_time_per_1k_steps = 0.95
        total_steps = self.training_config['total_timesteps']
        return (total_steps / 1000) * base_time_per_1k_steps

def main():
    print("Enhanced TD3 District Heating Training Pipeline")
    print("Schweinfurt Network Optimization")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    trainer = TD3Trainer()

    try:
        model_path, training_time = trainer.train_td3_agent()

        if model_path is None:
            print("Training failed. Exiting.")
            return

        td3_results, comparison = trainer.evaluate_trained_model(model_path)

        sac_comparison = trainer.compare_with_sac(td3_results)

        report = trainer.generate_training_report(
            model_path, training_time, td3_results, comparison, sac_comparison
        )

        trainer.print_final_summary(report)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()