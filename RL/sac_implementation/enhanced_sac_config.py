import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from common.utils import ConfigManager
from typing import Dict, Any

class EnhancedSACConfig:

    def __init__(self):
        self.base_config = ConfigManager.get_sac_config()
        self.training_config = ConfigManager.get_training_config()

        self.sac_hyperparameters = self._get_enhanced_sac_params()
        self.environment_config = self._get_enhanced_environment_config()
        self.training_params = self._get_enhanced_training_params()
        self.evaluation_params = self._get_enhanced_evaluation_params()

    def _get_enhanced_sac_params(self) -> Dict[str, Any]:
        config = self.base_config.copy()

        config.update({
            'learning_rate': 2e-4,

            'buffer_size': 200000,
            'batch_size': 512,

            'gamma': 0.99,
            'tau': 0.005,

            'ent_coef': 'auto',
            'target_entropy': 'auto',

            'policy_kwargs': {
                'net_arch': [512, 256, 128],
                'log_std_init': -3,
                'use_sde': False,
            },

            'gradient_steps': 2,
            'train_freq': 1,
            'learning_starts': 20000,

            'device': 'auto',
        })

        return config

    def _get_enhanced_environment_config(self) -> Dict[str, Any]:
        return {
            'episode_length': 24,
            'observation_dim': 31,
            'action_dim': 11,
            'action_bounds': (0.0, 1.0),

            'train_start_date': '2021-01-01',
            'train_end_date': '2023-12-31',
            'eval_start_date': '2024-01-01',
            'test_start_date': '2024-06-01',

            'cost_weight': 0.6,
            'efficiency_weight': 0.25,
            'demand_weight': 0.1,
            'stability_weight': 0.05,

            'data_file': 'data/schweinfurt_data.csv',
            'svr_model_path': 'data/svr_model_33features.pkl',
        }

    def _get_enhanced_training_params(self) -> Dict[str, Any]:
        return {
            'total_timesteps': 300000,
            'warmup_timesteps': 20000,

            'eval_freq': 10000,
            'n_eval_episodes': 10,
            'eval_deterministic': True,

            'reward_threshold': 500.0,
            'patience': 8,
            'cost_threshold': 25.0,

            'log_interval': 5000,
            'save_freq': 25000,
            'tensorboard_log': True,

            'progress_bar': True,
        }

    def _get_enhanced_evaluation_params(self) -> Dict[str, Any]:
        return {
            'n_eval_episodes': 15,
            'eval_deterministic': True,
            'render_episodes': False,

            'track_metrics': [
                'avg_reward', 'std_reward',
                'avg_cost', 'std_cost',
                'avg_efficiency',
                'demand_satisfaction',
                'temperature_stability',
                'cost_reduction_vs_baseline'
            ],

            'compare_with_baseline': True,

            'save_episode_data': True,
            'create_plots': True,
        }

    def get_algorithm_config(self) -> Dict[str, Any]:
        return self.sac_hyperparameters

    def get_complete_config(self) -> Dict[str, Any]:
        return {
            'algorithm': 'SAC',
            'sac_hyperparameters': self.sac_hyperparameters,
            'environment_config': self.environment_config,
            'training_params': self.training_params,
            'evaluation_params': self.evaluation_params,
            'description': 'SAC for cost-optimized Schweinfurt district heating control'
        }

    def print_config_summary(self):
        print("SAC Configuration Summary")
        print("=" * 50)
        print("Focus: Cost Optimization + Stability")

        print("\nKey Enhancements:")
        print(f"  Learning Rate:       {self.sac_hyperparameters['learning_rate']}")
        print(f"  Buffer Size:         {self.sac_hyperparameters['buffer_size']:,}")
        print(f"  Batch Size:          {self.sac_hyperparameters['batch_size']}")
        print(f"  Network:             {self.sac_hyperparameters['policy_kwargs']['net_arch']}")
        print(f"  Gradient Steps:      {self.sac_hyperparameters['gradient_steps']}")

        print(f"\nTraining Configuration:")
        print(f"  Total Timesteps:     {self.training_params['total_timesteps']:,}")
        print(f"  Warmup Timesteps:    {self.training_params['warmup_timesteps']:,}")
        print(f"  Evaluation Freq:     {self.training_params['eval_freq']:,}")
        print(f"  Reward Threshold:    {self.training_params['reward_threshold']}")
        print(f"  Cost Target:         < {self.training_params['cost_threshold']}€/day")

        print(f"\nCost Optimization Focus:")
        print(f"  Cost Weight:         {self.environment_config['cost_weight']:.1f}")
        print(f"  Efficiency Weight:   {self.environment_config['efficiency_weight']:.2f}")
        print(f"  Demand Weight:       {self.environment_config['demand_weight']:.1f}")

        print(f"\nTarget Performance:")
        print(f"  Target Cost:         < 25€/day")
        print(f"  Maintain Demand Sat: > 90%")

    def validate_config(self) -> bool:
        issues = []

        if self.sac_hyperparameters['buffer_size'] < 50000:
            issues.append("Buffer size too small for SAC")

        if self.sac_hyperparameters['batch_size'] > self.sac_hyperparameters['buffer_size'] / 10:
            issues.append("Batch size too large relative to buffer size")

        if self.training_params['total_timesteps'] < 200000:
            issues.append("Total timesteps too low for production results")

        if self.training_params['warmup_timesteps'] >= self.training_params['total_timesteps'] / 5:
            issues.append("Warmup period too long relative to total training")

        weights = [
            self.environment_config['cost_weight'],
            self.environment_config['efficiency_weight'],
            self.environment_config['demand_weight'],
            self.environment_config['stability_weight']
        ]
        if abs(sum(weights) - 1.0) > 0.01:
            issues.append(f"Reward weights don't sum to 1.0 (sum={sum(weights):.3f})")

        parent_dir = Path(__file__).parent.parent
        data_file = parent_dir / self.environment_config['data_file']
        svr_file = parent_dir / self.environment_config['svr_model_path']

        if not data_file.exists():
            issues.append(f"Data file not found: {data_file}")

        if not svr_file.exists():
            issues.append(f"SVR model file not found: {svr_file}")

        if issues:
            print("Configuration validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("Configuration validation passed!")
        return True

enhanced_sac_config = EnhancedSACConfig()

def get_enhanced_sac_hyperparameters():
    return enhanced_sac_config.get_algorithm_config()

def get_enhanced_training_config():
    return enhanced_sac_config.training_params

def get_enhanced_environment_config():
    return enhanced_sac_config.environment_config

def get_enhanced_evaluation_config():
    return enhanced_sac_config.evaluation_params

if __name__ == "__main__":
    print("SAC Configuration for District Heating")
    print("=" * 60)

    config = EnhancedSACConfig()
    config.print_config_summary()

    print(f"\nValidating configuration...")
    is_valid = config.validate_config()

    if is_valid:
        print(f"\n✓ SAC configuration ready!")
        print(f"\nKey Improvements:")
        print(f"1. 3x longer training (300k vs 100k timesteps)")
        print(f"2. Cost weight increased to 0.6")
        print(f"3. Larger network and buffer for stability")
        print(f"4. Target: 60% cost reduction")

        print(f"\nNext steps:")
        print(f"1. Create environment: enhanced_sac_env.py")
        print(f"2. Test setup: enhanced_test_sac.py")
        print(f"3. Start training: enhanced_sac_training.py")
    else:
        print(f"\n✗ Please fix configuration issues before proceeding.")