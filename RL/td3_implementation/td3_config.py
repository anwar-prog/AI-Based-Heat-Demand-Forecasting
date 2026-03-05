import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from common.utils import ConfigManager
from typing import Dict, Any

class TD3Config:

    def __init__(self):
        self.base_config = ConfigManager.get_td3_config()
        self.training_config = ConfigManager.get_training_config()
        self.td3_hyperparameters = self._get_optimized_td3_params()
        self.environment_config = self._get_environment_config()
        self.training_params = self._get_training_params()
        self.evaluation_params = self._get_evaluation_params()

    def _get_optimized_td3_params(self) -> Dict[str, Any]:
        config = self.base_config.copy()

        config.update({
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'policy_kwargs': {
                'net_arch': [256, 256],
            },
            'gradient_steps': 1,
            'train_freq': 1,
            'learning_starts': 10000,
            'device': 'auto',
        })

        return config

    def _get_environment_config(self) -> Dict[str, Any]:
        return {
            'episode_length': 24,
            'observation_dim': 31,
            'action_dim': 11,
            'action_bounds': (0.0, 1.0),
            'train_start_date': '2021-01-01',
            'train_end_date': '2023-12-31',
            'eval_start_date': '2024-01-01',
            'test_start_date': '2024-06-01',
            'data_file': 'data/schweinfurt_data.csv',
            'svr_model_path': 'data/svr_model_33features.pkl',
        }

    def _get_training_params(self) -> Dict[str, Any]:
        return {
            'total_timesteps': 100000,
            'warmup_timesteps': 10000,
            'eval_freq': 5000,
            'n_eval_episodes': 5,
            'eval_deterministic': True,
            'reward_threshold': 50.0,
            'patience': 5,
            'log_interval': 1000,
            'save_freq': 10000,
            'tensorboard_log': True,
            'progress_bar': True,
        }

    def _get_evaluation_params(self) -> Dict[str, Any]:
        return {
            'n_eval_episodes': 10,
            'eval_deterministic': True,
            'render_episodes': False,
            'track_metrics': [
                'avg_reward',
                'avg_cost',
                'avg_efficiency',
                'demand_satisfaction',
                'temperature_stability',
                'cost_reduction_vs_baseline'
            ],
            'compare_with_baseline': True,
            'compare_with_sac': True,
            'save_episode_data': True,
            'create_plots': True,
        }

    def get_algorithm_config(self) -> Dict[str, Any]:
        return self.td3_hyperparameters

    def get_complete_config(self) -> Dict[str, Any]:
        return {
            'algorithm': 'TD3',
            'td3_hyperparameters': self.td3_hyperparameters,
            'environment_config': self.environment_config,
            'training_params': self.training_params,
            'evaluation_params': self.evaluation_params,
            'description': 'TD3 configuration optimized for Schweinfurt district heating network'
        }

    def print_config_summary(self):
        print("TD3 Configuration Summary")
        print("=" * 40)

        print("\nKey TD3 Parameters:")
        print(f"  Learning Rate:     {self.td3_hyperparameters['learning_rate']}")
        print(f"  Buffer Size:       {self.td3_hyperparameters['buffer_size']:,}")
        print(f"  Batch Size:        {self.td3_hyperparameters['batch_size']}")
        print(f"  Gamma:             {self.td3_hyperparameters['gamma']}")
        print(f"  Tau:               {self.td3_hyperparameters['tau']}")
        print(f"  Policy Delay:      {self.td3_hyperparameters['policy_delay']}")
        print(f"  Target Noise:      {self.td3_hyperparameters['target_policy_noise']}")
        print(f"  Action Noise:      {self.td3_hyperparameters['action_noise']}")

        print(f"\nTraining Configuration:")
        print(f"  Total Timesteps:   {self.training_params['total_timesteps']:,}")
        print(f"  Warmup Timesteps:  {self.training_params['warmup_timesteps']:,}")
        print(f"  Evaluation Freq:   {self.training_params['eval_freq']:,}")
        print(f"  Reward Threshold:  {self.training_params['reward_threshold']}")

        print(f"\nEnvironment Setup:")
        print(f"  Episode Length:    {self.environment_config['episode_length']} hours")
        print(f"  Observation Dim:   {self.environment_config['observation_dim']}")
        print(f"  Action Dim:        {self.environment_config['action_dim']}")
        print(f"  Training Period:   {self.environment_config['train_start_date']} to {self.environment_config['train_end_date']}")

    def validate_config(self) -> bool:
        issues = []

        if self.td3_hyperparameters['buffer_size'] < 10000:
            issues.append("Buffer size too small for TD3 (recommend >= 10,000)")

        if self.td3_hyperparameters['batch_size'] > self.td3_hyperparameters['buffer_size'] / 10:
            issues.append("Batch size too large relative to buffer size")

        if self.training_params['total_timesteps'] < self.training_params['warmup_timesteps']:
            issues.append("Total timesteps must be greater than warmup timesteps")

        if self.td3_hyperparameters['policy_delay'] < 1:
            issues.append("Policy delay must be >= 1")

        if not (0 < self.td3_hyperparameters['target_policy_noise'] < 1):
            issues.append("Target policy noise should be between 0 and 1")

        if self.training_params['eval_freq'] <= 0:
            issues.append("Evaluation frequency must be positive")

        data_file = Path(self.environment_config['data_file'])
        svr_file = Path(self.environment_config['svr_model_path'])

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

td3_config = TD3Config()

def get_td3_hyperparameters():
    return td3_config.get_algorithm_config()

def get_training_config():
    return td3_config.training_params

def get_environment_config():
    return td3_config.environment_config

def get_evaluation_config():
    return td3_config.evaluation_params

if __name__ == "__main__":
    print("TD3 Configuration for District Heating")
    print("=" * 50)

    config = TD3Config()
    config.print_config_summary()

    print(f"\nValidating configuration...")
    is_valid = config.validate_config()

    if is_valid:
        print(f"\nConfiguration ready for TD3 training!")
        print(f"\nNext steps:")
        print(f"1. Create TD3 environment: python td3_implementation/td3_env.py")
        print(f"2. Test TD3 setup: python td3_implementation/test_td3.py")
        print(f"3. Start TD3 training: python td3_implementation/td3_training.py")
    else:
        print(f"\nPlease fix configuration issues before proceeding.")