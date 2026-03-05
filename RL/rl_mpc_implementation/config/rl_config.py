import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

class RLConfig:

    def __init__(self):
        self.sac_params = self._define_sac_parameters()
        self.training_config = self._define_training_config()
        self.env_config = self._define_environment_config()
        self.eval_config = self._define_evaluation_config()
        self.monitoring_config = self._define_monitoring_config()
        self.deployment_config = self._define_deployment_config()

        print("RL Configuration initialized for MPC parameter tuning")
        print(f"Total training timesteps: {self.training_config['total_timesteps']}")
        print(f"Target performance: {self.training_config['target_daily_cost']}€/day")

    def _define_sac_parameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 3e-4,
            'policy_lr': 3e-4,
            'qf_lr': 3e-4,
            'alpha_lr': 3e-4,
            'net_arch': [256, 256],
            'activation_fn': 'relu',
            'use_sde': False,
            'buffer_size': 100000,
            'batch_size': 256,
            'learning_starts': 2000,
            'tau': 0.005,
            'target_update_interval': 1,
            'ent_coef': 'auto',
            'target_entropy': 'auto',
            'use_sde_at_warmup': False,
            'train_freq': (1, 'step'),
            'gradient_steps': 1,
            'weight_decay': 0.0,
            'clip_grad_norm': None,
            'device': 'auto',
            'seed': 42,
            'verbose': 1
        }

    def _define_training_config(self) -> Dict[str, Any]:
        return {
            'total_timesteps': 50000,
            'max_episodes': 1000,
            'early_stopping_patience': 100,
            'target_daily_cost': 25.0,
            'acceptable_daily_cost': 30.0,
            'min_demand_satisfaction': 0.85,
            'min_efficiency': 0.75,
            'max_constraint_violations': 0,
            'success_threshold_cost': 28.0,
            'success_threshold_demand': 0.87,
            'success_episodes': 10,
            'warmup_episodes': 50,
            'exploration_schedule': 'linear',
            'final_exploration_rate': 0.05,
            'use_curriculum': True,
            'curriculum_stages': [
                {'episodes': 100, 'cost_tolerance': 40.0, 'demand_min': 0.80},
                {'episodes': 200, 'cost_tolerance': 35.0, 'demand_min': 0.83},
                {'episodes': 300, 'cost_tolerance': 30.0, 'demand_min': 0.85},
                {'episodes': None, 'cost_tolerance': 25.0, 'demand_min': 0.85}
            ]
        }

    def _define_environment_config(self) -> Dict[str, Any]:
        return {
            'episode_length': 24,
            'max_episode_steps': 24,
            'random_start': True,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'weather_noise': 0.05,
            'demand_noise': 0.02,
            'equipment_failures': False,
            'reward_scaling': 1.0,
            'cost_weight_in_reward': 0.6,
            'demand_weight_in_reward': 0.3,
            'efficiency_weight_in_reward': 0.1,
            'state_normalization': True,
            'include_history': True,
            'history_length': 3,
            'action_noise': 0.1,
            'action_smoothing': 0.1,
            'parameter_bounds_strict': True
        }

    def _define_evaluation_config(self) -> Dict[str, Any]:
        return {
            'eval_freq': 1000,
            'eval_episodes': 5,
            'eval_deterministic': True,
            'track_best_model': True,
            'best_model_metric': 'cost',
            'save_best_threshold': 35.0,
            'eval_on_train': False,
            'eval_on_val': True,
            'eval_on_test': False,
            'confidence_interval': 0.95,
            'statistical_tests': True,
            'baseline_comparison': True,
            'baselines': {
                'original_sac': {'cost': 53.35, 'demand_satisfaction': 0.963},
                'enhanced_sac': {'cost': 47.97, 'demand_satisfaction': 0.895},
                'hybrid_sac_td3': {'cost': 37.44, 'demand_satisfaction': 0.834},
                'td3_benchmark': {'cost': 9.37, 'demand_satisfaction': 0.244}
            }
        }

    def _define_monitoring_config(self) -> Dict[str, Any]:
        return {
            'log_level': 'INFO',
            'log_to_file': True,
            'log_dir': 'logs',
            'tensorboard_log': True,
            'track_metrics': [
                'daily_cost', 'demand_satisfaction', 'efficiency',
                'constraint_violations', 'mpc_solve_time', 'reward',
                'cost_weight', 'comfort_weight', 'efficiency_weight', 'stability_weight'
            ],
            'early_stopping': True,
            'patience': 20,
            'min_delta': 0.5,
            'progress_bar': True,
            'print_freq': 5,
            'save_freq': 2000,
            'cost_alert_threshold': 50.0,
            'demand_alert_threshold': 0.80,
            'violation_alert_threshold': 1,
            'plot_training_curves': True,
            'plot_parameter_evolution': True,
            'plot_cost_vs_demand': True,
            'save_plots': True
        }

    def _define_deployment_config(self) -> Dict[str, Any]:
        return {
            'export_onnx': False,
            'export_tensorrt': False,
            'quantization': False,
            'safety_checks': True,
            'fallback_controller': True,
            'max_parameter_change': 0.2,
            'parameter_change_smoothing': 0.1,
            'performance_monitoring': True,
            'alert_system': True,
            'data_logging': True,
            'model_updating': False,
            'hard_constraint_enforcement': True,
            'soft_constraint_penalties': True,
            'emergency_shutdown': True,
            'regulatory_compliance': True,
            'audit_trail': True,
            'backup_systems': True,
            'redundancy': True
        }

    def get_hyperparameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'learning_rate': (1e-5, 1e-2),
            'buffer_size': (10000, 200000),
            'batch_size': (64, 512),
            'tau': (0.001, 0.01),
            'target_daily_cost': (20.0, 35.0),
            'success_threshold_cost': (25.0, 32.0),
            'eval_freq': (500, 2000),
            'episode_length': (12, 48)
        }

    def create_curriculum_config(self, stage: int) -> Dict[str, Any]:
        if not self.training_config['use_curriculum']:
            return {}

        stages = self.training_config['curriculum_stages']
        if stage >= len(stages):
            stage = len(stages) - 1

        current_stage = stages[stage]

        return {
            'cost_tolerance': current_stage['cost_tolerance'],
            'demand_minimum': current_stage['demand_min'],
            'episodes_in_stage': current_stage['episodes'],
            'stage_number': stage,
            'total_stages': len(stages)
        }

    def get_training_schedule(self) -> Dict[str, Any]:
        return {
            'phases': [
                {
                    'name': 'warmup',
                    'timesteps': 2000,
                    'learning_rate': self.sac_params['learning_rate'] * 0.5,
                    'exploration_noise': 0.3,
                    'focus': 'exploration'
                },
                {
                    'name': 'learning',
                    'timesteps': 30000,
                    'learning_rate': self.sac_params['learning_rate'],
                    'exploration_noise': 0.1,
                    'focus': 'optimization'
                },
                {
                    'name': 'fine_tuning',
                    'timesteps': 18000,
                    'learning_rate': self.sac_params['learning_rate'] * 0.3,
                    'exploration_noise': 0.05,
                    'focus': 'stability'
                }
            ]
        }

    def validate_configuration(self) -> bool:
        try:
            if self.training_config['target_daily_cost'] <= 0:
                print("Error: Target daily cost must be positive")
                return False

            if not (0 < self.training_config['min_demand_satisfaction'] <= 1):
                print("Error: Demand satisfaction must be between 0 and 1")
                return False

            if self.sac_params['learning_rate'] <= 0:
                print("Error: Learning rate must be positive")
                return False

            if self.sac_params['buffer_size'] < self.sac_params['batch_size']:
                print("Error: Buffer size must be larger than batch size")
                return False

            if self.env_config['episode_length'] <= 0:
                print("Error: Episode length must be positive")
                return False

            total_split = (self.env_config['train_split'] +
                           self.env_config['val_split'] +
                           self.env_config['test_split'])
            if abs(total_split - 1.0) > 0.01:
                print(f"Warning: Data splits sum to {total_split:.3f}, not 1.0")

            if self.eval_config['eval_freq'] <= 0:
                print("Error: Evaluation frequency must be positive")
                return False

            print("RL configuration validation passed")
            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def get_comparison_with_baselines(self) -> Dict[str, Dict[str, float]]:
        baselines = self.eval_config['baselines'].copy()

        baselines['target'] = {
            'cost': self.training_config['target_daily_cost'],
            'demand_satisfaction': self.training_config['min_demand_satisfaction']
        }

        baselines['professor_requirement'] = {
            'cost': 25.0,
            'demand_satisfaction': 0.85,
            'combined_success': True
        }

        return baselines

def get_rl_config_development() -> RLConfig:
    config = RLConfig()

    config.training_config['total_timesteps'] = 5000
    config.sac_params['buffer_size'] = 10000
    config.eval_config['eval_freq'] = 500
    config.env_config['episode_length'] = 12

    return config

def get_rl_config_production() -> RLConfig:
    config = RLConfig()

    config.training_config['total_timesteps'] = 100000
    config.sac_params['buffer_size'] = 200000
    config.eval_config['eval_freq'] = 2000
    config.env_config['episode_length'] = 24

    return config

def get_rl_config_ablation() -> RLConfig:
    config = RLConfig()

    config.training_config['total_timesteps'] = 30000
    config.eval_config['eval_episodes'] = 10
    config.monitoring_config['save_freq'] = 1000

    return config

def get_rl_config() -> RLConfig:
    return RLConfig()

if __name__ == "__main__":
    print("RL Configuration for Hybrid MPC Parameter Tuning")
    print("=" * 60)

    config = get_rl_config()

    print(f"\nConfiguration Summary:")
    print(f"  - Total timesteps: {config.training_config['total_timesteps']}")
    print(f"  - Target cost: {config.training_config['target_daily_cost']}€/day")
    print(f"  - Min demand satisfaction: {config.training_config['min_demand_satisfaction']*100}%")
    print(f"  - Episode length: {config.env_config['episode_length']}h")
    print(f"  - Evaluation frequency: {config.eval_config['eval_freq']} steps")

    print(f"\nSAC Parameters:")
    print(f"  - Learning rate: {config.sac_params['learning_rate']}")
    print(f"  - Buffer size: {config.sac_params['buffer_size']}")
    print(f"  - Batch size: {config.sac_params['batch_size']}")
    print(f"  - Network architecture: {config.sac_params['net_arch']}")

    print(f"\nBaseline Comparisons:")
    baselines = config.get_comparison_with_baselines()
    for name, metrics in baselines.items():
        if 'cost' in metrics and 'demand_satisfaction' in metrics:
            print(f"  {name}: {metrics['cost']:.1f}€, {metrics['demand_satisfaction']:.1%}")

    print(f"\nTraining Schedule:")
    schedule = config.get_training_schedule()
    for phase in schedule['phases']:
        print(f"  {phase['name']}: {phase['timesteps']} steps, LR={phase['learning_rate']:.1e}")

    print(f"\nValidation result: {config.validate_configuration()}")
    print(f"\nRL configuration ready for hybrid MPC training!")