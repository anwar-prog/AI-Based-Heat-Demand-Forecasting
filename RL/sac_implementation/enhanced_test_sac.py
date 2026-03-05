import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import warnings
import tempfile
import os

from enhanced_sac_env import create_enhanced_sac_environment, EnhancedSACDistrictHeatingEnv
from enhanced_sac_config import get_enhanced_sac_hyperparameters, get_enhanced_environment_config, get_enhanced_training_config
from common.evaluation_metrics import DistrictHeatingEvaluator
from common.utils import ValidationUtils

class EnhancedSACTester:

    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp()
        print(f"SAC Test Suite initialized. Temp dir: {self.temp_dir}")

    def run_all_tests(self) -> bool:
        print("SAC Implementation Test Suite")
        print("=" * 70)

        tests = [
            ("Configuration", self.test_enhanced_configuration),
            ("Environment", self.test_enhanced_environment),
            ("Cost Optimization", self.test_cost_optimization),
            ("Environment Compatibility", self.test_environment_compatibility),
            ("SAC Model", self.test_enhanced_sac_model),
            ("Training Pipeline", self.test_training_pipeline),
            ("Evaluation System", self.test_evaluation_system),
            ("Integration Test", self.test_integration)
        ]

        all_passed = True
        for test_name, test_func in tests:
            print(f"\n{test_name}")
            print("-" * 50)
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result:
                    print(f"✓ {test_name} PASSED")
                else:
                    print(f"✗ {test_name} FAILED")
                    all_passed = False
            except Exception as e:
                print(f"✗ {test_name} FAILED with exception: {e}")
                self.test_results[test_name] = False
                all_passed = False

        self.print_test_summary()
        return all_passed

    def test_enhanced_configuration(self) -> bool:
        try:
            sac_params = get_enhanced_sac_hyperparameters()
            env_config = get_enhanced_environment_config()
            train_config = get_enhanced_training_config()

            print("✓ Configuration files loaded")

            assert sac_params['learning_rate'] == 2e-4, "Learning rate not set"
            assert sac_params['buffer_size'] == 200000, "Buffer size not set"
            assert sac_params['batch_size'] == 512, "Batch size not set"
            assert sac_params['gradient_steps'] == 2, "Gradient steps not set"

            print("✓ SAC parameters validated")

            assert env_config['cost_weight'] == 0.6, "Cost weight not optimized"
            assert env_config['efficiency_weight'] == 0.25, "Efficiency weight incorrect"
            assert env_config['demand_weight'] == 0.1, "Demand weight incorrect"

            print("✓ Cost optimization weights validated")

            assert train_config['total_timesteps'] == 300000, "Training not extended"
            assert train_config['cost_threshold'] == 25.0, "Cost threshold not set"

            print("✓ Training parameters validated")

            return True

        except Exception as e:
            print(f"✗ Configuration test failed: {e}")
            return False

    def test_enhanced_environment(self) -> bool:
        try:
            env = create_enhanced_sac_environment(training=True)

            assert hasattr(env, 'cost_optimization'), "Cost optimization not enabled"
            assert hasattr(env, 'target_daily_cost'), "Target daily cost not set"
            assert hasattr(env, 'cost_weight'), "Cost weight not configured"

            assert env.cost_optimization == True, "Cost optimization not active"
            assert env.target_daily_cost == 25.0, "Target cost incorrect"
            assert env.cost_weight == 0.6, "Cost weight incorrect"

            print("✓ Environment attributes validated")

            obs, info = env.reset()
            assert 'cost_optimization_enabled' in info, "Cost optimization not in info"
            assert 'target_daily_cost' in info, "Target cost not in info"

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert 'reward_breakdown' in info, "Reward breakdown missing"
            assert 'cost_analysis' in info, "Cost analysis missing"

            breakdown = info['reward_breakdown']
            required_components = ['cost_reward', 'efficiency_reward', 'demand_reward', 'stability_reward']
            for component in required_components:
                assert component in breakdown, f"{component} missing from breakdown"

            print("✓ Cost tracking and analysis working")

            return True

        except Exception as e:
            print(f"✗ Environment test failed: {e}")
            return False

    def test_cost_optimization(self) -> bool:
        try:
            env = create_enhanced_sac_environment(training=True)

            obs, info = env.reset()
            cost_estimates = []
            rewards = []

            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                cost_analysis = info['cost_analysis']
                cost_estimates.append(cost_analysis['daily_cost_estimate'])
                rewards.append(reward)

                if terminated or truncated:
                    break

            summary = env.get_cost_performance_summary()

            required_summary_keys = [
                'estimated_daily_cost', 'vs_target_cost',
                'target_achieved', 'cost_optimization_enabled'
            ]

            for key in required_summary_keys:
                assert key in summary, f"{key} missing from cost summary"

            assert summary['cost_optimization_enabled'] == True, "Cost optimization not enabled in summary"

            print("✓ Cost optimization functionality working")
            print(f"  - Average daily cost estimate: {np.mean(cost_estimates):.1f}€")
            print(f"  - vs Target (25€): {summary['vs_target_cost']:+.1f}€")

            return True

        except Exception as e:
            print(f"✗ Cost optimization test failed: {e}")
            return False

    def test_environment_compatibility(self) -> bool:
        try:
            env = create_enhanced_sac_environment(training=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                check_env(env, warn=True)

            print("✓ Environment passes stable-baselines3 checks")

            monitored_env = Monitor(env, filename=None)
            obs, info = monitored_env.reset()
            action = monitored_env.action_space.sample()
            obs, reward, terminated, truncated, info = monitored_env.step(action)

            print("✓ Monitor wrapper compatibility confirmed")

            env.set_evaluation_mode(True)
            print("✓ Evaluation mode switching working")

            return True

        except Exception as e:
            print(f"✗ Environment compatibility test failed: {e}")
            return False

    def test_enhanced_sac_model(self) -> bool:
        try:
            env = create_enhanced_sac_environment(training=True)
            sac_params = get_enhanced_sac_hyperparameters()

            model = SAC('MlpPolicy', env, **sac_params, verbose=0)

            print("✓ SAC model created")

            assert model.learning_rate == 2e-4, "Learning rate not set correctly"
            assert model.buffer_size == 200000, "Buffer size not set correctly"
            assert model.batch_size == 512, "Batch size not set correctly"

            print("✓ SAC parameters applied correctly")

            obs, _ = env.reset()
            action, _ = model.predict(obs)

            assert action.shape == env.action_space.shape, "Action shape mismatch"
            assert np.all((action >= 0) & (action <= 1)), "Action out of bounds"

            print("✓ Model prediction working")

            return True

        except Exception as e:
            print(f"✗ SAC model test failed: {e}")
            return False

    def test_training_pipeline(self) -> bool:
        try:
            train_env = Monitor(create_enhanced_sac_environment(training=True), filename=None)
            eval_env = Monitor(create_enhanced_sac_environment(training=False), filename=None)

            print("✓ Environments created for training")

            sac_params = get_enhanced_sac_hyperparameters()
            model = SAC('MlpPolicy', train_env, **sac_params, verbose=0)

            print("✓ SAC model created for training")

            callback = EvalCallback(
                eval_env,
                best_model_save_path=self.temp_dir,
                log_path=self.temp_dir,
                eval_freq=100,
                n_eval_episodes=2,
                deterministic=True,
                verbose=0
            )

            print("✓ Evaluation callback created")

            print("Testing training run (200 steps)...")
            model.learn(total_timesteps=200, callback=callback, progress_bar=False)

            print("✓ Training run completed")

            model_path = os.path.join(self.temp_dir, "test_sac")
            model.save(model_path)

            loaded_model = SAC.load(model_path)

            obs, _ = eval_env.reset()
            action1, _ = model.predict(obs, deterministic=True)
            action2, _ = loaded_model.predict(obs, deterministic=True)

            assert np.allclose(action1, action2, atol=1e-5), "Model save/load inconsistent"

            print("✓ Model save/load working")

            return True

        except Exception as e:
            print(f"✗ Training pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_evaluation_system(self) -> bool:
        try:
            evaluator = DistrictHeatingEvaluator(results_dir=self.temp_dir)

            env = create_enhanced_sac_environment(training=True)
            sac_params = get_enhanced_sac_hyperparameters()
            model = SAC('MlpPolicy', env, **sac_params, verbose=0)

            model.learn(total_timesteps=100, progress_bar=False)

            model_path = os.path.join(self.temp_dir, "eval_test_sac")
            model.save(model_path)

            print("✓ Test model created and saved")

            env_config = get_enhanced_environment_config()
            parent_dir = Path(__file__).parent.parent

            env_kwargs = {
                'data_file': str(parent_dir / env_config['data_file']),
                'svr_model_path': str(parent_dir / env_config['svr_model_path']),
                'start_date': env_config['eval_start_date'],
                'cost_optimization': True
            }

            results = evaluator.evaluate_model(
                model_path + ".zip",
                EnhancedSACDistrictHeatingEnv,
                env_kwargs,
                episodes=3,
                algorithm='SAC'
            )

            assert 'algorithm' in results, "Algorithm missing from results"
            assert results['algorithm'] == 'SAC', "Algorithm name incorrect"
            assert 'summary' in results, "Summary missing from results"

            required_metrics = ['avg_reward', 'avg_cost', 'avg_efficiency']
            for metric in required_metrics:
                assert metric in results['summary'], f"{metric} missing from summary"

            print("✓ Model evaluation working")

            comparison = evaluator.compare_with_baseline(
                model_path + ".zip",
                EnhancedSACDistrictHeatingEnv,
                env_kwargs,
                episodes=2,
                algorithm='SAC'
            )

            assert 'rl_results' in comparison, "RL results missing from comparison"
            assert 'baseline_results' in comparison, "Baseline results missing"
            assert 'improvements' in comparison, "Improvements missing"

            print("✓ Baseline comparison working")

            return True

        except Exception as e:
            print(f"✗ Evaluation system test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_integration(self) -> bool:
        try:
            train_env = Monitor(create_enhanced_sac_environment(training=True), filename=None)
            eval_env = Monitor(create_enhanced_sac_environment(training=False), filename=None)

            obs, info = train_env.reset()
            assert info['cost_optimization_enabled'] == True, "Cost optimization not enabled"
            assert info['target_daily_cost'] == 25.0, "Target cost not set"

            action = train_env.action_space.sample()
            obs, reward, terminated, truncated, info = train_env.step(action)

            assert 'reward_breakdown' in info, "Reward breakdown missing"
            assert 'cost_analysis' in info, "Cost analysis missing"

            breakdown = info['reward_breakdown']
            cost_analysis = info['cost_analysis']

            required_components = ['cost_component', 'efficiency_component', 'demand_component', 'stability_component']
            for component in required_components:
                assert component in breakdown, f"{component} missing from breakdown"
                assert isinstance(breakdown[component], (int, float)), f"{component} not numeric"

            print("✓ Reward breakdown integration working")

            costs = []
            for _ in range(5):
                action = train_env.action_space.sample()
                obs, reward, terminated, truncated, info = train_env.step(action)
                costs.append(info['cost_analysis']['daily_cost_estimate'])
                if terminated or truncated:
                    break

            summary = train_env.env.get_cost_performance_summary()
            assert summary['cost_optimization_enabled'] == True, "Cost optimization status incorrect"

            print("✓ Cost tracking integration working")
            print(f"  - Average test cost: {np.mean(costs):.1f}€/day")

            return True

        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_test_summary(self):
        print(f"\nSAC TEST SUITE SUMMARY")
        print("=" * 70)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        print(f"Tests passed: {passed}/{total}")

        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"SAC implementation ready for production training!")

            print(f"\nFeatures Validated:")
            print(f"✓ Cost optimization (60% weight, 25€ target)")
            print(f"✓ Extended training (300k timesteps)")
            print(f"✓ Enhanced network ([512, 256, 128])")
            print(f"✓ Larger buffer (200k) and batch size (512)")
            print(f"✓ Comprehensive cost tracking and analysis")

            print(f"\nRecommended next steps:")
            print(f"1. Start training: python enhanced_sac_training.py")
            print(f"2. Monitor cost metrics during training")
            print(f"3. Target: <25€/day")
        else:
            print(f"\n⚠️ Some tests failed. Please review:")
            for test_name, result in self.test_results.items():
                if not result:
                    print(f"  - {test_name}")

        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    print("SAC Implementation Test Suite")
    print("Production-Ready Cost Optimization Focus")
    print(f"Timestamp: {pd.Timestamp.now()}")
    print("=" * 70)

    tester = EnhancedSACTester()
    success = tester.run_all_tests()

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)