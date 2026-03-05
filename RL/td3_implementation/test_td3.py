import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import warnings
import tempfile
import os

from td3_env import create_td3_environment, TD3DistrictHeatingEnv
from td3_config import get_td3_hyperparameters, get_environment_config, get_training_config
from common.evaluation_metrics import DistrictHeatingEvaluator
from common.utils import ValidationUtils

class TD3Tester:

    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp()
        print(f"TD3 Test Suite initialized. Temp dir: {self.temp_dir}")

    def run_all_tests(self) -> bool:
        print("TD3 Implementation Test Suite")
        print("=" * 60)

        tests = [
            ("Configuration Validation", self.test_configuration),
            ("Environment Setup", self.test_environment_setup),
            ("Environment Compatibility", self.test_environment_compatibility),
            ("TD3 Model Creation", self.test_td3_model_creation),
            ("Training Pipeline", self.test_training_pipeline),
            ("Evaluation System", self.test_evaluation_system),
            ("Integration Test", self.test_integration)
        ]

        all_passed = True
        for test_name, test_func in tests:
            print(f"\n{test_name}")
            print("-" * 40)
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

    def test_configuration(self) -> bool:
        try:
            td3_params = get_td3_hyperparameters()
            env_config = get_environment_config()
            train_config = get_training_config()

            print(f"✓ Configuration files loaded successfully")

            required_td3_params = ['learning_rate', 'buffer_size', 'batch_size', 'gamma', 'tau', 'policy_delay']
            missing_params = [p for p in required_td3_params if p not in td3_params]
            if missing_params:
                print(f"✗ Missing TD3 parameters: {missing_params}")
                return False

            print(f"✓ All required TD3 parameters present")

            if not (1e-5 <= td3_params['learning_rate'] <= 1e-2):
                print(f"✗ Learning rate out of reasonable range: {td3_params['learning_rate']}")
                return False

            if td3_params['buffer_size'] < 1000:
                print(f"✗ Buffer size too small: {td3_params['buffer_size']}")
                return False

            if td3_params['policy_delay'] < 1:
                print(f"✗ Policy delay must be >= 1: {td3_params['policy_delay']}")
                return False

            print(f"✓ Parameter ranges validated")

            data_file = Path(env_config['data_file'])
            svr_file = Path(env_config['svr_model_path'])

            if not data_file.exists():
                print(f"✗ Data file not found: {data_file}")
                return False

            if not svr_file.exists():
                print(f"✗ SVR model not found: {svr_file}")
                return False

            print(f"✓ Required files exist")

            return True

        except Exception as e:
            print(f"✗ Configuration test failed: {e}")
            return False

    def test_environment_setup(self) -> bool:
        try:
            print("Testing training environment...")
            train_env = create_td3_environment(training=True)

            assert train_env.observation_space.shape == (31,)
            assert train_env.action_space.shape == (11,)
            assert train_env.td3_optimizations == True

            print(f"✓ Training environment created")

            print("Testing evaluation environment...")
            eval_env = create_td3_environment(training=False)

            assert eval_env.observation_space.shape == (31,)
            assert eval_env.action_space.shape == (11,)

            print(f"✓ Evaluation environment created")

            obs, info = train_env.reset()
            assert obs.shape == (31,)
            assert 'td3_optimizations' in info

            action = train_env.action_space.sample()
            obs, reward, terminated, truncated, info = train_env.step(action)

            assert obs.shape == (31,)
            assert isinstance(reward, (int, float))
            assert 'reward_components' in info
            assert 'enhanced_reward' in info
            assert 'action_consistency' in info

            print(f"✓ Environment reset and step working")

            noisy_action = train_env.add_exploration_noise(action)
            assert noisy_action.shape == action.shape
            assert np.all(noisy_action >= 0) and np.all(noisy_action <= 1)

            print(f"✓ TD3 exploration noise working")

            return True

        except Exception as e:
            print(f"✗ Environment setup test failed: {e}")
            return False

    def test_environment_compatibility(self) -> bool:
        try:
            print("Testing gymnasium compatibility...")

            env = create_td3_environment(training=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                check_env(env, warn=True)

            print(f"✓ Environment passes stable-baselines3 checks")

            monitored_env = Monitor(env, filename=None)
            obs, info = monitored_env.reset()
            action = monitored_env.action_space.sample()
            obs, reward, terminated, truncated, info = monitored_env.step(action)

            print(f"✓ Monitor wrapper compatibility confirmed")

            return True

        except Exception as e:
            print(f"✗ Environment compatibility test failed: {e}")
            return False

    def test_td3_model_creation(self) -> bool:
        try:
            print("Testing TD3 model creation...")

            env = create_td3_environment(training=True)
            td3_params = get_td3_hyperparameters()

            model = TD3(
                'MlpPolicy',
                env,
                **td3_params,
                verbose=0
            )

            print(f"✓ TD3 model created successfully")
            print(f"  - Policy: {model.policy}")
            print(f"  - Learning rate: {model.learning_rate}")
            print(f"  - Buffer size: {model.buffer_size}")
            print(f"  - Batch size: {model.batch_size}")
            print(f"  - Policy delay: {model.policy_delay}")

            obs, _ = env.reset()
            action, _states = model.predict(obs, deterministic=False)
            assert action.shape == (11,)
            assert np.all(action >= 0) and np.all(action <= 1)

            print(f"✓ Model prediction working")

            det_action, _states = model.predict(obs, deterministic=True)
            assert det_action.shape == (11,)

            print(f"✓ Deterministic prediction working")

            return True

        except Exception as e:
            print(f"✗ TD3 model creation test failed: {e}")
            return False

    def test_training_pipeline(self) -> bool:
        try:
            print("Testing training pipeline...")

            train_env = Monitor(create_td3_environment(training=True), filename=None)
            eval_env = Monitor(create_td3_environment(training=False), filename=None)

            td3_params = get_td3_hyperparameters()
            model = TD3('MlpPolicy', train_env, **td3_params, verbose=0)

            print(f"✓ Training components created")

            callback = EvalCallback(
                eval_env,
                best_model_save_path=self.temp_dir,
                log_path=self.temp_dir,
                eval_freq=1000,
                n_eval_episodes=3,
                deterministic=True,
                verbose=0
            )

            print(f"✓ Evaluation callback created")

            print("Testing short training run (100 steps)...")
            model.learn(total_timesteps=100, callback=callback, progress_bar=False)

            print(f"✓ Short training run completed")

            model_path = os.path.join(self.temp_dir, "test_td3_model")
            model.save(model_path)

            loaded_model = TD3.load(model_path)

            obs, _ = eval_env.reset()
            action1, _ = model.predict(obs, deterministic=True)
            action2, _ = loaded_model.predict(obs, deterministic=True)

            assert np.allclose(action1, action2, atol=1e-5)

            print(f"✓ Model save/load working")

            return True

        except Exception as e:
            print(f"✗ Training pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_evaluation_system(self) -> bool:
        try:
            print("Testing evaluation system...")

            evaluator = DistrictHeatingEvaluator(results_dir=self.temp_dir)

            env = create_td3_environment(training=True)
            td3_params = get_td3_hyperparameters()
            model = TD3('MlpPolicy', env, **td3_params, verbose=0)

            model.learn(total_timesteps=50, progress_bar=False)

            model_path = os.path.join(self.temp_dir, "eval_test_model")
            model.save(model_path)

            print(f"✓ Test model created and saved")

            env_config = get_environment_config()
            env_kwargs = {
                'data_file': env_config['data_file'],
                'svr_model_path': env_config['svr_model_path'],
                'start_date': env_config['eval_start_date']
            }

            results = evaluator.evaluate_model(
                model_path + ".zip",
                TD3DistrictHeatingEnv,
                env_kwargs,
                episodes=2,
                algorithm='TD3'
            )

            assert 'algorithm' in results
            assert 'summary' in results
            assert 'avg_reward' in results['summary']
            assert 'avg_cost' in results['summary']

            print(f"✓ Model evaluation working")

            comparison = evaluator.compare_with_baseline(
                model_path + ".zip",
                TD3DistrictHeatingEnv,
                env_kwargs,
                episodes=2,
                algorithm='TD3'
            )

            assert 'rl_results' in comparison
            assert 'baseline_results' in comparison
            assert 'improvements' in comparison

            print(f"✓ Baseline comparison working")

            return True

        except Exception as e:
            print(f"✗ Evaluation system test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_integration(self) -> bool:
        try:
            print("Testing full integration...")

            train_env = Monitor(create_td3_environment(training=True), filename=None)
            eval_env = Monitor(create_td3_environment(training=False), filename=None)

            obs, info = train_env.reset()
            assert 'td3_optimizations' in info
            assert 'exploration_noise_std' in info

            action = train_env.action_space.sample()
            obs, reward, terminated, truncated, info = train_env.step(action)
            assert 'reward_components' in info
            assert 'action_consistency' in info['reward_components']
            assert 'consistency_reward' in info['reward_components']

            print(f"✓ TD3 environment features working")

            train_env.env.set_evaluation_mode(True)
            assert train_env.env.exploration_noise_std == 0.0

            train_env.env.set_evaluation_mode(False)
            assert train_env.env.exploration_noise_std == 0.1

            print(f"✓ Evaluation mode switching working")

            for _ in range(5):
                action = train_env.action_space.sample()
                train_env.step(action)

            summary = train_env.env.get_episode_summary()
            required_keys = ['avg_efficiency', 'production_demand_ratio', 'avg_action_consistency']
            for key in required_keys:
                assert key in summary

            print(f"✓ Episode summary and action consistency tracking working")

            base_action = np.array([0.5] * 11)
            noisy_action = train_env.env.add_exploration_noise(base_action)
            noise_magnitude = np.mean(np.abs(noisy_action - base_action))
            assert 0 <= noise_magnitude <= 0.5

            print(f"✓ Exploration noise system working (noise magnitude: {noise_magnitude:.3f})")

            return True

        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_test_summary(self):
        print(f"\nTD3 TEST SUITE SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        print(f"Tests passed: {passed}/{total}")

        if passed == total:
            print(f"\nTD3 implementation is ready for training!")
            print(f"\nRecommended next steps:")
            print(f"1. Start TD3 training: python td3_implementation/td3_training.py")
            print(f"2. Monitor training progress with TensorBoard")
            print(f"3. Compare results with SAC implementation")
        else:
            print(f"\nSome tests failed. Please review and fix issues.")
            print(f"\nFailed tests:")
            for test_name, result in self.test_results.items():
                if not result:
                    print(f"  - {test_name}")

        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    print("TD3 Implementation Test Suite")
    print("Starting comprehensive testing...")
    print(f"Timestamp: {pd.Timestamp.now()}")

    tester = TD3Tester()
    success = tester.run_all_tests()

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)