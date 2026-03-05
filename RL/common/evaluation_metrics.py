import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from stable_baselines3 import PPO, SAC, TD3, A2C

class DistrictHeatingEvaluator:

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.plots_dir = self.results_dir.parent / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.metrics = [
            'avg_reward', 'avg_cost', 'avg_efficiency',
            'cost_reduction', 'demand_satisfaction', 'temperature_stability'
        ]

        print(f"District Heating Evaluator initialized. Results: {self.results_dir}")
        print(f"Plots will be saved to: {self.plots_dir}")

    def evaluate_model(self, model_path: str, env_class, env_kwargs: Dict,
                       episodes: int = 10, algorithm: str = None) -> Dict[str, Any]:
        print(f"\nEvaluating model: {Path(model_path).name}")
        print("=" * 40)

        try:
            model = self._load_model(model_path, algorithm)
            print(f"Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        test_env = env_class(**env_kwargs)

        episode_results = []
        episode_rewards = []
        episode_costs = []
        episode_efficiencies = []
        episode_demand_satisfaction = []
        episode_temp_stability = []

        print(f"Running {episodes} evaluation episodes...")

        for episode in range(episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_cost = 0
            episode_efficiency_scores = []
            episode_demand_scores = []
            episode_temp_scores = []

            step_data = []

            for step in range(test_env.episode_length):
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)

                    episode_reward += reward
                    episode_cost += info['cost']
                    episode_efficiency_scores.append(info['efficiency'])

                    demand_satisfaction = min(info['total_production'], info['total_demand']) / max(info['total_demand'], 1.0)
                    episode_demand_scores.append(demand_satisfaction)

                    temp_stability = 1.0 - np.mean(np.abs(info['zone_temps'] - 85.0)) / 85.0
                    episode_temp_scores.append(max(temp_stability, 0.0))

                    step_data.append({
                        'step': step,
                        'reward': reward,
                        'total_production': info['total_production'],
                        'total_demand': info['total_demand'],
                        'cost': info['cost'],
                        'efficiency': info['efficiency']
                    })

                    if terminated or truncated:
                        break

                except Exception as e:
                    print(f"Error in episode {episode+1}, step {step+1}: {e}")
                    break

            avg_efficiency = np.mean(episode_efficiency_scores) if episode_efficiency_scores else 0.0
            avg_demand_satisfaction = np.mean(episode_demand_scores) if episode_demand_scores else 0.0
            avg_temp_stability = np.mean(episode_temp_scores) if episode_temp_scores else 0.0

            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            episode_efficiencies.append(avg_efficiency)
            episode_demand_satisfaction.append(avg_demand_satisfaction)
            episode_temp_stability.append(avg_temp_stability)

            episode_results.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'cost': episode_cost,
                'efficiency': avg_efficiency,
                'demand_satisfaction': avg_demand_satisfaction,
                'temperature_stability': avg_temp_stability,
                'steps': step_data
            })

            print(f"Episode {episode+1:2d}: Reward={episode_reward:6.2f}, "
                  f"Cost={episode_cost:6.2f}€, Efficiency={avg_efficiency:.3f}")

        results = {
            'model_path': model_path,
            'algorithm': type(model).__name__,
            'episodes': episodes,
            'episode_results': episode_results,
            'summary': {
                'avg_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'avg_cost': float(np.mean(episode_costs)),
                'std_cost': float(np.std(episode_costs)),
                'avg_efficiency': float(np.mean(episode_efficiencies)),
                'std_efficiency': float(np.std(episode_efficiencies)),
                'avg_demand_satisfaction': float(np.mean(episode_demand_satisfaction)),
                'avg_temperature_stability': float(np.mean(episode_temp_stability))
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }

        self._print_evaluation_summary(results)

        plots_saved = self._create_individual_model_plots(results)
        results['plots_saved'] = plots_saved

        return results

    def compare_with_baseline(self, model_path: str, env_class, env_kwargs: Dict,
                              episodes: int = 5, algorithm: str = None) -> Dict[str, Any]:
        print(f"\nBASELINE COMPARISON")
        print("=" * 40)

        rl_results = self.evaluate_model(model_path, env_class, env_kwargs, episodes, algorithm)

        print("\nEvaluating Baseline Controller...")
        baseline_results = self._evaluate_baseline(env_class, env_kwargs, episodes)

        improvements = self._calculate_improvements(rl_results, baseline_results)

        self._print_comparison_results(rl_results, baseline_results, improvements)

        comparison_plots = self._create_baseline_comparison_plots(rl_results, baseline_results)

        comparison_results = {
            'rl_results': rl_results,
            'baseline_results': baseline_results,
            'improvements': improvements,
            'comparison_plots': comparison_plots
        }

        return comparison_results

    def compare_algorithms(self, model_paths: Dict[str, str], env_class,
                           env_kwargs: Dict, episodes: int = 5) -> Dict[str, Any]:
        print(f"\nMULTI-ALGORITHM COMPARISON")
        print("=" * 50)

        model_results = []

        for algorithm, model_path in model_paths.items():
            print(f"\nEvaluating {algorithm}...")
            try:
                results = self.evaluate_model(model_path, env_class, env_kwargs, episodes, algorithm)
                model_results.append(results)
            except Exception as e:
                print(f"Failed to evaluate {algorithm}: {e}")

        comparison_df = pd.DataFrame([
            {
                'algorithm': result['algorithm'],
                'avg_reward': result['summary']['avg_reward'],
                'avg_cost': result['summary']['avg_cost'],
                'avg_efficiency': result['summary']['avg_efficiency'],
                'avg_demand_satisfaction': result['summary']['avg_demand_satisfaction'],
                'avg_temperature_stability': result['summary']['avg_temperature_stability']
            }
            for result in model_results
        ])

        self._print_algorithm_comparison_table(comparison_df)

        best_performers = self._identify_best_performers(comparison_df)
        print(f"\nBEST PERFORMERS:")
        for metric, algorithm in best_performers.items():
            print(f"  {metric}: {algorithm}")

        comparison_plots = self._create_comparison_plots(comparison_df, model_results)

        comparison_results = {
            'model_results': model_results,
            'comparison_dataframe': comparison_df.to_dict('records'),
            'best_performers': best_performers,
            'comparison_plots': comparison_plots
        }

        return comparison_results

    def _load_model(self, model_path: str, algorithm: str = None):
        if algorithm:
            algorithm = algorithm.upper()

        if algorithm == 'PPO' or (not algorithm and 'ppo' in model_path.lower()):
            return PPO.load(model_path)
        elif algorithm == 'SAC' or (not algorithm and 'sac' in model_path.lower()):
            return SAC.load(model_path)
        elif algorithm == 'TD3' or (not algorithm and 'td3' in model_path.lower()):
            return TD3.load(model_path)
        elif algorithm == 'A2C' or (not algorithm and 'a2c' in model_path.lower()):
            return A2C.load(model_path)
        else:
            try:
                return PPO.load(model_path)
            except:
                try:
                    return SAC.load(model_path)
                except:
                    try:
                        return TD3.load(model_path)
                    except:
                        return A2C.load(model_path)

    def _evaluate_baseline(self, env_class, env_kwargs: Dict, episodes: int) -> Dict[str, Any]:
        test_env = env_class(**env_kwargs)

        episode_results = []
        episode_rewards = []
        episode_costs = []
        episode_efficiencies = []
        episode_demand_satisfaction = []
        episode_temp_stability = []

        for episode in range(episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_cost = 0
            episode_efficiency_scores = []
            episode_demand_scores = []
            episode_temp_scores = []

            for step in range(test_env.episode_length):
                action = self._proportional_control_action(obs, test_env)
                obs, reward, terminated, truncated, info = test_env.step(action)

                episode_reward += reward
                episode_cost += info['cost']
                episode_efficiency_scores.append(info['efficiency'])

                demand_satisfaction = min(info['total_production'], info['total_demand']) / max(info['total_demand'], 1.0)
                episode_demand_scores.append(demand_satisfaction)

                temp_stability = 1.0 - np.mean(np.abs(info['zone_temps'] - 85.0)) / 85.0
                episode_temp_scores.append(max(temp_stability, 0.0))

                if terminated or truncated:
                    break

            avg_efficiency = np.mean(episode_efficiency_scores) if episode_efficiency_scores else 0.0
            avg_demand_satisfaction = np.mean(episode_demand_scores) if episode_demand_scores else 0.0
            avg_temp_stability = np.mean(episode_temp_scores) if episode_temp_scores else 0.0

            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            episode_efficiencies.append(avg_efficiency)
            episode_demand_satisfaction.append(avg_demand_satisfaction)
            episode_temp_stability.append(avg_temp_stability)

        results = {
            'algorithm': 'Baseline',
            'episodes': episodes,
            'summary': {
                'avg_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'avg_cost': float(np.mean(episode_costs)),
                'std_cost': float(np.std(episode_costs)),
                'avg_efficiency': float(np.mean(episode_efficiencies)),
                'std_efficiency': float(np.std(episode_efficiencies)),
                'avg_demand_satisfaction': float(np.mean(episode_demand_satisfaction)),
                'avg_temperature_stability': float(np.mean(episode_temp_stability))
            }
        }

        return results

    def _proportional_control_action(self, obs: np.ndarray, env) -> np.ndarray:
        zone_temps = obs[:11] * 100.0
        forecasts = obs[16:27] * 100.0

        target_temp = 85.0
        temp_errors = target_temp - zone_temps

        actions = np.zeros(11, dtype=np.float32)

        for i in range(11):
            base_production = forecasts[i] / 100.0

            temp_adjustment = temp_errors[i] * 0.02

            actions[i] = np.clip(base_production + temp_adjustment, 0.0, 1.0)

        return actions

    def _calculate_improvements(self, rl_results: Dict, baseline_results: Dict) -> Dict[str, float]:
        rl_summary = rl_results['summary']
        baseline_summary = baseline_results['summary']

        reward_improvement = ((rl_summary['avg_reward'] - baseline_summary['avg_reward']) /
                              abs(baseline_summary['avg_reward']) * 100)

        cost_reduction = ((baseline_summary['avg_cost'] - rl_summary['avg_cost']) /
                          baseline_summary['avg_cost'] * 100)

        efficiency_improvement = ((rl_summary['avg_efficiency'] - baseline_summary['avg_efficiency']) /
                                  baseline_summary['avg_efficiency'] * 100)

        demand_satisfaction_improvement = ((rl_summary['avg_demand_satisfaction'] -
                                            baseline_summary['avg_demand_satisfaction']) /
                                           baseline_summary['avg_demand_satisfaction'] * 100)

        return {
            'reward_improvement': reward_improvement,
            'cost_reduction': cost_reduction,
            'efficiency_improvement': efficiency_improvement,
            'demand_satisfaction_improvement': demand_satisfaction_improvement
        }

    def _create_individual_model_plots(self, results: Dict) -> List[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm = results['algorithm']
        plots_saved = []

        plt.style.use('seaborn-v0_8')

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{algorithm} Performance Analysis', fontsize=16)

        episodes = [ep['episode'] for ep in results['episode_results']]
        rewards = [ep['reward'] for ep in results['episode_results']]
        costs = [ep['cost'] for ep in results['episode_results']]
        efficiencies = [ep['efficiency'] for ep in results['episode_results']]
        demand_sats = [ep['demand_satisfaction'] for ep in results['episode_results']]

        axes[0, 0].plot(episodes, rewards, marker='o', color='blue', alpha=0.8)
        axes[0, 0].set_title('Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(episodes, costs, marker='s', color='red', alpha=0.8)
        axes[0, 1].set_title('Cost per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cost (€)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(episodes, efficiencies, marker='^', color='green', alpha=0.8)
        axes[1, 0].set_title('Efficiency per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(episodes, demand_sats, marker='d', color='orange', alpha=0.8)
        axes[1, 1].set_title('Demand Satisfaction per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Demand Satisfaction')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.plots_dir / f'{algorithm.lower()}_performance_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots_saved.append(str(plot_file))

        if results['episode_results']:
            first_episode = results['episode_results'][0]
            if 'steps' in first_episode and first_episode['steps']:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f'{algorithm} Episode 1 Detailed Analysis', fontsize=16)

                steps = [s['step'] for s in first_episode['steps']]
                step_rewards = [s['reward'] for s in first_episode['steps']]
                step_production = [s['total_production'] for s in first_episode['steps']]
                step_demand = [s['total_demand'] for s in first_episode['steps']]
                step_costs = [s['cost'] for s in first_episode['steps']]

                axes[0, 0].plot(steps, step_rewards, color='blue', alpha=0.8)
                axes[0, 0].set_title('Reward per Step')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].grid(True, alpha=0.3)

                axes[0, 1].plot(steps, step_production, label='Production', color='green', alpha=0.8)
                axes[0, 1].plot(steps, step_demand, label='Demand', color='red', alpha=0.8)
                axes[0, 1].set_title('Production vs Demand')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Heat (units)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                axes[1, 0].plot(steps, step_costs, color='red', alpha=0.8)
                axes[1, 0].set_title('Cost per Step')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Cost (€)')
                axes[1, 0].grid(True, alpha=0.3)

                efficiency_per_step = [min(p, d) / max(p, 1.0) for p, d in zip(step_production, step_demand)]
                axes[1, 1].plot(steps, efficiency_per_step, color='purple', alpha=0.8)
                axes[1, 1].set_title('Efficiency per Step')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Efficiency')
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plot_file = self.plots_dir / f'{algorithm.lower()}_episode_detail_{timestamp}.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plots_saved.append(str(plot_file))

        print(f"Individual model plots saved: {len(plots_saved)} files")
        return plots_saved

    def _create_baseline_comparison_plots(self, rl_results: Dict, baseline_results: Dict) -> List[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_saved = []

        plt.style.use('seaborn-v0_8')

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('RL Agent vs Baseline Comparison', fontsize=16)

        algorithms = [rl_results['algorithm'], 'Baseline']

        rewards = [rl_results['summary']['avg_reward'], baseline_results['summary']['avg_reward']]
        axes[0, 0].bar(algorithms, rewards, color=['blue', 'gray'], alpha=0.8)
        axes[0, 0].set_title('Average Reward')
        axes[0, 0].set_ylabel('Reward')

        costs = [rl_results['summary']['avg_cost'], baseline_results['summary']['avg_cost']]
        axes[0, 1].bar(algorithms, costs, color=['red', 'gray'], alpha=0.8)
        axes[0, 1].set_title('Average Cost')
        axes[0, 1].set_ylabel('Cost (€)')

        efficiencies = [rl_results['summary']['avg_efficiency'], baseline_results['summary']['avg_efficiency']]
        axes[1, 0].bar(algorithms, efficiencies, color=['green', 'gray'], alpha=0.8)
        axes[1, 0].set_title('Average Efficiency')
        axes[1, 0].set_ylabel('Efficiency')

        demand_sats = [rl_results['summary']['avg_demand_satisfaction'],
                       baseline_results['summary']['avg_demand_satisfaction']]
        axes[1, 1].bar(algorithms, demand_sats, color=['orange', 'gray'], alpha=0.8)
        axes[1, 1].set_title('Demand Satisfaction')
        axes[1, 1].set_ylabel('Satisfaction Ratio')

        plt.tight_layout()
        plot_file = self.plots_dir / f'baseline_comparison_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots_saved.append(str(plot_file))

        return plots_saved

    def _print_evaluation_summary(self, results: Dict[str, Any]):
        summary = results['summary']
        print(f"\nEVALUATION RESULTS ({results['episodes']} episodes)")
        print("-" * 40)
        print(f"Average Reward:           {summary['avg_reward']:8.2f} ± {summary['std_reward']:6.2f}")
        print(f"Average Daily Cost:       {summary['avg_cost']:8.2f} ± {summary['std_cost']:6.2f} €")
        print(f"Average Efficiency:       {summary['avg_efficiency']:8.3f} ± {summary['std_efficiency']:6.3f}")
        print(f"Demand Satisfaction:      {summary['avg_demand_satisfaction']:8.3f}")
        print(f"Temperature Stability:    {summary['avg_temperature_stability']:8.3f}")

    def _print_comparison_results(self, rl_results: Dict, baseline_results: Dict, improvements: Dict):
        print(f"\nCOMPARISON RESULTS")
        print("-" * 60)
        print(f"{'Metric':<25} {'RL Agent':<12} {'Baseline':<12} {'Improvement':<12}")
        print("-" * 60)

        rl_summary = rl_results['summary']
        baseline_summary = baseline_results['summary']

        print(f"{'Average Reward':<25} {rl_summary['avg_reward']:>8.2f}   {baseline_summary['avg_reward']:>8.2f}   {improvements['reward_improvement']:>+8.1f}%")
        print(f"{'Daily Cost (€)':<25} {rl_summary['avg_cost']:>8.2f}   {baseline_summary['avg_cost']:>8.2f}   {improvements['cost_reduction']:>+8.1f}%")
        print(f"{'Efficiency':<25} {rl_summary['avg_efficiency']:>8.3f}   {baseline_summary['avg_efficiency']:>8.3f}   {improvements['efficiency_improvement']:>+8.1f}%")
        print(f"{'Demand Satisfaction':<25} {rl_summary['avg_demand_satisfaction']:>8.3f}   {baseline_summary['avg_demand_satisfaction']:>8.3f}   {improvements['demand_satisfaction_improvement']:>+8.1f}%")

        print(f"\nOVERALL ASSESSMENT:")
        if improvements['reward_improvement'] > 5 and improvements['cost_reduction'] > 0:
            print("✓ RL Agent significantly outperforms baseline!")
        elif improvements['reward_improvement'] > 0:
            print("✓ RL Agent performs better than baseline")
        else:
            print("⚠ RL Agent needs improvement - consider longer training or hyperparameter tuning")

    def _print_algorithm_comparison_table(self, df: pd.DataFrame):
        print(f"\nALGORITHM PERFORMANCE COMPARISON")
        print("-" * 80)
        print(f"{'Algorithm':<12} {'Reward':<10} {'Cost (€)':<10} {'Efficiency':<10} {'Demand Sat.':<12} {'Temp. Stab.':<12}")
        print("-" * 80)

        for _, row in df.iterrows():
            print(f"{row['algorithm']:<12} {row['avg_reward']:>7.2f}   {row['avg_cost']:>7.2f}   "
                  f"{row['avg_efficiency']:>7.3f}    {row['avg_demand_satisfaction']:>7.3f}     "
                  f"{row['avg_temperature_stability']:>7.3f}")

    def _identify_best_performers(self, df: pd.DataFrame) -> Dict[str, str]:
        return {
            'highest_reward': df.loc[df['avg_reward'].idxmax(), 'algorithm'],
            'lowest_cost': df.loc[df['avg_cost'].idxmin(), 'algorithm'],
            'highest_efficiency': df.loc[df['avg_efficiency'].idxmax(), 'algorithm'],
            'best_demand_satisfaction': df.loc[df['avg_demand_satisfaction'].idxmax(), 'algorithm'],
            'best_temperature_stability': df.loc[df['avg_temperature_stability'].idxmax(), 'algorithm']
        }

    def _create_comparison_plots(self, df: pd.DataFrame, model_results: List[Dict]) -> List[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_saved = []

        plt.style.use('seaborn-v0_8')

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)

        metrics = ['avg_reward', 'avg_cost', 'avg_efficiency',
                   'avg_demand_satisfaction', 'avg_temperature_stability']
        titles = ['Average Reward', 'Average Cost (€)', 'Average Efficiency',
                  'Demand Satisfaction', 'Temperature Stability']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            if idx < 5:
                row = idx // 3
                col = idx % 3

                bars = axes[row, col].bar(df['algorithm'], df[metric], alpha=0.8)
                axes[row, col].set_title(title)
                axes[row, col].set_ylabel(metric.replace('avg_', '').replace('_', ' ').title())

                for bar, value in zip(bars, df[metric]):
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                        f'{value:.3f}', ha='center', va='bottom')

        fig.delaxes(axes[1, 2])

        plt.tight_layout()
        plot_file = self.plots_dir / f'algorithm_comparison_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots_saved.append(str(plot_file))

        if len(model_results) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Episode Progression Comparison', fontsize=16)

            for result in model_results:
                algorithm = result['algorithm']
                episodes = [ep['episode'] for ep in result['episode_results']]
                rewards = [ep['reward'] for ep in result['episode_results']]
                costs = [ep['cost'] for ep in result['episode_results']]
                efficiencies = [ep['efficiency'] for ep in result['episode_results']]

                axes[0, 0].plot(episodes, rewards, marker='o', label=algorithm, alpha=0.8)
                axes[0, 1].plot(episodes, costs, marker='s', label=algorithm, alpha=0.8)
                axes[1, 0].plot(episodes, efficiencies, marker='^', label=algorithm, alpha=0.8)

            axes[0, 0].set_title('Reward per Episode')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].set_title('Cost per Episode')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Cost (€)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].set_title('Efficiency per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Efficiency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            fig.delaxes(axes[1, 1])

            plt.tight_layout()
            plot_file = self.plots_dir / f'episode_progression_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plots_saved.append(str(plot_file))

        print(f"Comparison plots saved: {len(plots_saved)} files")
        return plots_saved

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algorithm = results.get('algorithm', 'unknown')
            filename = f"evaluation_{algorithm.lower()}_{timestamp}.json"

        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")
        return str(filepath)


if __name__ == "__main__":
    evaluator = DistrictHeatingEvaluator()
    print("District Heating Evaluator ready for use!")
    print("\nExample usage:")
    print("results = evaluator.evaluate_model('models/ppo_model.zip', DistrictHeatingEnv, env_kwargs)")
    print("comparison = evaluator.compare_with_baseline('models/ppo_model.zip', DistrictHeatingEnv, env_kwargs)")