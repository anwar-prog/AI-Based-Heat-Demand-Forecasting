Title: AI-Based Heat Demand Forecasting

Author: Sharik Anwar Zahir Hussain

Institution: Technische Hochschule Würzburg-Schweinfurt (THWS)

Thesis Year: 2025



Description:

This repository contains the complete source code and supporting files for the master’s thesis “AI-Based Optimization Framework for the Schweinfurt District Heating Network.” The project integrates multi-horizon heat demand forecasting and reinforcement learning–based control optimization using real operational data from the Schweinfurt district heating system.



Repository Structure:



01\_data/ – Raw, processed, and logged datasets (proprietary data excluded).



02\_preprocessing/ – Feature generation and selection scripts used to prepare model inputs.



03\_baseline\_models/ – Traditional forecasting baselines (MLR) with performance results.



04\_svr\_implementation/, 05\_ffnn\_implementation/, 06\_lstm\_implementation/ – Machine learning and deep learning models trained for multiple forecasting horizons (1h–72h).



07\_utilities/ – Common helper modules for evaluation, visualization, and configuration.



08\_cache\_and\_temp/ – Temporary storage and compiled cache files (non-critical).



RL/ – Reinforcement Learning and Model Predictive Control implementations:



RL/sac\_implementation/ – Soft Actor-Critic training scripts and result plots.



RL/td3\_implementation/ – TD3 experiments with evaluation logs, models, and plots.



RL/rl\_mpc\_implementation/ – Hybrid RL–MPC integration, including configuration and results folders.



Dependencies:



Python ≥ 3.9



NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib, Seaborn, Gymnasium, Stable-Baselines3, OSQP



Install all dependencies:



pip install -r requirements.txt





Execution Example:



Preprocess data



python 02\_preprocessing/preprocess.py





Train baseline models



python 04\_svr\_implementation/train\_svr.py





Run hybrid RL–MPC simulation



python RL/rl\_mpc\_implementation/main\_hybrid.py





Note:

The dataset used for this work originates from the Schweinfurt District Heating Network and is not publicly available. Only anonymized or sample data may be included for reference.

