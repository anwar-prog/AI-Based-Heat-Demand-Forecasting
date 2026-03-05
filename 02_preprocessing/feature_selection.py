import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
from utils import create_directory, plot_and_save

def correlation_analysis(df, target_col='hdd_18', output_dir='feature_selection'):

    create_directory(output_dir)

    num_df = df.select_dtypes(include=['number'])

    initial_cols = len(num_df.columns)
    num_df = num_df.dropna(axis=1)
    final_cols = len(num_df.columns)

    if initial_cols != final_cols:
        print(f"Dropped {initial_cols - final_cols} columns with NaN values")

    correlations = num_df.corrwith(num_df[target_col])
    correlations = correlations.drop(target_col)

    abs_corr = correlations.abs().sort_values(ascending=False)
    sorted_corr = correlations.loc[abs_corr.index]

    corr_df = pd.DataFrame({
        'feature': sorted_corr.index,
        'correlation': sorted_corr.values,
        'abs_correlation': abs_corr.values
    })
    corr_df.to_csv(f"{output_dir}/feature_correlations.csv", index=False)

    plt.figure(figsize=(12, 8))
    top_features = sorted_corr.head(20)

    colors = ['red' if x < 0 else 'blue' for x in top_features.values]
    bars = plt.barh(range(len(top_features)), top_features.values, color=colors)

    plt.yticks(range(len(top_features)), top_features.index)
    plt.title(f"Top 20 Features Correlated with {target_col}\n(5-Year Dataset: {len(df):,} records)")
    plt.xlabel("Correlation Coefficient")
    plt.grid(axis='x', alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        plt.text(value + 0.01 if value > 0 else value - 0.01, i,
                 f'{value:.3f}', va='center', ha='left' if value > 0 else 'right')

    plt.tight_layout()
    plot_and_save(plt, f"{output_dir}/correlation_analysis.png")

    return sorted_corr

def feature_importance_analysis(df, target_col='hdd_18', output_dir='feature_selection'):

    create_directory(output_dir)

    num_df = df.select_dtypes(include=['number'])
    num_df = num_df.dropna(axis=1)

    X = num_df.drop(columns=[target_col])
    y = num_df[target_col]

    print(f"Training Random Forest on {len(X):,} samples with {len(X.columns)} features")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })

    importance_df = importance_df.sort_values('importance', ascending=False)

    importance_df.to_csv(f"{output_dir}/feature_importances.csv", index=False)

    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)

    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title(f"Top 20 Important Features for Predicting {target_col}\n(Random Forest, 5-Year Dataset)")
    plt.xlabel("Feature Importance")
    plt.grid(axis='x', alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        plt.text(value + 0.001, i, f'{value:.3f}', va='center', ha='left')

    plt.tight_layout()
    plot_and_save(plt, f"{output_dir}/feature_importance.png")

    return importance_df

def mutual_information_analysis(df, target_col='hdd_18', output_dir='feature_selection'):

    print(f"Analyzing mutual information with target: {target_col}")

    num_df = df.select_dtypes(include=['number'])
    num_df = num_df.dropna(axis=1)

    X = num_df.drop(columns=[target_col])
    y = num_df[target_col]

    if len(X) > 10000:
        print("Large dataset detected, sampling 10,000 records for MI analysis")
        sample_idx = np.random.choice(len(X), 10000, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample, y_sample = X, y

    mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)

    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mutual_info': mi_scores
    })

    mi_df = mi_df.sort_values('mutual_info', ascending=False)

    mi_df.to_csv(f"{output_dir}/mutual_information.csv", index=False)

    print(f"  - Highest MI score: {mi_df.iloc[0]['mutual_info']:.3f} ({mi_df.iloc[0]['feature']})")

    return mi_df

def correlation_heatmap(df, output_dir='feature_selection'):

    create_directory(output_dir)

    weather_cols = ['temp', 'app_temp', 'rh', 'dewpt', 'wind_spd', 'clouds', 'solar_rad', 'precip']
    temp_cols = [col for col in df.columns if 'expected_supply_temp' in col]
    hdd_cols = [col for col in df.columns if 'hdd' in col]
    time_cols = ['hour', 'day_of_week', 'month', 'season']
    derived_cols = ['temp_change', 'is_daytime', 'is_weekend']

    key_features = (weather_cols + temp_cols[:5] + hdd_cols +
                    time_cols + derived_cols)
    key_features = [col for col in key_features if col in df.columns]

    print(f"Creating heatmap for {len(key_features)} key features")

    corr_df = df[key_features].corr()

    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    sns.heatmap(
        corr_df,
        mask=mask,
        cmap='coolwarm',
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8}
    )

    plt.title('Correlation Heatmap of Key Features\n(5-Year District Heating Dataset)')
    plt.tight_layout()

    plot_and_save(plt, f"{output_dir}/correlation_heatmap.png")

def select_features(correlation_results, importance_results, mi_results=None, n_features=15):

    top_corr_features = correlation_results.index[:n_features].tolist()
    top_imp_features = importance_results['feature'][:n_features].tolist()

    combined_features = set(top_corr_features + top_imp_features)

    if mi_results is not None:
        top_mi_features = mi_results['feature'][:n_features].tolist()
        combined_features.update(top_mi_features)

    final_features = list(combined_features)

    print(f"Selected {len(final_features)} unique features from multiple methods")

    return final_features

def main():

    merged_file = "processed_data/merged_dataset.csv"
    if not os.path.exists(merged_file):
        print(f"Error: {merged_file} not found. Run data_integration.py first.")
        return

    merged_df = pd.read_csv(merged_file, parse_dates=['datetime'])
    print(f"Loaded merged dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
    print(f"Date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")

    output_dir = "feature_selection"
    create_directory(output_dir)

    target_column = 'hdd_18'
    print(f"Using {target_column} as target variable (proxy for heat demand)")

    corr_results = correlation_analysis(merged_df, target_col=target_column, output_dir=output_dir)

    imp_results = feature_importance_analysis(merged_df, target_col=target_column, output_dir=output_dir)

    mi_results = mutual_information_analysis(merged_df, target_col=target_column, output_dir=output_dir)

    correlation_heatmap(merged_df, output_dir=output_dir)

    selected_features = select_features(corr_results, imp_results, mi_results, n_features=20)

    for i, feature in enumerate(selected_features, 1):
        print(f"{i:2d}. {feature}")

    selected_df = pd.DataFrame({'feature': selected_features})
    selected_df.to_csv(f"{output_dir}/selected_features.csv", index=False)


if __name__ == "__main__":
    main()