import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_weather_data(csv_file):

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])

        df['date'] = df['datetime'].dt.date
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['season'] = df['month'].apply(lambda x:
                                         1 if x in [12, 1, 2] else
                                         2 if x in [3, 4, 5] else
                                         3 if x in [6, 7, 8] else
                                         4)

    print("\n===== Weather Data Analysis =====")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0])

    key_columns = ['temp', 'app_temp', 'rh', 'wind_spd', 'clouds', 'precip', 'solar_rad']
    available_columns = [col for col in key_columns if col in df.columns]

    print("\nStatistics for key weather parameters:")
    print(df[available_columns].describe())

    print("\nRecords per month/year:")
    monthly_counts = df.groupby(['year', 'month']).size()
    print(monthly_counts)

    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    processed_file = f"{output_dir}/processed_weather_data.csv"
    df.to_csv(processed_file, index=False)
    print(f"\nProcessed data saved to {processed_file}")

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['temp'], linewidth=0.5)
    plt.title('Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/temperature_over_time.png")
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.boxplot(x='month', y='temp', data=df)
    plt.title('Monthly Temperature Distribution')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.savefig(f"{plots_dir}/monthly_temperature_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    correlation = df[available_columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Between Weather Parameters')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for season in range(1, 5):
        season_data = df[df['season'] == season]
        hourly_avg = season_data.groupby('hour')['temp'].mean()
        plt.plot(hourly_avg.index, hourly_avg.values, label=f"Season {season}")
    plt.title('Average Hourly Temperature by Season')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Temperature (°C)')
    plt.legend(labels=['Winter', 'Spring', 'Summer', 'Fall'])
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/hourly_temperature_by_season.png")
    plt.close()

    print(f"Visualizations saved to {plots_dir}/")

    return df

if __name__ == "__main__":
    weather_file = "weatherbit_data/schweinfurt_weather_2021-01-01_to_2025-06-05.csv"
    weather_data = analyze_weather_data(weather_file)