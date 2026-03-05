import pandas as pd
import os
from utils import load_weather_data, save_dataframe
from district_heating_parameters import create_district_heating_parameters
from temperature_curve_generation import generate_expected_temperature_data

def create_merged_dataset(weather_data_path, output_dir="processed_data"):

    weather_data = load_weather_data(weather_data_path)
    if weather_data is None:
        print("Failed to load weather data. Exiting.")
        return None

    print("Loading district heating parameters...")
    heating_params_df, zone_params_dict = create_district_heating_parameters()

    expected_temp_df = generate_expected_temperature_data(weather_data, zone_params_dict)

    expected_temp_filepath = f"{output_dir}/expected_temperatures.csv"
    save_dataframe(expected_temp_df, expected_temp_filepath)

    merged_df = pd.merge(weather_data, expected_temp_df, on='datetime', how='left')

    merged_df['hdd_18'] = merged_df['temp'].apply(lambda x: max(0, 18 - x))
    merged_df['hdd_15_5'] = merged_df['temp'].apply(lambda x: max(0, 15.5 - x))

    merged_df['temp_change'] = merged_df['temp'].diff()

    merged_df['is_daytime'] = ((merged_df['hour'] >= 6) & (merged_df['hour'] < 18)).astype(int)

    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)

    merged_filepath = f"{output_dir}/merged_dataset.csv"
    save_dataframe(merged_df, merged_filepath)

    return merged_df

def find_weather_data_file():

    possible_files = [
        "processed_data/processed_weather_data.csv",
        "weatherbit_data/schweinfurt_weather_2021-01-01_to_2025-06-05.csv"
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Found weather data: {file_path}")
            return file_path

    print("No weather data file found!")
    return None

if __name__ == "__main__":

    weather_data_path = find_weather_data_file()

    if weather_data_path:
        merged_df = create_merged_dataset(weather_data_path)

        if merged_df is not None:
            print("Sample of merged data:")
            print(merged_df.head())
            print(f"\nDataset summary:")
            print(f"- Total records: {len(merged_df):,}")
            print(f"- Date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")
            print(f"- Total features: {len(merged_df.columns)}")
    else:
        print("No weather data found. Please run analyze_weather.py first.")