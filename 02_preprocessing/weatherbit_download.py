import requests
import pandas as pd
import time
import datetime
import os
import json
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weatherbit_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEY = "128b7e85758749d5ab49cbbe37121ca6"
LATITUDE = "50.045"
LONGITUDE = "10.234"

DATA_DIR = Path("weatherbit_data")
DATA_DIR.mkdir(exist_ok=True)

def get_historical_weather(start_date, end_date, max_retries=3):

    base_url = "https://api.weatherbit.io/v2.0/history/hourly"
    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "key": API_KEY
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"Requesting data from {start_date} to {end_date} (attempt {attempt + 1})")
            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    logger.info(f"Successfully retrieved {len(data['data'])} records")
                    return data
                else:
                    logger.warning(f"Empty data response for {start_date} to {end_date}")
                    return None
            elif response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(10 * (attempt + 1))

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return None
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            logger.error(f"Exception on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(5 * (attempt + 1))

    return None

def calculate_optimal_chunk_size(start_date, end_date):

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    if total_days > 1400:
        return 15
    elif total_days > 1000:
        return 20
    elif total_days > 730:
        return 25
    else:
        return 30

def download_historical_data(start_date, end_date, output_dir="weatherbit_data", resume=True):

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    chunk_days = calculate_optimal_chunk_size(start_date, end_date)

    all_data = []
    current = start

    total_days = (end - start).days
    total_chunks = (total_days + chunk_days - 1) // chunk_days

    logger.info(f"Downloading data from {start_date} to {end_date}")
    logger.info(f"Total time span: {total_days} days, will be processed in {total_chunks} chunks of {chunk_days} days each")

    successful_chunks = 0
    failed_chunks = []

    with tqdm(total=total_chunks, desc="Downloading weather data") as pbar:
        for chunk_idx in range(total_chunks):
            next_date = min(current + datetime.timedelta(days=chunk_days), end)

            start_str = current.strftime("%Y-%m-%d")
            end_str = next_date.strftime("%Y-%m-%d")

            chunk_file = Path(output_dir) / f"chunk_{chunk_idx:03d}_{start_str}_to_{end_str}.json"

            if resume and chunk_file.exists():
                try:
                    with open(chunk_file, 'r') as f:
                        response = json.load(f)
                    logger.info(f"Loading existing chunk {chunk_idx + 1}/{total_chunks}: {start_str} to {end_str}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Corrupted chunk file {chunk_file}, re-downloading...")
                    response = get_historical_weather(start_str, end_str)
            else:
                response = get_historical_weather(start_str, end_str)

                if response:
                    try:
                        with open(chunk_file, 'w') as f:
                            json.dump(response, f, indent=2)
                    except IOError as e:
                        logger.error(f"Failed to save chunk {chunk_file}: {e}")

            if response and 'data' in response:
                chunk_data = response['data']
                all_data.extend(chunk_data)
                successful_chunks += 1

                logger.info(f"Chunk {chunk_idx + 1}/{total_chunks}: Collected {len(chunk_data)} hours")

                pbar.set_postfix({
                    'Successful': successful_chunks,
                    'Records': len(all_data),
                    'Current': f"{start_str} to {end_str}"
                })

                time.sleep(2)
            else:
                failed_chunks.append((chunk_idx + 1, start_str, end_str))
                logger.error(f"Failed to fetch chunk {chunk_idx + 1}: {start_str} to {end_str}")

            pbar.update(1)

            current = next_date

    logger.info(f"Download complete: {successful_chunks}/{total_chunks} chunks successful")
    if failed_chunks:
        logger.warning(f"Failed chunks: {failed_chunks}")

    if all_data:
        df = pd.DataFrame(all_data)

        if 'datetime' in df.columns:
            try:
                if df['datetime'].dtype == 'object':
                    df['datetime'] = df['datetime'].str.replace(':00,', ':00')
                    df['datetime'] = df['datetime'].str.replace(':', ':')

                try:
                    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d:%H')
                except:
                    try:
                        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S')
                    except:
                        df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)

                df = df.sort_values('datetime').reset_index(drop=True)
                logger.info("Successfully parsed datetime column")
            except Exception as e:
                logger.warning(f"Datetime parsing issue: {e}")
                logger.info("Continuing without datetime sorting - data is still valid")

        output_file = Path(output_dir) / f"schweinfurt_weather_{start_date}_to_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"All data saved to {output_file}")
        logger.info(f"Total records collected: {len(df)}")

        return df
    else:
        logger.error("No data was collected")
        return None

def analyze_weather_data(df):

    if df is None or df.empty:
        logger.error("No data available for analysis")
        return

    logger.info("===== Weather Data Analysis =====")
    logger.info(f"Total records: {len(df):,}")

    if 'datetime' in df.columns:
        try:
            if df['datetime'].dtype == 'object':
                df['datetime'] = df['datetime'].str.replace(':00,', ':00')
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

            time_span = df['datetime'].max() - df['datetime'].min()
            expected_hours = int(time_span.total_seconds() / 3600) + 1
            actual_hours = len(df)
            completeness = actual_hours / expected_hours * 100

            logger.info(f"Data completeness: {actual_hours:,}/{expected_hours:,} hours ({completeness:.1f}%)")
        except Exception as e:
            logger.warning(f"DateTime analysis failed: {e}")
            logger.info("Skipping datetime analysis - data is still valid")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning("Missing values detected:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
    else:
        logger.info("No missing values found")

    key_columns = ['temp', 'app_temp', 'rh', 'wind_spd', 'clouds', 'precip', 'solar_rad']
    available_columns = [col for col in key_columns if col in df.columns]

    if available_columns:
        logger.info("\nStatistics for key weather parameters:")
        stats = df[available_columns].describe()
        print(stats.round(2))

    if 'datetime' in df.columns:
        df['year_month'] = df['datetime'].dt.to_period('M')
        monthly_counts = df.groupby('year_month').size()

        logger.info("\nMonthly data completeness:")
        incomplete_months = []
        for period, count in monthly_counts.items():
            expected_hours = period.days_in_month * 24
            if count < expected_hours:
                incomplete_months.append((period, count, expected_hours))

        if incomplete_months:
            logger.warning("Months with incomplete data:")
            for period, actual, expected in incomplete_months:
                logger.warning(f"  {period}: {actual}/{expected} hours ({actual/expected:.1%})")
        else:
            logger.info("All months have complete hourly data")

def validate_api_key():

    test_response = get_historical_weather("2024-01-01", "2024-01-02")
    if test_response:
        logger.info("API key validation successful")
        return True
    else:
        logger.error("API key validation failed")
        return False

if __name__ == "__main__":
    START_DATE = "2021-01-01"
    END_DATE = "2025-06-05"

    logger.info("Starting maximum weather data download for Schweinfurt")
    logger.info(f"Date range: {START_DATE} to {END_DATE} (4+ years)")
    logger.info("This will collect ~38,700 hours of weather data")

    if not validate_api_key():
        logger.error("API key validation failed. Please check your API key.")
        exit(1)

    weather_data = download_historical_data(
        start_date=START_DATE,
        end_date=END_DATE,
        resume=True
    )

    if weather_data is not None:
        analyze_weather_data(weather_data)

        logger.info("\nData collection summary:")
        logger.info(f"Shape: {weather_data.shape}")
        logger.info(f"Memory usage: {weather_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        logger.info(f"\nAvailable weather parameters ({len(weather_data.columns)}):")
        for i, col in enumerate(weather_data.columns, 1):
            logger.info(f"  {i:2d}. {col}")

        with open(DATA_DIR / "column_info.txt", 'w') as f:
            f.write("Weather Data Columns:\n")
            f.write("=" * 30 + "\n")
            for i, col in enumerate(weather_data.columns, 1):
                f.write(f"{i:2d}. {col}\n")

        logger.info("Weather data download complete!")
        logger.info(f"Final dataset: {len(weather_data):,} records covering 4+ years")
        logger.info("Perfect dataset for your thesis forecasting models!")
    else:
        logger.error("Data collection failed completely")
        exit(1)