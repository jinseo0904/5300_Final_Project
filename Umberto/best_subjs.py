import pandas as pd
import os
from glob import glob
from typing import Tuple
from tqdm import tqdm


def analyze_data_quality(base_path: str) -> pd.DataFrame:
    """
    Analyzes data quality metrics across subjects.

    Args:
        base_path: Root path containing the data files

    Returns:
        DataFrame with subject-level quality metrics
    """

    def get_watch_data_completeness(subject_id: str) -> float:
        """
        Analyzes watch accelerometer data completeness from missingness report
        """
        try:
            # Read missingness report
            # missingness_files = glob(os.path.join(base_path, subject_id, "*watch_accel_missingness_report*.csv"))
            # if not missingness_files:
            #     return 0.0

            required_files = [
                "phone_detected_activity_day.csv",
                "watch_accelerometer_mims_day.csv",
                "watch_accelerometer_swan_day.csv"
            ]

            for file in required_files:
                if not os.path.exists(os.path.join(
                        base_path, subject_id, file)):
                    # print("Subject and missing file: ", subject_id, file)
                    return 0.0

            return 100.0
        except Exception as e:
            print(
                f"Error processing watch data for subject {subject_id}: {str(e)}")
            return 0.0

    def get_ema_completeness(subject_id: str) -> Tuple[float, float, int]:
        """
        Analyzes EMA response rates for both phone and watch
        """
        try:
            # Phone EMAs
            phone_files = glob(os.path.join(
                base_path, subject_id, "*phone_promptresponse*.csv"))
            phone_responses = []
            for file in phone_files:
                df = pd.read_csv(file, low_memory=False)
                phone_responses.append(df)

            if phone_responses:
                phone_df = pd.concat(phone_responses)
                phone_complete = len(
                    phone_df[phone_df['Answer_Status'] == 'Completed'])
                phone_total = len(phone_df)
                phone_rate = (phone_complete / phone_total * 100) if phone_total > 0 else 0.0
            else:
                phone_rate, phone_total = 0.0, 0

            # Watch EMAs (μEMA)
            watch_files = glob(os.path.join(
                base_path, subject_id, "*watch_promptresponse*.csv"))
            watch_responses = []
            for file in watch_files:
                df = pd.read_csv(file)
                watch_responses.append(df)

            if watch_responses:
                watch_df = pd.concat(watch_responses)
                watch_complete = len(watch_df[watch_df['Answer_Status'].isin(
                    ['Completed', 'CompletedThenDismissed'])])
                watch_total = len(watch_df)
                watch_rate = (watch_complete / watch_total * 100) if watch_total > 0 else 0.0
            else:
                watch_rate, watch_total = 0.0, 0

            return phone_rate, watch_rate, phone_total + watch_total
        except Exception as e:
            print(f"Error processing EMAs for subject {subject_id}: {str(e)}")
            return 0.0, 0.0, 0

    def get_phone_usage_data(subject_id: str) -> Tuple[float, int]:
        """
        Analyzes phone usage data completeness
        """
        try:
            usage_files = glob(os.path.join(
                base_path, subject_id, "*phone_app_usage*.csv"))
            if not usage_files:
                return 0.0, 0

            df_list = []
            for file in usage_files:
                df = pd.read_csv(file)
                df_list.append(df)

            if not df_list:
                return 0.0, 0

            df_combined = pd.concat(df_list)

            # Get unique days with data
            days_with_data = len(df_combined['LOG_TIME'].str[:10].unique())

            return days_with_data, len(df_combined)
        except Exception as e:
            print(
                f"Error processing phone usage for subject {subject_id}: {str(e)}")
            return 0.0, 0

    # Get all subject IDs
    subject_dirs = [d for d in os.listdir(
        base_path) if os.path.isdir(os.path.join(base_path, d))]

    results = []
    for subject_id in tqdm(subject_dirs, desc="Processing subjects"):
        watch_completeness = get_watch_data_completeness(subject_id)
        if watch_completeness == 100.0:
            phone_ema_rate, watch_ema_rate, total_emas = get_ema_completeness(
                subject_id)
            days_with_usage, usage_events = get_phone_usage_data(subject_id)

            results.append({
                'subject_id': subject_id,
                'watch_data_completeness': watch_completeness,
                'phone_ema_response_rate': phone_ema_rate,
                'watch_ema_response_rate': watch_ema_rate,
                'total_emas': total_emas,
                'days_with_phone_data': days_with_usage,
                'phone_usage_events': usage_events,
                # Composite score (you can adjust weights as needed)
                'data_quality_score': (0.4 * watch_completeness + 0.3 * phone_ema_rate + 0.3 * watch_ema_rate)
            })

    return pd.DataFrame(results)


def get_top_subjects(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Returns the top n subjects based on data quality score
    """
    return df.nlargest(n, 'data_quality_score')


# Usage example:
if __name__ == "__main__":
    # Replace with your actual data path
    DATA_PATH = "/media/umberto/T7/intermediate_file"

    # Analyze all subjects
    quality_df = analyze_data_quality(DATA_PATH)

    # Get top 10 subjects
    top_subjects = get_top_subjects(quality_df)

    print("\nTop 10 Subjects by Data Quality:")
    print(top_subjects.to_string())

    # Save results to CSV
    # quality_df.to_csv("data_quality_analysis.csv", index=False)
    top_subjects.to_csv("top_10_subjects.csv", index=False)
