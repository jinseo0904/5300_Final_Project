from typing import Dict
import pandas as pd
import os
from glob import glob


def process_subject_data(subject_path: str) -> pd.DataFrame:
    """
    Process and combine stress, physical activity, and phone usage data for one subject.
    Returns hourly-level dataframe with all metrics.
    """

    def get_stress_data() -> pd.DataFrame:
        ema_files = glob(os.path.join(subject_path, "*phone_promptresponse*.csv"))
        if not ema_files:
            return pd.DataFrame()

        dfs = []
        for file in ema_files:
            df = pd.read_csv(file, low_memory=False)
            df = df[df['Answer_Status'] == 'Completed']

            # Clean and parse datetime
            df['Initial_Prompt_Local_Time'] = df['Initial_Prompt_Local_Time'].str.replace('EDT', '').str.replace('EST', '')
            df['Initial_Prompt_Local_Time'] = df['Initial_Prompt_Local_Time'].str.replace('-', '/')
            df['Initial_Prompt_Local_Time'] = df['Initial_Prompt_Local_Time'].str.strip()
            df['hour'] = pd.to_datetime(df['Initial_Prompt_Local_Time'], format='%Y/%m/%d %H:%M:%S').dt.floor('h')

            dfs.append(df)

        return pd.concat(dfs) if dfs else pd.DataFrame()

    def get_physical_activity() -> pd.DataFrame:
        """Get hourly physical activity from MIMS data"""
        activity_files = glob(os.path.join(subject_path, "*watch_accelerometer_mims_hour*.csv"))
        if not activity_files:
            return pd.DataFrame()

        dfs = []
        for file in activity_files:
            df = pd.read_csv(file)
            # Create datetime from YEAR_MONTH_DAY and HOUR
            df['hour'] = pd.to_datetime(df['YEAR'].astype(str) + '/' + df['MONTH'].astype(str) + '/' + df['DAY'].astype(str) + ' ' + df['HOUR'].astype(str) + ':00:00')
            df['activity_level'] = df['MIMS_SUM_WEAR']
            dfs.append(df[['hour', 'activity_level']])

        return pd.concat(dfs) if dfs else pd.DataFrame()

    def get_phone_usage() -> pd.DataFrame:
        """Get hourly phone usage duration"""
        usage_files = glob(os.path.join(subject_path, "*phone_apps_usage_duration*.csv"))
        if not usage_files:
            return pd.DataFrame()

        dfs = []
        for file in usage_files:
            df = pd.read_csv(file)
            # Convert LOG_TIME to datetime and floor to hour
            df['hour'] = pd.to_datetime(df['LOG_TIME'].str.split(' ').str[:2].str.join(' ')).dt.floor('h')
            hourly_usage = df.groupby('hour')['USAGE_DURATION_MIN'].sum().reset_index()
            dfs.append(hourly_usage)

        return pd.concat(dfs) if dfs else pd.DataFrame()

    # Get all data
    stress_df = get_stress_data()
    activity_df = get_physical_activity()
    usage_df = get_phone_usage()

    # Verify we have data from all sources
    if not all([len(df) > 0 for df in [stress_df, activity_df, usage_df]]):
        raise ValueError("Missing one or more required datasets")

    # Merge all data on hour
    final_df = stress_df.merge(activity_df, on='hour', how='inner')\
                        .merge(usage_df, on='hour', how='inner')

    return final_df


def analyze_relationships(df: pd.DataFrame) -> Dict:
    """Calculate correlations and basic statistics"""
    corr_matrix = df[['stress_level', 'activity_level', 'USAGE_DURATION_MIN']].corr()

    stats = {
        'total_hours': len(df),
        'avg_stress': df['stress_level'].mean(),
        'avg_activity': df['activity_level'].mean(),
        'avg_phone_usage': df['USAGE_DURATION_MIN'].mean(),
        'correlations': corr_matrix.to_dict()
    }

    return stats


if __name__ == "__main__":
    SUBJECT_ID = "afflictedrevenueepilepsy"
    SUBJECT_PATH = f"/Volumes/T7/intermediate_file/{SUBJECT_ID}@timestudy_com"

    try:
        combined_df = process_subject_data(SUBJECT_PATH)
        # results = analyze_relationships(combined_df)

        # print(f"\nAnalysis Results:")
        # print(f"Total hours analyzed: {results['total_hours']}")
        # print(f"Average stress level: {results['avg_stress']:.2f}")
        # print(f"Average activity level: {results['avg_activity']:.2f}")
        # print(f"Average phone usage (minutes/hour): {results['avg_phone_usage']:.2f}")
        # print("\nCorrelations:")
        # print(pd.DataFrame(results['correlations']))

        combined_df.to_csv(f"{SUBJECT_ID}_processed_data.csv", index=False)

    except Exception as e:
        print(f"Error processing data: {str(e)}")
