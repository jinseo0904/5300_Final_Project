import os
import pandas as pd
from tqdm import tqdm


def process_subject(subject_id: str, base_path: str) -> pd.DataFrame:
    """Generate missingness report for a subject."""
    # Read MIMS minute data for basic timeline
    mims_path = f'{base_path}/{subject_id}/watch_accelerometer_mims_minute.csv'
    if not os.path.exists(mims_path):
        return None

    df = pd.read_csv(mims_path)

    # Create datetime index
    df['datetime'] = pd.to_datetime(df['YEAR'].astype(str) + '/' + df['MONTH'].astype(str) + '/' + df['DAY'].astype(str) + ' ' + df['HOUR'].astype(str) + ':00:00')

    # Initialize all required columns from codebook
    df['ACCEL_MISSING'] = df['MIMS_SAMPLE_NUM'] == 0
    df['WATCH_OFF'] = False  # From PowerConnectionReceiverNotes.log.csv
    df['LOW_WATCH_BATTERY'] = False
    df['LOW_WATCH_RAM'] = False
    df['20_MIN_BUG'] = False
    df['DATA_TRANSFERRING_BUG'] = df['datetime'] < pd.Timestamp('2020-08-01')
    df['WATCH_SYSTEM_TIME_BUG'] = False
    df['WATCH_CHARGING'] = False
    df['NONWEAR_SWAN'] = False
    df['MISSING_CATEGORY'] = 'NONE'

    # Process battery data
    battery_path = f'{base_path}/{subject_id}/watch_battery_clean.csv'
    if os.path.exists(battery_path):
        battery_df = pd.read_csv(battery_path)
        battery_df['LOG_TIME'] = pd.to_datetime(battery_df['LOG_TIME'])
        # Mark low battery periods
        for _, row in battery_df.iterrows():
            if row['BATTERY_LEVEL'] < 10:
                closest_idx = (df['datetime'] - row['LOG_TIME']).abs().argsort()[:1]
                df.loc[closest_idx, 'LOW_WATCH_BATTERY'] = True
            if row['BATTERY_CHARGING'] is True:
                closest_idx = (df['datetime'] - row['LOG_TIME']).abs().argsort()[:1]
                df.loc[closest_idx, 'WATCH_CHARGING'] = True

    # Process SWaN data
    swan_path = f'{base_path}/{subject_id}/watch_accelerometer_swan_minute.csv'
    if os.path.exists(swan_path):
        swan_df = pd.read_csv(swan_path)
        df['NONWEAR_SWAN'] = (swan_df['SWAN_PREDICTION'] == 'Nonwear')

    # Detect 20-minute bug pattern
    missing_blocks = df[df['ACCEL_MISSING']].index.to_numpy()
    for i in range(len(missing_blocks) - 1):
        block_length = missing_blocks[i + 1] - missing_blocks[i]
        if 18 <= block_length <= 22 or (block_length % 20 <= 2 and block_length > 0):
            df.loc[missing_blocks[i]:missing_blocks[i + 1], '20_MIN_BUG'] = True

    # Select required columns
    required_cols = ['PARTICIPANT_ID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'ACCEL_MISSING',
                     'WATCH_OFF', 'LOW_WATCH_BATTERY', 'LOW_WATCH_RAM', '20_MIN_BUG',
                     'DATA_TRANSFERRING_BUG', 'WATCH_SYSTEM_TIME_BUG', 'WATCH_CHARGING',
                     'NONWEAR_SWAN', 'MISSING_CATEGORY']

    df['PARTICIPANT_ID'] = subject_id
    result_df = df[required_cols]

    return result_df


if __name__ == "__main__":
    base_path = '/Volumes/T7/intermediate_file'
    subject_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for subject_id in tqdm(subject_dirs, desc="Processing subjects"):
        result_df = process_subject(subject_id, base_path)
        if result_df is not None:
            result_df.to_csv(f'{base_path}/{subject_id}/watch_accel_missingness_report.csv', index=False)
