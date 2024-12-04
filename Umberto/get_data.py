from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import os

SOCIAL_MEDIA_PACKAGES = {
    'com.instagram.android',
    'com.facebook.katana',
    'com.twitter.android',
    'com.snapchat.android',
    'com.whatsapp',
    'com.linkedin.android',
    'com.pinterest',
    'com.reddit.frontpage',
    'com.tiktok.app',
    'com.tumblr',
    'com.facebook.orca'
}

WATCH_QUESTIONS_OF_INTEREST = {
    'proc_4', 'frust_12', 'tense_11', 'int_exer_3', 'happy_8',
    'focus_3', 'relax_10', 'sad_7', 'stress_2', 'excite_14',
    'nervous_4', 'trait_31'
}

# Map responses to numeric values for analysis
RESPONSE_MAP = {
    'yes': 5,
    'frequently': 5,
    'sort of': 4,
    'sometimes': 3,
    'rarely': 2,
    'no': 1
}


class TimeStudyProcessor:
    def __init__(self, subjects_file: Path, data_root: Path):
        self.data_root = Path(data_root)
        self.subject_ids = self._load_subject_ids(subjects_file)
        self.subject_paths = self._get_subject_paths()

    def _load_subject_ids(self, file_path: Path) -> List[str]:
        return pd.read_csv(file_path)['subject_id'].tolist()

    def _get_subject_paths(self) -> Dict[str, Path]:
        return {
            subject_id: self.data_root / subject_id
            for subject_id in self.subject_ids
            if (self.data_root / subject_id).exists()
        }

    def _process_datetime(self, time_str: str) -> pd.Timestamp:
        """Process datetime string by removing timezone abbreviation"""
        # Remove all possible US timezone abbreviations
        for tz in ['EDT', 'EST', 'CDT', 'CST', 'MDT', 'MST', 'PDT', 'PST']:
            time_str = time_str.replace(tz, '')
        return pd.to_datetime(time_str.strip())

    def _create_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create consistent date column from YEAR, MONTH, DAY columns"""
        if not df.empty and all(col in df.columns for col in ['YEAR', 'MONTH', 'DAY']):
            df['date'] = pd.to_datetime(
                df[['YEAR', 'MONTH', 'DAY']].astype(str).agg('-'.join, axis=1)
            ).dt.date  # Convert to date object for consistent type
        return df

    def process_single_subject(self, subject_id: str):
        """Process data for a single subject"""
        subject_path = self.subject_paths[subject_id]

        # Read all required files
        activities_df = self._read_and_process_activities(subject_path)
        mims_df = self._read_and_process_mims(subject_path)
        swan_df = self._read_and_process_swan(subject_path)
        screen_df = self._read_and_process_screen(subject_path)
        social_media_df = self._read_and_process_social_media(subject_path)
        # phone_ema_df = self._read_and_process_phone_ema(subject_path)
        watch_ema_df = self._read_and_process_watch_ema(subject_path)

        # Combine all metrics
        daily_df = self._combine_daily_metrics(
            subject_id.split('@')[0],
            activities_df,
            mims_df,
            swan_df,
            screen_df,
            social_media_df,
            # phone_ema_df,
            watch_ema_df
        )

        # Save results
        if not daily_df.empty:
            output_path = os.path.join("Umberto/data", f"{subject_id.split('@')[0]}_daily_metrics.csv")
            daily_df.to_csv(output_path, index=False)

    def _read_and_process_activities(self, subject_path: Path) -> pd.DataFrame:
        df = self._read_csv_safe(subject_path / "phone_detected_activity_day.csv")
        if not df.empty:
            df = self._create_date_column(df)
            activity_cols = ['IN_VEHICLE', 'ON_BIKE', 'ON_FOOT', 'RUNNING',
                             'STILL', 'TILTING', 'WALKING', 'UNKNOWN']
            return df[['date'] + activity_cols]
        return pd.DataFrame()

    def _read_and_process_mims(self, subject_path: Path) -> pd.DataFrame:
        df = self._read_csv_safe(subject_path / "watch_accelerometer_mims_day.csv")
        if not df.empty:
            df = self._create_date_column(df)
            return df[['date', 'MIMS_SUM_WEAR']]
        return pd.DataFrame()

    def _read_and_process_swan(self, subject_path: Path) -> pd.DataFrame:
        df = self._read_csv_safe(subject_path / "watch_accelerometer_swan_day.csv")
        if not df.empty:
            df = self._create_date_column(df)
            return df[['date', 'SLEEP_MINUTES']]
        return pd.DataFrame()

    def _read_and_process_screen(self, subject_path: Path) -> pd.DataFrame:
        df = self._read_csv_safe(subject_path / "phone_screen_status_day.csv")
        if not df.empty:
            df = self._create_date_column(df)
            return df[['date', 'SCREEN_ON_SECONDS', 'UNLOCK_EVENTS_NUM']]
        return pd.DataFrame()

    def _read_and_process_social_media(self, subject_path: Path) -> pd.DataFrame:
        df = self._read_csv_safe(subject_path / "phone_apps_usage_duration.csv")
        if df.empty:
            return pd.DataFrame()

        social_media_df = df[df['APP_PACKAGE_NAME'].isin(SOCIAL_MEDIA_PACKAGES)].copy()

        if not social_media_df.empty:
            # Ensure consistent date type
            social_media_df['date'] = pd.to_datetime(
                social_media_df['LOG_TIME'].str.split().str[:2].str.join(' ')
            ).dt.date
            return social_media_df.groupby('date')['USAGE_DURATION_MIN'].sum().reset_index()

        return pd.DataFrame()

    # def _read_and_process_phone_ema(self, subject_path: Path) -> pd.DataFrame:
    #     """Process phone EMA responses"""
    #     df = self._read_csv_safe(subject_path / "phone_promptresponse.csv")
    #     if df.empty:
    #         return pd.DataFrame()

    #     # Filter for completed responses
    #     df = df[df['Answer_Status'] == 'Completed'].copy()

    #     # Convert timestamp to date
    #     df['date'] = df['Initial_Prompt_Local_Time'].apply(self._process_datetime).dt.date

    #     # Aggregate by date - counting responses
    #     daily_ema = df.groupby('date').size().reset_index(name='phone_ema_count')

    #     return daily_ema

    def _read_and_process_watch_ema(self, subject_path: Path) -> pd.DataFrame:
        """Process watch EMA responses for specific questions"""
        df = self._read_csv_safe(subject_path / "watch_promptresponse.csv")
        if df.empty:
            return pd.DataFrame()

        # Filter for completed responses and exclude Trivia
        df = df[
            (df['Answer_Status'] == 'Completed') &
            (df['Prompt_Type'] != 'Trivia_EMA_Micro')
        ].copy()

        if df.empty:
            return pd.DataFrame()

        # Get answer if Question_X_ID matches our interests
        if df['Question_X_ID'].isin(WATCH_QUESTIONS_OF_INTEREST).any():
            # Create date from Unix timestamp
            df['date'] = pd.to_datetime(df['Question_X_Answer_Unixtime'], unit='ms').dt.date

            # Map answers to numeric values
            df['answer_value'] = df['Question_X_Answer_Text '].str.lower().map(RESPONSE_MAP)

            # Drop rows where mapping failed
            df = df.dropna(subset=['answer_value'])

            # Create DataFrame with date and answer
            responses_df = df[['date', 'Question_X_ID', 'answer_value']].copy()

            # Pivot to get one column per question type
            daily_ema = responses_df.pivot_table(
                index='date',
                columns='Question_X_ID',
                values='answer_value',
                aggfunc='mean'
            ).reset_index()

            # Rename columns to add _response suffix
            daily_ema.columns = [
                'date' if col == 'date' else f'{col}_response'
                for col in daily_ema.columns
            ]

            return daily_ema

        return pd.DataFrame()

    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        """Safely read CSV file, return empty DataFrame if file doesn't exist"""
        if path.exists():
            return pd.read_csv(path, low_memory=False)
        return pd.DataFrame()

    def _combine_daily_metrics(self,
                               subject_id: str,
                               activities_df: pd.DataFrame,
                               mims_df: pd.DataFrame,
                               swan_df: pd.DataFrame,
                               screen_df: pd.DataFrame,
                               social_media_df: pd.DataFrame,
                               # phone_ema_df: pd.DataFrame,
                               watch_ema_df: pd.DataFrame
                               ) -> pd.DataFrame:
        """Combine all metrics into a single daily DataFrame"""
        # Get all valid dates
        date_sets = []
        for df in [activities_df, mims_df, swan_df, screen_df,
                   social_media_df, watch_ema_df]:
            if not df.empty and 'date' in df.columns:
                date_sets.append(set(df['date']))

        if not date_sets:
            return pd.DataFrame()

        # Get the complete date range
        all_dates = sorted(set.union(*date_sets))

        # Create base DataFrame with consistent date type
        daily_df = pd.DataFrame({
            'id': subject_id,
            'date': all_dates
        })

        # Merge all metrics
        dfs_to_merge = [
            (activities_df, ''),
            (mims_df, ''),
            (swan_df, ''),
            (screen_df, ''),
            (social_media_df, '_social_media'),
            # (phone_ema_df, ''),
            (watch_ema_df, '')
        ]

        for df, suffix in dfs_to_merge:
            if not df.empty:
                daily_df = daily_df.merge(df, on='date', how='left', suffixes=('', suffix))

        return daily_df

    def process_all_subjects(self):
        """Process data for all subjects"""
        failed_subjects = []
        for subject_id in tqdm(self.subject_paths, desc="Processing subjects"):
            try:
                self.process_single_subject(subject_id)
            except Exception as e:
                print(f"Error processing subject {subject_id}: {str(e)}")
                failed_subjects.append(subject_id)

        if failed_subjects:
            print(f"\nFailed to process {len(failed_subjects)} subjects: {failed_subjects}")


def main():
    subjects_file = Path("Umberto/data/top_10_subjects.csv")
    data_root = Path("/media/umberto/T7/intermediate_file")

    processor = TimeStudyProcessor(subjects_file, data_root)
    processor.process_all_subjects()


if __name__ == "__main__":
    main()
