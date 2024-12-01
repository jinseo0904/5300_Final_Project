"""
TIME Study Analysis Script.

This script analyzes the relationship between stress/anxiety levels,
physical activity, and mobile phone usage using the TIME study dataset.
"""

import logging
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TimeStudyProcessor:
    """Base class for processing TIME study data."""

    def __init__(self, participant_id: str):
        """Initialize the processor with participant ID.

        Args:
            participant_id: Unique identifier for the participant
        """
        self.participant_id = participant_id
        self.ema_data = None
        self.activity_data = None
        self.phone_usage_data = None

    def load_ema_data(self, file_path: str) -> pd.DataFrame:
        """Load and process EMA responses.

        Args:
            file_path: Path to the EMA data file

        Returns:
            Processed EMA DataFrame
        """
        df = pd.read_csv(file_path, low_memory=False)
        # Split off the timezone abbreviation and keep just the datetime part
        df['Initial_Prompt_Local_Time'] = df['Initial_Prompt_Local_Time'].apply(lambda x: x.rsplit(' ', 1)[0])
        df['Initial_Prompt_Local_Time'] = pd.to_datetime(df['Initial_Prompt_Local_Time'])

        df['timezone_offset'] = df['Initial_Prompt_UTC_Offset ']

        # Filter completed responses
        df = df[df['Answer_Status'].isin(['Completed', 'CompletedThenDismissed'])]
        df = df[df['Prompt_Type'].isin(['EMA', 'EMA_Micro'])]

        processed_responses = []
        for _, row in df.iterrows():
            response_dict = {
                'timestamp': row['Initial_Prompt_Local_Time'],
                'prompt_type': row['Prompt_Type'],
                'study_mode': row['Study_Mode'],
                'timezone_offset': row['timezone_offset']
            }

            question_count = row['Number_Of_Questions_Presented']
            for i in range(question_count):
                q_text = row.get(f'Question_{i}_Text')
                q_answer = row.get(f'Question_{i}_Answer_Text')

                if pd.notna(q_text) and pd.notna(q_answer) and q_answer != '-NOT_ANS-':
                    response_dict[f'Q{i}_text'] = q_text
                    response_dict[f'Q{i}_answer'] = q_answer

            processed_responses.append(response_dict)

        self.ema_data = pd.DataFrame(processed_responses)
        self.ema_data.set_index('timestamp', inplace=True)
        return self.ema_data

    def load_activity_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process physical activity data.
        """
        # Load data
        df = pd.read_csv(file_path)

        # Create timestamp from separate YEAR, MONTH, DAY columns
        df['timestamp'] = pd.to_datetime(
            df['YEAR'].astype(str) + '-' +
            df['MONTH'].astype(str).str.zfill(2) + '-' +
            df['DAY'].astype(str).str.zfill(2) + ' ' +
            df['HOUR'].astype(str).str.zfill(2) + ':00:00'
        )

        # Create activity features using the actual column names
        activity_features = {
            'timestamp': df['timestamp'],
            'total_activity': df['MIMS_SUM'],
            'active_samples': df['MIMS_SAMPLE_NUM'],
            'invalid_samples': df['MIMS_INVALID_SAMPLE_NUM'],
            'wear_activity': df['MIMS_SUM_WEAR'],
            'sleep_activity': df['MIMS_SUM_SLEEP'],
            'nonwear_activity': df['MIMS_SUM_NONWEAR'],
            'wear_minutes': df['MIMS_SAMPLE_NUM_WEAR'],
            'sleep_minutes': df['MIMS_SAMPLE_NUM_SLEEP'],
            'nonwear_minutes': df['MIMS_SAMPLE_NUM_NONWEAR']
        }

        self.activity_data = pd.DataFrame(activity_features)
        self.activity_data.set_index('timestamp', inplace=True)
        return self.activity_data

    def load_phone_usage_data(self, app_usage_file: str, screen_events_file: str = None) -> pd.DataFrame:
        """
        Load and process phone usage data.
        """
        # Process app usage data
        app_df = pd.read_csv(app_usage_file)

        # Extract datetime without timezone
        def clean_timestamp(ts):
            try:
                # Split on space and take everything except the timezone
                return pd.to_datetime(' '.join(str(ts).split()[:-1]))
            except (ValueError, TypeError):
                return pd.NaT

        # Clean timestamps
        app_df['START_TIME'] = app_df['START_TIME'].apply(clean_timestamp)
        app_df['STOP_TIME'] = app_df['STOP_TIME'].apply(clean_timestamp)

        # Create hourly app usage features
        hourly_usage = self._process_app_usage(app_df)

        if screen_events_file:
            events_df = pd.read_csv(screen_events_file, low_memory=False)
            events_df['LOG_TIME'] = events_df['LOG_TIME'].apply(clean_timestamp)
            screen_events = self._process_screen_events(events_df)

            # Merge app usage and screen events
            self.phone_usage_data = pd.merge(
                hourly_usage,
                screen_events,
                left_index=True,
                right_index=True,
                how='outer'
            )
        else:
            self.phone_usage_data = hourly_usage

        return self.phone_usage_data

    def _process_app_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process app usage data."""
        df['hour'] = df['START_TIME'].dt.floor('h')

        hourly_metrics = df.groupby('hour').agg({
            'USAGE_DURATION_MIN': 'sum',
            'APP_PACKAGE_NAME': 'nunique'
        }).rename(columns={
            'USAGE_DURATION_MIN': 'total_usage_minutes',
            'APP_PACKAGE_NAME': 'unique_apps_used'
        })

        return hourly_metrics

    def _process_screen_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process screen events data."""
        df['hour'] = df['LOG_TIME'].dt.floor('h')

        screen_metrics = pd.DataFrame()

        screen_on = df[df['PHONE_EVENT'] == 'PHONE_SCREEN_ON'].groupby('hour').size()
        screen_off = df[df['PHONE_EVENT'] == 'PHONE_SCREEN_OFF'].groupby('hour').size()
        unlocked = df[df['PHONE_EVENT'] == 'PHONE_UNLOCKED'].groupby('hour').size()

        screen_metrics['screen_on_count'] = screen_on
        screen_metrics['screen_off_count'] = screen_off
        screen_metrics['unlock_count'] = unlocked

        return screen_metrics


class StressAnalyzer(TimeStudyProcessor):
    """Analyzer for stress and anxiety patterns in TIME study data."""

    def __init__(self, participant_id: str):
        """Initialize the stress analyzer.

        Args:
            participant_id: Unique identifier for the participant
        """
        super().__init__(participant_id)
        # These are the actual TIME study questions related to stress/anxiety
        self.stress_questions = {
            'stress': ['How stressed are you feeling right now?'],
            'anxiety': ['How anxious are you feeling right now?'],
            'restlessness': ['How restless are you right now?'],
            'nervous': ['How nervous are you right now?'],
            'worry': ['How worried are you right now?']
        }

    def process_stress_responses(self) -> pd.DataFrame:
        """Extract and process stress/anxiety related responses from EMA data.

        Returns:
            DataFrame containing processed stress responses
        """
        if self.ema_data is None:
            raise ValueError("Please load EMA data first")

        stress_responses = []

        # Get the timestamp from the index since we set it during load_ema_data
        for idx, row in self.ema_data.iterrows():
            # Initialize response with the current timestamp
            response = {}
            found_stress_question = False

            # Look through all questions in this EMA
            for i in range(1, 50):  # Based on codebook structure
                q_text = row.get(f'Question_{i}_Text')
                q_answer = row.get(f'Question_{i}_Answer_Text ')  # Note the space after Text

                if pd.notna(q_text) and pd.notna(q_answer):
                    # Match question to stress/anxiety categories
                    for category, patterns in self.stress_questions.items():
                        if any(pattern.lower() in str(q_text).lower() for pattern in patterns):
                            normalized_answer = self._normalize_response(q_answer)
                            if not pd.isna(normalized_answer):
                                response[category] = normalized_answer
                                found_stress_question = True

            # Only add responses that have at least one stress measure
            if found_stress_question:
                stress_responses.append({
                    'timestamp': idx,  # Add timestamp from the index
                    **response        # Unpack the stress responses
                })

        if not stress_responses:
            raise ValueError("No stress/anxiety responses found in the data")

        # Create DataFrame and set index
        stress_df = pd.DataFrame(stress_responses)
        return stress_df.set_index('timestamp')

    def _merge_stress_with_features(
        self,
        stress_data: pd.DataFrame,
        features: pd.DataFrame,
        window: str = '1H'
    ) -> pd.DataFrame:
        """Merge stress measurements with feature data.

        Args:
            stress_data: DataFrame containing stress measurements
            features: DataFrame containing feature data
            window: Time window for aggregating features

        Returns:
            Merged DataFrame
        """
        # Resample features to hourly data
        features_hourly = features.resample(window).mean()

        # For each stress measure, find the corresponding feature values
        merged_data = []

        for stress_idx in stress_data.index:
            # Get feature values within the window before the stress measurement
            window_start = stress_idx - pd.Timedelta(window)
            window_features = features_hourly.loc[window_start:stress_idx].mean()

            if not window_features.empty:
                row_data = {
                    'timestamp': stress_idx,
                    **stress_data.loc[stress_idx].to_dict(),
                    **window_features.to_dict()
                }
                merged_data.append(row_data)

        return pd.DataFrame(merged_data)

    def create_feature_matrix(self, window_size: str = '1H') -> pd.DataFrame:
        """Create combined feature matrix.

        Args:
            window_size: Time window for feature aggregation

        Returns:
            Feature matrix DataFrame
        """
        if any(data is None for data in [self.activity_data, self.phone_usage_data]):
            raise ValueError("Please load all required data first")

        # Create base feature matrix using activity data timepoints
        feature_matrix = self.activity_data.copy()

        # Add phone usage features
        feature_matrix = feature_matrix.join(self.phone_usage_data)

        # Create rolling window features
        feature_matrix['activity_1h_avg'] = feature_matrix['total_activity'].rolling(window='1h').mean()
        feature_matrix['activity_3h_avg'] = feature_matrix['total_activity'].rolling(window='3h').mean()
        feature_matrix['phone_usage_1h_avg'] = feature_matrix['total_usage_minutes'].rolling(window='1h').mean()
        feature_matrix['phone_usage_3h_avg'] = feature_matrix['total_usage_minutes'].rolling(window='3h').mean()

        # Add time-based features
        feature_matrix['hour'] = feature_matrix.index.hour
        feature_matrix['day_of_week'] = feature_matrix.index.dayofweek
        feature_matrix['is_weekend'] = feature_matrix['day_of_week'].isin([5, 6]).astype(int)

        return feature_matrix

    def _calculate_feature_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations between stress/anxiety and all features.

        Args:
            data: Merged DataFrame containing stress measures and features

        Returns:
            DataFrame containing correlation results
        """
        stress_measures = list(self.stress_questions.keys())
        feature_cols = [col for col in data.columns
                        if col not in stress_measures + ['timestamp']]

        correlations = []

        for stress_measure in stress_measures:
            if stress_measure not in data.columns:
                continue

            for feature in feature_cols:
                # Remove any missing values
                valid_data = data[[stress_measure, feature]].dropna()

                if len(valid_data) < 2:
                    continue

                # Calculate Pearson correlation
                correlation = stats.pearsonr(valid_data[stress_measure],
                                             valid_data[feature])

                correlations.append({
                    'stress_measure': stress_measure,
                    'feature': feature,
                    'correlation': correlation[0],
                    'p_value': correlation[1],
                    'n_samples': len(valid_data)
                })

        return pd.DataFrame(correlations)

    def _normalize_response(self, response: str) -> float:
        """Normalize different response formats to a 0-100 scale.

        Args:
            response: Raw response string from TIME study

        Returns:
            Normalized response value (0-100)
        """
        try:
            # TIME study uses numeric responses 0-100
            if isinstance(response, (int, float)):
                return float(response)
            elif isinstance(response, str):
                # Extract numeric value if present
                numeric_str = ''.join(c for c in response if c.isdigit() or c == '.')
                if numeric_str:
                    value = float(numeric_str)
                    # Ensure value is within 0-100 range
                    return max(0, min(100, value))

            return np.nan

        except (ValueError, TypeError):
            return np.nan

    def analyze_correlations(
        self,
        feature_window: str = '1h'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze correlations between stress/anxiety and features.

        Args:
            feature_window: Time window for feature aggregation

        Returns:
            Tuple of (correlations DataFrame, merged data DataFrame)
        """
        stress_data = self.process_stress_responses()
        features = self.create_feature_matrix(window_size=feature_window)
        merged_data = self._merge_stress_with_features(
            stress_data,
            features,
            window=feature_window
        )
        correlations = self._calculate_feature_correlations(merged_data)
        return correlations, merged_data

    def plot_correlations(
        self,
        correlations: pd.DataFrame,
        n_top: int = 10
    ) -> None:
        """Plot correlation results.

        Args:
            correlations: Correlation results DataFrame
            n_top: Number of top correlations to plot
        """
        for stress_measure in self.stress_questions.keys():
            measure_correlations = correlations[
                correlations['stress_measure'] == stress_measure
            ].sort_values('correlation', key=abs, ascending=False).head(n_top)

            if not measure_correlations.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=measure_correlations,
                    y='feature',
                    x='correlation',
                    palette='RdBu_r'
                )
                plt.title(f'Top {n_top} Correlations with {stress_measure}')
                plt.xlabel('Correlation Coefficient')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.show()


def main():
    """Main execution function."""
    # Set up logging
    logging.info("Starting TIME study analysis")

    # Initialize analyzer
    participant_id = "your_participant_id"
    analyzer = StressAnalyzer(participant_id)

    try:
        # Load data
        logging.info("Loading data files...")
        analyzer.load_ema_data("phone_promptresponse_clean.csv")
        analyzer.load_activity_data("watch_accelerometer_mims_hour.csv")
        analyzer.load_phone_usage_data(
            app_usage_file="phone_app_usage_duration.csv",
            screen_events_file="phone_system_events_clean.csv"
        )

        # Analyze correlations
        logging.info("Analyzing correlations...")
        correlations, merged_data = analyzer.analyze_correlations()

        # Print results
        logging.info("\nTop correlations with stress/anxiety:")
        for measure in analyzer.stress_questions.keys():
            print(f"\nTop 5 correlations with {measure}:")
            measure_corr = correlations[
                correlations['stress_measure'] == measure
            ]
            print(measure_corr.sort_values(
                'correlation',
                key=abs,
                ascending=False
            ).head())

        # Plot results
        logging.info("\nPlotting correlation results...")
        analyzer.plot_correlations(correlations)

        return correlations, merged_data

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    correlations, merged_data = main()
