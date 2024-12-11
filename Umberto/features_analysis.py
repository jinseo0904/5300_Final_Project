import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os


def load_all_subjects(data_folder):
    """
    Load data for all subjects from the data folder
    """
    all_files = glob.glob(os.path.join(data_folder, "*_daily_metrics.csv"))

    dfs = []
    for file_path in all_files:
        subject_id = os.path.basename(file_path).split('_daily_metrics.csv')[0]
        df = pd.read_csv(file_path)
        df['id'] = subject_id
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded data for {len(dfs)} subjects")

    return combined_df


def load_and_preprocess_data(df):
    """
    Preprocess the data with normalization for both phone and watch stress measures
    """
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Phone stress normalization
    stress_mean = df.groupby('id')['Q7_STRESS'].transform('mean')
    # Fix the chained assignment warning
    df.loc[:, 'Q7_STRESS'] = df['Q7_STRESS'].fillna(stress_mean)
    df.loc[:, 'stress_normalized'] = df.groupby(
        'id')['Q7_STRESS'].transform(lambda x: (x - x.mean()) / x.std())

    # Watch stress normalization (if available)
    if 'stress_2' in df.columns:
        watch_stress_mean = df.groupby('id')['stress_2'].transform('mean')
        df.loc[:, 'stress_2'] = df['stress_2'].fillna(watch_stress_mean)
        df.loc[:, 'watch_stress_normalized'] = df.groupby(
            'id')['stress_2'].transform(lambda x: (x - x.mean()) / x.std())

    df.loc[:, 'stress_high'] = df['stress_normalized'] > 0

    return df


def analyze_stress_by_subject(df):
    """
    Analyze stress statistics for each subject
    """
    stress_measures = [
        'Q7_STRESS', 'stress_2'] if 'stress_2' in df.columns else ['Q7_STRESS']

    subject_stats = df.groupby('id').agg({
        **{measure: ['mean', 'std', 'count'] for measure in stress_measures},
        'stress_normalized': ['mean', 'std']
    }).round(3)

    return subject_stats


def analyze_feature_correlations(df):
    """
    Analyze correlations with normalized stress scores
    """
    # Phone EMA features
    phone_features = [
        'stress_normalized', 'Q1_SAD', 'Q2_HAPP', 'Q3_FATIG', 'Q4_EN',
        'Q5_REL', 'Q6_TEN', 'Q7_STRESS', 'Q8_FRUST', 'Q9_NERV'
    ]

    # Watch EMA features
    watch_features = [
        'excite_14', 'focus_3', 'frust_12', 'happy_8', 'int_exer_3',
        'nervous_4', 'proc_4', 'relax_10', 'sad_7', 'stress_2', 'tense_11'
    ]

    # Behavioral/sensor features
    behavioral_features = [
        'SLEEP_MINUTES', 'SCREEN_ON_SECONDS', 'UNLOCK_EVENTS_NUM',
        'MIMS_SUM_WEAR', 'USAGE_DURATION_MIN',
        'IN_VEHICLE', 'ON_BIKE', 'ON_FOOT', 'RUNNING', 'STILL', 'TILTING',
        'WALKING', 'UNKNOWN'
    ]

    # Combine all features
    all_features = phone_features + watch_features + behavioral_features

    # Get available features that exist in the dataframe
    available_features = [f for f in all_features if f in df.columns]

    # Remove features with constant values (all zeros or all same value)
    non_constant_features = []
    constant_features = []
    for feature in available_features:
        if df[feature].nunique() > 1:
            non_constant_features.append(feature)
        else:
            constant_features.append(feature)

    if constant_features:
        print("\nFeatures with constant values (excluded from correlation analysis):")
        for feature in constant_features:
            print(f"- {feature}: all values = {df[feature].iloc[0]}")

    # Calculate correlations only for non-constant features
    corr_matrix = df[non_constant_features].corr(
    )['stress_normalized'].sort_values(ascending=False)

    # Group correlations by feature type
    grouped_correlations = {
        'Phone EMA': corr_matrix[corr_matrix.index.isin(phone_features)],
        'Watch EMA': corr_matrix[corr_matrix.index.isin(watch_features)],
        'Behavioral': corr_matrix[corr_matrix.index.isin(behavioral_features)]
    }

    return corr_matrix, grouped_correlations


def plot_normalized_distributions(df):
    """
    Plot original and normalized stress distributions from both phone and watch
    """
    n_subjects = df['id'].nunique()
    fig_height = max(6, n_subjects * 0.5)

    n_plots = 3 if 'watch_stress_normalized' in df.columns else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(15, fig_height))

    # Plot phone stress scores
    sns.boxplot(x='id', y='Q7_STRESS', data=df, ax=axes[0])
    axes[0].set_title('Original Phone Stress Scores by Subject')
    # Fix ticklabels warning
    ticks = axes[0].get_xticks()
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Plot normalized phone stress
    sns.boxplot(x='id', y='stress_normalized', data=df, ax=axes[1])
    axes[1].set_title('Normalized Phone Stress Scores by Subject')
    # Fix ticklabels warning
    ticks = axes[1].get_xticks()
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    # Plot watch stress if available
    if 'watch_stress_normalized' in df.columns:
        sns.boxplot(x='id', y='watch_stress_normalized', data=df, ax=axes[2])
        axes[2].set_title('Normalized Watch Stress Scores by Subject')
        # Fix ticklabels warning
        ticks = axes[2].get_xticks()
        axes[2].set_xticks(ticks)
        axes[2].set_xticklabels(
            axes[2].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_correlation_heatmaps(df, output_dir):
    """
    Create separate heatmaps for each feature group
    """
    # Define feature groups
    phone_features = [
        'Q1_SAD', 'Q2_HAPP', 'Q3_FATIG', 'Q4_EN',
        'Q5_REL', 'Q6_TEN', 'Q7_STRESS', 'Q8_FRUST', 'Q9_NERV'
    ]

    watch_features = [
        'excite_14', 'focus_3', 'frust_12', 'happy_8', 'int_exer_3',
        'nervous_4', 'proc_4', 'relax_10', 'sad_7', 'stress_2', 'tense_11'
    ]

    behavioral_features = [
        'SLEEP_MINUTES', 'SCREEN_ON_SECONDS', 'UNLOCK_EVENTS_NUM',
        'MIMS_SUM_WEAR', 'USAGE_DURATION_MIN',
        'IN_VEHICLE', 'ON_BIKE', 'ON_FOOT', 'RUNNING', 'STILL', 'TILTING',
        'WALKING', 'UNKNOWN'
    ]

    feature_groups = {
        'phone_ema': phone_features,
        'watch_ema': watch_features,
        'behavioral': behavioral_features
    }

    for group_name, features in feature_groups.items():
        # Get available features and remove constant features
        available_features = []
        constant_features = []

        for feature in features:
            if feature in df.columns:
                if df[feature].nunique() > 1:
                    available_features.append(feature)
                else:
                    constant_features.append(feature)

        if available_features:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[available_features].corr(),
                        annot=True,
                        cmap='coolwarm',
                        center=0,
                        fmt='.2f')

            title = f'{group_name.replace("_", " ").title()} Feature Correlations'
            if constant_features:
                title += f'\n(Excluded constant features: {", ".join(constant_features)})'
            plt.title(title)

            plt.tight_layout()
            plt.savefig(output_dir / f'correlation_heatmap_{group_name}.png')
            plt.close()


def plot_stress_time_series_by_subject(df):
    """
    Plot normalized stress levels over time for each subject
    """
    subjects = df['id'].unique()
    n_subjects = len(subjects)
    n_cols = min(2, n_subjects)
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_subjects == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ax, subject) in enumerate(zip(axes, subjects)):
        subject_data = df[df['id'] == subject]

        # Plot phone stress
        ax.plot(subject_data['date'], subject_data['stress_normalized'],
                'b-', alpha=0.5, label='Phone Stress')

        # Plot watch stress if available
        if 'watch_stress_normalized' in df.columns:
            ax.plot(subject_data['date'], subject_data['watch_stress_normalized'],
                    'r-', alpha=0.5, label='Watch Stress')

        ax.set_title(f'Subject: {subject}')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def analyze_temporal_patterns(df):
    """
    Analyze temporal patterns in stress levels
    """
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

    stress_measures = ['stress_normalized']
    if 'watch_stress_normalized' in df.columns:
        stress_measures.append('watch_stress_normalized')

    patterns = {}
    for measure in stress_measures:
        patterns[measure] = {
            'weekday_vs_weekend': df.groupby('is_weekend')[measure].mean(),
            'day_of_week': df.groupby('day_of_week')[measure].mean(),
            'monthly': df.groupby('month')[measure].mean()
        }

    return patterns


def identify_stress_predictors(df):
    """
    Identify potential predictors of high stress including watch features
    """
    # Define lag features including watch EMA
    phone_lag_features = ['Q7_STRESS', 'Q2_HAPP', 'Q5_REL']
    watch_lag_features = ['happy_8', 'relax_10', 'stress_2']
    sensor_features = ['SLEEP_MINUTES', 'SCREEN_ON_SECONDS',
                       'UNLOCK_EVENTS_NUM', 'MIMS_SUM_WEAR']

    lag_features = phone_lag_features + watch_lag_features + sensor_features

    # Create lagged features
    for feature in lag_features:
        if feature in df.columns:
            df[f'prev_day_{feature}'] = df.groupby('id')[feature].shift(1)
            df[f'roll_3day_{feature}'] = df.groupby('id')[feature].rolling(
                window=3, min_periods=1).mean().reset_index(0, drop=True)
            df[f'roll_7day_{feature}'] = df.groupby('id')[feature].rolling(
                window=7, min_periods=1).mean().reset_index(0, drop=True)

    # Add activity features
    activity_cols = ['IN_VEHICLE', 'ON_BIKE', 'ON_FOOT',
                     'RUNNING', 'STILL', 'TILTING', 'WALKING']
    if all(col in df.columns for col in activity_cols):
        df['total_activity'] = df[activity_cols].sum(axis=1)
        df['prev_day_activity'] = df.groupby('id')['total_activity'].shift(1)
        df['roll_3day_activity'] = df.groupby('id')['total_activity'].rolling(
            window=3, min_periods=1).mean().reset_index(0, drop=True)

    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Group predictors
    predictor_groups = {
        'Previous Day EMA': [col for col in df.columns if 'prev_day_' in col and any(f in col for f in phone_lag_features + watch_lag_features)],
        'Rolling Average EMA': [col for col in df.columns if 'roll_' in col and any(f in col for f in phone_lag_features + watch_lag_features)],
        'Previous Day Sensors': [col for col in df.columns if 'prev_day_' in col and any(f in col for f in sensor_features)],
        'Rolling Average Sensors': [col for col in df.columns if 'roll_' in col and any(f in col for f in sensor_features)],
        'Activity': [col for col in df.columns if 'activity' in col],
        'Time Features': ['is_weekend']
    }

    # Calculate correlations for each group
    grouped_correlations = {}
    for group_name, group_cols in predictor_groups.items():
        available_cols = [col for col in group_cols if col in df.columns]
        if available_cols:
            correlations = df[available_cols + ['stress_normalized']
                              ].corr()['stress_normalized'][:-1]
            grouped_correlations[group_name] = correlations.sort_values(
                ascending=False)

    feature_importance = pd.concat(
        [corrs for corrs in grouped_correlations.values()])
    feature_importance = feature_importance.abs().sort_values(ascending=False)

    results = {
        'grouped_correlations': grouped_correlations,
        'feature_importance': feature_importance,
        'top_predictors': feature_importance.head(10)
    }

    return results, df


def plot_predictor_importance(results):
    """
    Plot the importance of different predictors
    """
    plt.figure(figsize=(12, 6))
    top_predictors = results['top_predictors']
    sns.barplot(x=top_predictors.values, y=top_predictors.index)
    plt.title('Top 10 Stress Predictors (Absolute Correlation)')
    plt.xlabel('|Correlation|')
    plt.tight_layout()

    return plt


def analyze_predictor_relationships(df, top_predictors):
    """
    Create detailed analysis of top predictor relationships
    """
    relationships = {}

    for predictor in top_predictors.index[:5]:
        if predictor in df.columns:
            stats = {
                'correlation': df[predictor].corr(df['stress_normalized']),
                'mean_high_stress': df[df['stress_high']][predictor].mean(),
                'mean_low_stress': df[~df['stress_high']][predictor].mean(),
                'std_high_stress': df[df['stress_high']][predictor].std(),
                'std_low_stress': df[~df['stress_high']][predictor].std()
            }
            relationships[predictor] = stats

    return relationships


def main():
    """
    Main function to run all analyses
    """
    output_dir = Path('stress_analysis_output')
    output_dir.mkdir(exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_all_subjects('Umberto/data')
    processed_df = load_and_preprocess_data(df)
    processed_df.to_csv(output_dir / 'processed_data.csv', index=False)

    print("\n1. Analyzing stress distributions...")
    subject_stats = analyze_stress_by_subject(processed_df)
    print("\nStress Statistics by Subject:")
    print(subject_stats)
    subject_stats.to_csv(output_dir / 'subject_statistics.csv')

    plot_normalized_distributions(processed_df)
    plt.savefig(output_dir / 'stress_distributions.png')
    plt.close()

    print("\n2. Analyzing temporal patterns...")
    temporal_patterns = analyze_temporal_patterns(processed_df)
    for measure, patterns in temporal_patterns.items():
        print(f"\nPatterns for {measure}:")
        for pattern_name, pattern_data in patterns.items():
            print(f"\n{pattern_name}:")
            print(pattern_data)
            pattern_data.to_frame().to_csv(
                output_dir / f'temporal_pattern_{measure}_{pattern_name}.csv')

    plot_stress_time_series_by_subject(processed_df)
    plt.savefig(output_dir / 'stress_time_series.png')
    plt.close()

    print("\n3. Analyzing feature correlations...")
    corr_matrix, grouped_correlations = analyze_feature_correlations(
        processed_df)
    print("\nFeature Correlations with Stress by Group:")
    for group, correlations in grouped_correlations.items():
        print(f"\n{group}:")
        print(correlations)

    pd.DataFrame(grouped_correlations).to_csv(
        output_dir / 'feature_correlations_by_group.csv')

    plot_correlation_heatmaps(processed_df, output_dir)

    print("\n4. Analyzing predictive relationships...")
    predictor_results, processed_df = identify_stress_predictors(processed_df)
    print("\nTop Predictors:")
    print(predictor_results['top_predictors'])
    predictor_results['top_predictors'].to_frame().to_csv(
        output_dir / 'top_predictors.csv')

    plot_predictor_importance(predictor_results)
    plt.savefig(output_dir / 'predictor_importance.png')
    plt.close()

    relationships = analyze_predictor_relationships(
        processed_df, predictor_results['top_predictors'])
    print("\nPredictor Relationships:")
    print(relationships)
    pd.DataFrame(relationships).to_csv(
        output_dir / 'predictor_relationships.csv')

    print("\n5. Analyzing activity patterns...")
    if 'total_activity' in processed_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=processed_df, x='total_activity',
                        y='stress_normalized', hue='id')
        plt.title('Activity Level vs Normalized Stress')
        plt.savefig(output_dir / 'activity_vs_stress.png')
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=processed_df, x='day_of_week', y='stress_normalized')
    plt.title('Stress Levels by Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'daily_patterns.png')
    plt.close()

    if 'SLEEP_MINUTES' in processed_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=processed_df, x='SLEEP_MINUTES',
                        y='stress_normalized', hue='id')
        plt.title('Sleep Duration vs Normalized Stress')
        plt.savefig(output_dir / 'sleep_vs_stress.png')
        plt.close()

    # Additional plots for watch-specific features
    watch_specific_features = ['MIMS_SUM_WEAR']
    for feature in watch_specific_features:
        if feature in processed_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=processed_df, x=feature,
                            y='stress_normalized', hue='id')
            plt.title(f'{feature} vs Normalized Stress')
            plt.savefig(output_dir / f'{feature.lower()}_vs_stress.png')
            plt.close()

    print(
        f"\nAnalysis complete! Visualizations have been saved to {output_dir}/")

    return processed_df, subject_stats, grouped_correlations, temporal_patterns, predictor_results, relationships


if __name__ == "__main__":
    processed_df, subject_stats, grouped_correlations, temporal_patterns, predictor_results, relationships = main()
