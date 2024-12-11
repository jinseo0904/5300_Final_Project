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
        'Q7_STRESS',
        'stress_2'] if 'stress_2' in df.columns else ['Q7_STRESS']

    subject_stats = df.groupby('id').agg({
        **{measure: ['mean', 'std', 'count'] for measure in stress_measures},
        'stress_normalized': ['mean', 'std']
    }).round(3)

    return subject_stats


def analyze_all_features_with_stress(df):
    """
    Comprehensive analysis of all features' relationships with stress
    """
    # All features to analyze (excluding metadata columns, stress measures,
    # and derived columns)
    exclude_cols = [
        'id', 'date',  # Metadata
        # Stress measures and derived
        'Q7_STRESS', 'stress_2', 'stress_normalized', 'watch_stress_normalized', 'stress_high',
        'day_of_week', 'month', 'is_weekend'  # Temporal columns
    ]
    features = [col for col in df.columns if col not in exclude_cols]

    # Calculate correlations with stress for all features
    correlations = {}
    feature_stats = {}

    for feature in features:
        # Check if feature is numeric
        if pd.api.types.is_numeric_dtype(df[feature]):
            if df[feature].nunique() > 1:  # Skip constant features
                # Calculate correlation
                correlation = df[feature].corr(df['stress_normalized'])
                correlations[feature] = correlation

                # Calculate basic statistics
                feature_stats[feature] = {
                    'correlation': correlation,
                    'mean_high_stress': df[df['stress_high']][feature].mean(),
                    'mean_low_stress': df[~df['stress_high']][feature].mean(),
                    'std_high_stress': df[df['stress_high']][feature].std(),
                    'std_low_stress': df[~df['stress_high']][feature].std(),
                    'effect_size': (df[df['stress_high']][feature].mean() -
                                    df[~df['stress_high']][feature].mean()) / df[feature].std()
                }

    # Create sorted correlation series
    correlation_series = pd.Series(correlations).sort_values(ascending=False)

    # Convert feature stats to DataFrame
    stats_df = pd.DataFrame(feature_stats).T

    return correlation_series, stats_df


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
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Plot normalized phone stress
    sns.boxplot(x='id', y='stress_normalized', data=df, ax=axes[1])
    axes[1].set_title('Normalized Phone Stress Scores by Subject')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    # Plot watch stress if available
    if 'watch_stress_normalized' in df.columns:
        sns.boxplot(x='id', y='watch_stress_normalized', data=df, ax=axes[2])
        axes[2].set_title('Normalized Watch Stress Scores by Subject')
        axes[2].set_xticklabels(
            axes[2].get_xticklabels(),
            rotation=45,
            ha='right')

    plt.tight_layout()
    return fig


def plot_comprehensive_correlations(df, correlation_series, output_dir):
    """
    Create comprehensive correlation visualizations
    """
    # 1. Overall correlation plot
    plt.figure(figsize=(15, 8))
    correlation_series.plot(kind='bar')
    plt.title('Correlation of Features with Normalized Stress')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_features_correlation.png')
    plt.close()

    # 2. Top 10 positive and negative correlations
    plt.figure(figsize=(12, 6))
    top_correlations = pd.concat([
        correlation_series.head(10),
        correlation_series.tail(10)
    ])
    top_correlations.plot(kind='bar')
    plt.title('Top 10 Positive and Negative Correlations with Normalized Stress')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_correlations.png')
    plt.close()

    # 3. Complete correlation heatmap including stress measures
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude derived columns and metadata
    exclude_cols = ['stress_high', 'date']
    cols_for_heatmap = [col for col in numeric_cols if col not in exclude_cols]

    # Calculate correlation matrix
    corr_matrix = df[cols_for_heatmap].corr()

    # Create heatmap
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix), k=0)
    sns.heatmap(corr_matrix,
                mask=mask,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.title('Complete Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'complete_correlation_heatmap.png')
    plt.close()


def plot_feature_distributions(df, stats_df, output_dir):
    """
    Create distribution plots for top correlated features
    """
    # Get top 10 features by absolute correlation
    top_features = stats_df.nlargest(10, 'correlation').index

    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df[df['stress_high']], x=feature, label='High Stress')
        sns.kdeplot(data=df[~df['stress_high']], x=feature, label='Low Stress')
        plt.title(f'{feature} Distribution by Stress Level')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'{feature}_distribution.png')
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


def create_feature_relationship_report(correlation_series, stats_df):
    """
    Create a detailed report of feature relationships with stress
    """
    report = []

    # Overall summary
    report.append("Feature Relationship Analysis Report")
    report.append("=" * 40)

    # Top positive correlations
    report.append("\nTop 5 Positive Correlations with Stress:")
    for feature in correlation_series.head().index:
        stats = stats_df.loc[feature]
        report.append(f"\n{feature}:")
        report.append(f"- Correlation: {stats['correlation']:.3f}")
        report.append(f"- Mean (High Stress): {stats['mean_high_stress']:.3f}")
        report.append(f"- Mean (Low Stress): {stats['mean_low_stress']:.3f}")
        report.append(f"- Effect Size: {stats['effect_size']:.3f}")

    # Top negative correlations
    report.append("\nTop 5 Negative Correlations with Stress:")
    for feature in correlation_series.tail().index:
        stats = stats_df.loc[feature]
        report.append(f"\n{feature}:")
        report.append(f"- Correlation: {stats['correlation']:.3f}")
        report.append(f"- Mean (High Stress): {stats['mean_high_stress']:.3f}")
        report.append(f"- Mean (Low Stress): {stats['mean_low_stress']:.3f}")
        report.append(f"- Effect Size: {stats['effect_size']:.3f}")

    return "\n".join(report)


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
            pattern_data.to_frame().to_csv(output_dir /
                                           f'temporal_pattern_{measure}_{pattern_name}.csv')

    plot_stress_time_series_by_subject(processed_df)
    plt.savefig(output_dir / 'stress_time_series.png')
    plt.close()

    print("\n3. Analyzing feature relationships...")
    correlation_series, stats_df = analyze_all_features_with_stress(
        processed_df)

    # Save correlation results
    correlation_series.to_csv(output_dir / 'all_feature_correlations.csv')
    stats_df.to_csv(output_dir / 'feature_statistics.csv')

    # Create visualizations
    plot_comprehensive_correlations(
        processed_df, correlation_series, output_dir)
    plot_feature_distributions(processed_df, stats_df, output_dir)

    # Generate and save report
    report = create_feature_relationship_report(correlation_series, stats_df)
    with open(output_dir / 'feature_analysis_report.txt', 'w') as f:
        f.write(report)

    print("\n4. Analyzing daily patterns...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=processed_df, x='day_of_week', y='stress_normalized')
    plt.title('Stress Levels by Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'daily_patterns.png')
    plt.close()

    if 'SLEEP_MINUTES' in processed_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=processed_df,
            x='SLEEP_MINUTES',
            y='stress_normalized',
            hue='id')
        plt.title('Sleep Duration vs Normalized Stress')
        plt.savefig(output_dir / 'sleep_vs_stress.png')
        plt.close()

    # Additional plots for watch-specific features
    watch_specific_features = ['MIMS_SUM_WEAR']
    for feature in watch_specific_features:
        if feature in processed_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=processed_df,
                x=feature,
                y='stress_normalized',
                hue='id')
            plt.title(f'{feature} vs Normalized Stress')
            plt.savefig(output_dir / f'{feature.lower()}_vs_stress.png')
            plt.close()

    print(
        f"\nAnalysis complete! Visualizations have been saved to {output_dir}/")

    return processed_df, subject_stats, correlation_series, stats_df, temporal_patterns, report


if __name__ == "__main__":
    processed_df, subject_stats, correlation_series, stats_df, temporal_patterns, report = main()
