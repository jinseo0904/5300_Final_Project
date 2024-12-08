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
    Preprocess the data with normalization
    """
    df['date'] = pd.to_datetime(df['date'])
    
    stress_mean = df.groupby('id')['Q7_STRESS'].transform('mean')
    df['Q7_STRESS'].fillna(stress_mean, inplace=True)
    
    df['stress_normalized'] = df.groupby('id')['Q7_STRESS'].transform(lambda x: (x - x.mean()) / x.std())
    df['stress_high'] = df['stress_normalized'] > 0
    
    return df

def analyze_stress_by_subject(df):
    """
    Analyze stress statistics for each subject
    """
    subject_stats = df.groupby('id').agg({
        'Q7_STRESS': ['mean', 'std', 'count'],
        'stress_normalized': ['mean', 'std']
    }).round(3)
    
    return subject_stats

def plot_normalized_distributions(df):
    """
    Plot original and normalized stress distributions
    """
    n_subjects = df['id'].nunique()
    fig_height = max(6, n_subjects * 0.5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height))
    
    sns.boxplot(x='id', y='Q7_STRESS', data=df, ax=ax1)
    ax1.set_title('Original Stress Scores by Subject')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    sns.boxplot(x='id', y='stress_normalized', data=df, ax=ax2)
    ax2.set_title('Normalized Stress Scores by Subject')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def analyze_feature_correlations(df):
    """
    Analyze correlations with normalized stress scores
    """
    features = [
        'stress_normalized', 'Q1_SAD', 'Q2_HAPP', 'Q3_FATIG', 'Q4_EN', 
        'Q5_REL', 'Q6_TEN', 'Q8_FRUST', 'Q9_NERV',
        'SLEEP_MINUTES', 'SCREEN_ON_SECONDS', 'UNLOCK_EVENTS_NUM'
    ]
    
    available_features = [f for f in features if f in df.columns]
    corr_matrix = df[available_features].corr()['stress_normalized'].sort_values(ascending=False)
    
    return corr_matrix

def plot_stress_time_series_by_subject(df):
    """
    Plot normalized stress levels over time for each subject
    """
    subjects = df['id'].unique()
    n_subjects = len(subjects)
    n_cols = min(2, n_subjects)
    n_rows = (n_subjects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_subjects == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (ax, subject) in enumerate(zip(axes, subjects)):
        subject_data = df[df['id'] == subject]
        ax.plot(subject_data['date'], subject_data['stress_normalized'], 'b-', alpha=0.5)
        ax.set_title(f'Subject: {subject}')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
    
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
    
    patterns = {
        'weekday_vs_weekend': df.groupby('is_weekend')['stress_normalized'].mean(),
        'day_of_week': df.groupby('day_of_week')['stress_normalized'].mean(),
        'monthly': df.groupby('month')['stress_normalized'].mean()
    }
    
    return patterns

def identify_stress_predictors(df):
    """
    Identify potential predictors of high stress
    """
    lag_features = ['Q7_STRESS', 'SLEEP_MINUTES', 'SCREEN_ON_SECONDS', 
                   'UNLOCK_EVENTS_NUM', 'Q2_HAPP', 'Q5_REL']
    
    for feature in lag_features:
        if feature in df.columns:
            df[f'prev_day_{feature}'] = df.groupby('id')[feature].shift(1)
            df[f'roll_3day_{feature}'] = df.groupby('id')[feature].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
            df[f'roll_7day_{feature}'] = df.groupby('id')[feature].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    
    activity_cols = ['IN_VEHICLE', 'ON_BIKE', 'ON_FOOT', 'RUNNING', 'STILL', 'TILTING', 'WALKING']
    if all(col in df.columns for col in activity_cols):
        df['total_activity'] = df[activity_cols].sum(axis=1)
        df['prev_day_activity'] = df.groupby('id')['total_activity'].shift(1)
        df['roll_3day_activity'] = df.groupby('id')['total_activity'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    stress_cols = [col for col in df.columns if 'stress' in col.lower()]
    potential_predictors = [col for col in df.columns if any(x in col for x in 
                          ['prev_day_', 'roll_', 'activity', 'SLEEP', 'SCREEN', 'is_weekend'])]
    
    correlations = df[potential_predictors + stress_cols].corr()['stress_normalized']
    
    predictor_groups = {
        'Previous Day Metrics': [col for col in potential_predictors if 'prev_day_' in col],
        'Rolling Averages': [col for col in potential_predictors if 'roll_' in col],
        'Activity Metrics': [col for col in potential_predictors if 'activity' in col],
        'Time Features': ['is_weekend'],
        'Current Day Metrics': [col for col in potential_predictors if not any(x in col for x in 
                              ['prev_day_', 'roll_', 'activity', 'is_weekend'])]
    }
    
    grouped_correlations = {}
    for group_name, group_cols in predictor_groups.items():
        group_correlations = correlations[correlations.index.isin(group_cols)].sort_values(ascending=False)
        if not group_correlations.empty:
            grouped_correlations[group_name] = group_correlations
    
    feature_importance = correlations.abs().sort_values(ascending=False)
    
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
    for pattern_name, pattern_data in temporal_patterns.items():
        print(f"\n{pattern_name}:")
        print(pattern_data)
        pattern_data.to_frame().to_csv(output_dir / f'temporal_pattern_{pattern_name}.csv')
    
    plot_stress_time_series_by_subject(processed_df)
    plt.savefig(output_dir / 'stress_time_series.png')
    plt.close()
    
    print("\n3. Analyzing feature correlations...")
    correlations = analyze_feature_correlations(processed_df)
    print("\nFeature Correlations with Stress:")
    print(correlations)
    correlations.to_frame().to_csv(output_dir / 'feature_correlations.csv')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(processed_df[correlations.index].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()
    
    print("\n4. Analyzing predictive relationships...")
    predictor_results, processed_df = identify_stress_predictors(processed_df)
    print("\nTop Predictors:")
    print(predictor_results['top_predictors'])
    predictor_results['top_predictors'].to_frame().to_csv(output_dir / 'top_predictors.csv')
    
    plot_predictor_importance(predictor_results)
    plt.savefig(output_dir / 'predictor_importance.png')
    plt.close()
    
    relationships = analyze_predictor_relationships(processed_df, predictor_results['top_predictors'])
    print("\nPredictor Relationships:")
    print(relationships)
    pd.DataFrame(relationships).to_csv(output_dir / 'predictor_relationships.csv')
    
    print("\n5. Analyzing activity patterns...")
    activity_columns = ['IN_VEHICLE', 'ON_BIKE', 'ON_FOOT', 'RUNNING', 'STILL', 'TILTING', 'WALKING']
    
    if all(col in processed_df.columns for col in activity_columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=processed_df, x='total_activity', y='stress_normalized', hue='id')
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
        sns.scatterplot(data=processed_df, x='SLEEP_MINUTES', y='stress_normalized', hue='id')
        plt.title('Sleep Duration vs Normalized Stress')
        plt.savefig(output_dir / 'sleep_vs_stress.png')
        plt.close()
    
    print(f"\nAnalysis complete! Visualizations have been saved to {output_dir}/")
    
    return processed_df, subject_stats, correlations, temporal_patterns, predictor_results, relationships

if __name__ == "__main__":
    processed_df, subject_stats, correlations, temporal_patterns, predictor_results, relationships = main()
