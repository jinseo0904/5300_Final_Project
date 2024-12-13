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


def analyze_correlations_by_subject(df):
    """
    Calculate correlations separately for each subject
    Returns a dictionary with correlation results for each subject
    """
    # Exclude columns we don't want to correlate
    exclude_cols = [
        'id', 'date',  # Metadata
        'Q7_STRESS', 'stress_2', 'stress_normalized', 'watch_stress_normalized', 'stress_high',
        'day_of_week', 'month', 'is_weekend'  # Temporal columns
    ]
    
    # Get all numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols if col not in exclude_cols]
    
    # Dictionary to store results
    subject_correlations = {}
    
    # Calculate correlations for each subject
    for subject_id in df['id'].unique():
        subject_df = df[df['id'] == subject_id]
        
        # Calculate correlations with stress for all features
        correlations = {}
        for feature in features:
            if subject_df[feature].nunique() > 1:  # Skip constant features
                correlation = subject_df[feature].corr(subject_df['stress_normalized'])
                correlations[feature] = correlation
                
        # Convert to series and sort
        correlation_series = pd.Series(correlations).sort_values(ascending=False)
        subject_correlations[subject_id] = correlation_series
    
    return subject_correlations

def analyze_between_subject_correlations(df):
    """
    Calculate correlations using subject averages
    """
    # Exclude columns we don't want to average
    exclude_cols = [
        'id', 'date',  # Metadata
        'day_of_week', 'month', 'is_weekend'  # Temporal columns
    ]
    
    # Get all numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate subject averages
    subject_averages = df.groupby('id')[features].mean()
    
    # Calculate correlations between averaged features
    correlations = subject_averages.corr()
    
    return correlations

def export_correlation_results(subject_correlations, between_correlations, output_dir):
    """
    Export correlation results to CSV files
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Export within-subject correlations
    for subject_id, correlations in subject_correlations.items():
        correlations.to_csv(output_dir / f'within_subject_correlations_{subject_id}.csv')
    
    # Export between-subject correlations
    between_correlations.to_csv(output_dir / 'between_subject_correlations.csv')
    
    # Create summary report
    summary = []
    summary.append("Correlation Analysis Summary")
    summary.append("=" * 40)
    
    # Summarize within-subject correlations
    summary.append("\nWithin-Subject Correlation Summary:")
    for subject_id, correlations in subject_correlations.items():
        summary.append(f"\nSubject {subject_id}:")
        summary.append("Top 5 positive correlations:")
        for feat, corr in correlations.head().items():
            summary.append(f"  {feat}: {corr:.3f}")
        summary.append("Top 5 negative correlations:")
        for feat, corr in correlations.tail().items():
            summary.append(f"  {feat}: {corr:.3f}")
    
    # Summarize between-subject correlations
    summary.append("\nBetween-Subject Correlation Summary:")
    stress_corrs = between_correlations['stress_normalized'].sort_values(ascending=False)
    summary.append("\nTop 5 positive correlations with stress:")
    for feat, corr in stress_corrs.head().items():
        summary.append(f"  {feat}: {corr:.3f}")
    summary.append("\nTop 5 negative correlations with stress:")
    for feat, corr in stress_corrs.tail().items():
        summary.append(f"  {feat}: {corr:.3f}")
    
    # Save summary report
    with open(output_dir / 'correlation_analysis_summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    return '\n'.join(summary)

def plot_correlation_comparisons(subject_correlations, between_correlations, output_dir):
    """
    Create visualizations comparing within and between subject correlations
    """
    output_dir = Path(output_dir)
    
    # Get common features across all analyses
    common_features = set(between_correlations.index)
    for corr in subject_correlations.values():
        common_features &= set(corr.index)
    
    # Convert set to sorted list for consistent ordering
    common_features = sorted(list(common_features))
    
    # Create comparison plot
    plt.figure(figsize=(15, 8))
    
    # Plot between-subject correlations
    between_vals = between_correlations.loc[common_features, 'stress_normalized']
    plt.scatter(range(len(common_features)), between_vals, 
               label='Between-subject', color='black', s=100)
    
    # Plot within-subject correlations
    for subject_id, corr in subject_correlations.items():
        within_vals = corr[common_features]
        plt.scatter(range(len(common_features)), within_vals, 
                   label=f'Subject {subject_id}', alpha=0.5)
    
    plt.xticks(range(len(common_features)), common_features, rotation=45, ha='right')
    plt.ylabel('Correlation with Stress')
    plt.title('Comparison of Within and Between Subject Correlations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'correlation_comparison.png')
    plt.close()


def plot_correlation_heatmaps(subject_correlations, between_correlations, output_dir):
    """
    Create heatmap visualizations for both within-subject and between-subject correlations
    """
    output_dir = Path(output_dir)
    
    # Get common features
    common_features = set(between_correlations.index)
    for corr in subject_correlations.values():
        common_features &= set(corr.index)
    common_features = sorted(list(common_features))
    
    # 1. Between-subject correlation heatmap
    plt.figure(figsize=(20, 16))
    between_corr_subset = between_correlations.loc[common_features, common_features]
    mask = np.triu(np.ones_like(between_corr_subset), k=1)
    sns.heatmap(between_corr_subset,
                mask=mask,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Between-subject Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'between_subject_heatmap.png')
    plt.close()
    
    # 2. Within-subject correlation heatmaps
    for subject_id, corr_series in subject_correlations.items():
        subject_df = pd.DataFrame({feat: corr_series[feat] if feat in corr_series else np.nan 
                                 for feat in common_features}, index=common_features)
        
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(subject_df), k=1)
        sns.heatmap(subject_df,
                    mask=mask,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    square=True,
                    cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title(f'Within-subject Correlation Heatmap - Subject {subject_id}')
        plt.tight_layout()
        plt.savefig(output_dir / f'within_subject_heatmap_{subject_id}.png')
        plt.close()
    
    # 3. Average within-subject correlation heatmap
    avg_within_corr = np.zeros((len(common_features), len(common_features)))
    count_matrix = np.zeros((len(common_features), len(common_features)))
    
    for subject_id, corr_series in subject_correlations.items():
        for i, feat1 in enumerate(common_features):
            for j, feat2 in enumerate(common_features):
                if feat1 in corr_series and feat2 in corr_series:
                    val1 = corr_series[feat1]
                    val2 = corr_series[feat2]
                    if not np.isnan(val1) and not np.isnan(val2):
                        avg_within_corr[i, j] += val1
                        count_matrix[i, j] += 1
    
    # Calculate average (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_within_corr = np.divide(avg_within_corr, count_matrix, 
                                  out=np.zeros_like(avg_within_corr), 
                                  where=count_matrix != 0)
    
    avg_within_df = pd.DataFrame(avg_within_corr, 
                                index=common_features, 
                                columns=common_features)
    
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(avg_within_df), k=1)
    sns.heatmap(avg_within_df,
                mask=mask,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Average Correlation Coefficient'})
    
    plt.title('Average Within-subject Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'average_within_subject_heatmap.png')
    plt.close()

def analyze_correlation_differences(subject_correlations, between_correlations, output_dir):
    """
    Analyze and visualize differences between within and between subject correlations
    """
    output_dir = Path(output_dir)
    
    # Get common features
    common_features = set(between_correlations.index)
    for corr in subject_correlations.values():
        common_features &= set(corr.index)
    common_features = sorted(list(common_features))
    
    # Calculate average within-subject correlations
    avg_within = np.zeros(len(common_features))
    count = np.zeros(len(common_features))
    
    for subject_id, corr_series in subject_correlations.items():
        for i, feat in enumerate(common_features):
            if feat in corr_series:
                val = corr_series[feat]
                if not np.isnan(val):
                    avg_within[i] += val
                    count[i] += 1
    
    # Calculate average (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_within = np.divide(avg_within, count, 
                             out=np.zeros_like(avg_within), 
                             where=count != 0)
    
    # Get between-subject correlations
    between = between_correlations.loc[common_features, 'stress_normalized'].values
    
    # Calculate differences
    differences = between - avg_within
    
    # Create DataFrame for comparison
    comparison = pd.DataFrame({
        'between_subject': between,
        'avg_within_subject': avg_within,
        'difference': differences
    }, index=common_features)
    
    # Sort by absolute difference
    comparison['abs_difference'] = abs(comparison['difference'])
    comparison = comparison.sort_values('abs_difference', ascending=False)
    
    # Export results
    comparison.to_csv(output_dir / 'correlation_differences.csv')
    
    # Create visualization of differences
    plt.figure(figsize=(12, 6))
    comparison['difference'].plot(kind='bar')
    plt.title('Difference Between Between-subject and Average Within-subject Correlations')
    plt.xlabel('Features')
    plt.ylabel('Difference in Correlation (Between - Within)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_differences.png')
    plt.close()
    
    return comparison


def main():
    """
    Main function to run all analyses
    """
    output_dir = Path('stress_analysis_output')
    output_dir.mkdir(exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_all_subjects('Umberto/data')
    processed_df = load_and_preprocess_data(df)
    
    # Run both types of correlation analyses
    print("\nCalculating within-subject correlations...")
    subject_correlations = analyze_correlations_by_subject(processed_df)
    
    print("\nCalculating between-subject correlations...")
    between_correlations = analyze_between_subject_correlations(processed_df)
    
    # Export results
    print("\nExporting correlation results...")
    summary = export_correlation_results(subject_correlations, between_correlations, output_dir)
    
    # Create visualizations
    print("\nCreating correlation comparison plots...")
    plot_correlation_comparisons(subject_correlations, between_correlations, output_dir)
    
    print("\nGenerating correlation heatmaps...")
    plot_correlation_heatmaps(subject_correlations, between_correlations, output_dir)
    
    print("\nAnalyzing correlation differences...")
    correlation_differences = analyze_correlation_differences(subject_correlations, between_correlations, output_dir)
    
    print(f"\nAnalysis complete! Results have been saved to {output_dir}/")
    print("\nSummary of findings:")
    print(summary)
    
    return processed_df, subject_correlations, between_correlations, correlation_differences

if __name__ == "__main__":
    processed_df, subject_correlations, between_correlations, correlation_differences = main()
