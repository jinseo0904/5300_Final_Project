# Import the script
from features_extraction import StressAnalyzer


# Initialize analyzer with participant ID
participant_id = "sharpnessnextpouch@timestudy_com"
analyzer = StressAnalyzer(participant_id)

# Load all data files - Fixed file paths and column handling
analyzer.load_ema_data("/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/phone_promptresponse.csv")
analyzer.load_activity_data("/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/watch_accelerometer_mims_hour.csv")
analyzer.load_phone_usage_data(
    app_usage_file="/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/phone_apps_usage_duration.csv",
    screen_events_file="/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/phone_system_events.csv"
)

# Run correlation analysis with explicit window size
correlations, merged_data = analyzer.analyze_correlations(feature_window='1h')

# View correlations for each stress measure with minimum sample size filtering
for measure in analyzer.stress_questions.keys():
    print(f"\nTop 5 correlations with {measure}:")
    measure_corr = correlations[
        (correlations['stress_measure'] == measure) &
        (correlations['n_samples'] >= 10)  # Filter for reliable correlations
    ]
    print(measure_corr.sort_values('correlation', key=abs, ascending=False).head())

# Plot correlation results with statistical significance
analyzer.plot_correlations(correlations, n_top=10)

# Save results to CSV for further analysis
correlations.to_csv(f'correlations_{participant_id}.csv', index=False)
merged_data.to_csv(f'merged_data_{participant_id}.csv', index=True)
