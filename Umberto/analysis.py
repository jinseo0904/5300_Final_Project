# Import the script
from features_extraction import StressAnalyzer


# Initialize analyzer with participant ID
participant_id = "sharpnessnextpouch@timestudy_com"  # Replace with actual participant ID
analyzer = StressAnalyzer(participant_id)

# Load all data files
analyzer.load_ema_data("/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/phone_promptresponse.csv")
analyzer.load_activity_data("/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/watch_accelerometer_mims_hour.csv")
analyzer.load_phone_usage_data(
    app_usage_file="/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/phone_apps_usage_duration.csv",
    screen_events_file="/media/umberto/T7/intermediate_file/sharpnessnextpouch@timestudy_com/phone_system_events.csv"
)

# Run correlation analysis
correlations, merged_data = analyzer.analyze_correlations()

# View top correlations for each stress measure
for measure in analyzer.stress_questions.keys():
    print(f"\nTop 5 correlations with {measure}:")
    measure_corr = correlations[correlations['stress_measure'] == measure]
    print(measure_corr.sort_values('correlation', key=abs, ascending=False).head())

# Plot correlation results
analyzer.plot_correlations(correlations)
