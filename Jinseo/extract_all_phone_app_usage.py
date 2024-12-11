import os
import pandas as pd

# NOTE: change the subject name field here
subject_name = 'feistydaycarelung@timestudy_com'
time_study_path = '/work/mhealthresearchgroup/TIME_STD/time_study_preprocess/intermediate_file/'
sub_path = os.path.join(time_study_path, subject_name)

# Initialize an empty DataFrame to hold all data
all_data = pd.DataFrame()
file_counts = 0

for date in os.listdir(sub_path):
    date_folder = os.path.join(sub_path, date)
    target_phone_filename = f'phone_apps_usage_duration_clean_{date}.csv'
    phone_path = date + '/' + target_phone_filename

    # check if phone prompt response csv file exists
    if os.path.isfile(os.path.join(sub_path, phone_path)):
        file_counts += 1
        # Read each CSV file
        daily_data = pd.read_csv(os.path.join(sub_path, phone_path))

        # Append to the master DataFrame
        all_data = pd.concat([all_data, daily_data], ignore_index=True)

print(f'Collected a total of {file_counts} app usage reports.')
# Ensure USAGE_DURATION_MIN is numeric
all_data['USAGE_DURATION_MIN'] = pd.to_numeric(all_data['USAGE_DURATION_MIN'], errors='coerce')

# 1. Calculate total usage duration over the whole year
total_usage_duration = all_data['USAGE_DURATION_MIN'].sum()
print(f"Total usage duration (minutes): {total_usage_duration}")
print(f"Total usage duration (hours): {total_usage_duration / 60:.2f}")

# 2. Find the most frequently used apps
# Group by APP_PACKAGE_NAME and calculate total usage for each app
app_usage = all_data.groupby('APP_PACKAGE_NAME')['USAGE_DURATION_MIN'].sum()

# Sort apps by total usage duration in descending order
most_used_apps = app_usage.sort_values(ascending=False)

# Display the top 10 most used apps
print("\nTop 30 Most Frequently Used Apps:")
print(most_used_apps.head(30))
