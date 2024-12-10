import os
import pandas as pd

# NOTE: change the subject name field here
subject_name = 'feistydaycarelung@timestudy_com'
time_study_path = '/work/mhealthresearchgroup/TIME_STD/time_study_preprocess/intermediate_file/'
sub_path = os.path.join(time_study_path, subject_name)

# Initialize an empty list to hold filtered DataFrames
filtered_dfs = []
daily_report_counts = 0
step_counts = []

for date in os.listdir(sub_path):
    date_folder = os.path.join(sub_path, date)
    target_phone_filename = f'phone_promptresponse_clean_{date}.csv'
    phone_path = date + '/' + target_phone_filename

    # check if phone prompt response csv file exists
    if os.path.isfile(os.path.join(sub_path, phone_path)):
        # print(f'Phone prompt exists for date {date}')

        # check if the daily reports exists
        df = pd.read_csv(os.path.join(sub_path, phone_path))
        filtered_rows = df[(df['Prompt_Type'] == 'Daily') & (df['Answer_Status'] == 'Completed')]
        daily_report_counts += len(filtered_rows)

        # add step counts to the dataframe too
        if len(filtered_rows) > 0:
            steps_path = date + '/' + f'phone_stepCount_day_{date}.csv'
            steps_df = pd.read_csv(os.path.join(sub_path, steps_path))

            # considering edge case where single day has more than 1 daily reports
            # print(int(steps_df['TOTAL_STEPS'].iloc[0]))
            step_counts += [int(steps_df['TOTAL_STEPS'].iloc[0])] * len(filtered_rows)

        # Append the filtered DataFrame to the list
        filtered_dfs.append(filtered_rows)

# Concatenate all filtered DataFrames into a single DataFrame
final_filtered_data = pd.concat(filtered_dfs, ignore_index=True)

assert len(final_filtered_data) == len(step_counts)
final_filtered_data['Steps'] = step_counts

print(step_counts[:20])
# Display the resulting DataFrame
print(len(final_filtered_data))
print(daily_report_counts)

filename = subject_name + '_all_daily_reports_updated.csv'
final_filtered_data.to_csv(filename)
