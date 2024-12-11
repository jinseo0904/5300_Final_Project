import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

mood_scores_path = '/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/phone-EMAs/mood_scores_extracted'
mood_scores_files = os.listdir(mood_scores_path)
merged_folder_path = '/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/merged_for_training'


def merge_and_save_csvfile(file):
    print(f"Processing {file}...")
    df = pd.read_csv(os.path.join(mood_scores_path, file))
    sub_id = df.Participant_ID[0]

    # load watch daily metrics csv (for x features)
    watch_csv_filename = f"{sub_id}_daily_metrics.csv"
    watch_csv_folder = '/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Umberto/data'

    # if watch csv does not exist, skip this file
    if watch_csv_filename not in os.listdir(watch_csv_folder):
        print(f"Watch csv for {sub_id} not found.")
        return
    watch_df = pd.read_csv(os.path.join(watch_csv_folder, watch_csv_filename))

    df['date'] = pd.to_datetime(df['Initial_Prompt_Date']).dt.date
    watch_df['date'] = pd.to_datetime(watch_df['date']).dt.date
    merged = watch_df.merge(df, on='date', how='inner')
    # print(merged.head())
    merged.to_csv(f'/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/merged_for_training/{sub_id}_merged.csv')
    print(f"Saved {sub_id}_merged.csv")


def train_and_evaluate_models(merged_csv_path, summary_df):
    # load merged csv
    df = pd.read_csv(merged_csv_path)
    # Define the columns to select
    selected_columns = [
        'IN_VEHICLE', 'ON_BIKE', 'ON_FOOT', 'RUNNING', 'STILL', 
        'TILTING', 'WALKING', 'UNKNOWN', 'MIMS_SUM_WEAR', 
        'SLEEP_MINUTES', 'SCREEN_ON_SECONDS', 'UNLOCK_EVENTS_NUM', 
        'USAGE_DURATION_MIN', 'excite_14', 'focus_3', 'frust_12', 
        'happy_8', 'int_exer_3', 'nervous_4', 'proc_4', 'relax_10', 
        'sad_7', 'stress_2', 'tense_11'
    ]

    # Filter the DataFrame to include only selected columns
    X = df[selected_columns]

    # Print selected columns for verification
    print("Selected Features:", X.columns)
    #X = X.drop(columns=['Q7_STRESS'])
    # y = df['Q7_STRESS'] 
    # set the dependent variable
    y = df['mood_score']
    # drop 'Q7_STRESS' column from STRESS
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("test1")
    r2, mae = evaluate_and_print_metrics(y_test, y_pred, sub_id)

    # now, test stress
    y = df['Q7_STRESS'] 
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    r2_stress, mae_stress = evaluate_and_print_metrics(y_test, y_pred, sub_id)

    # add row to summary_df
    summary_df = pd.concat([summary_df, pd.DataFrame([{'Participant_ID': sub_id, 'Model_Type': 'All_Features', 'R2_mood': r2, 'MAE_mood': mae,
                                                       'R2_stress': r2_stress, 'MAE_stress': mae_stress}])], ignore_index=True)
    #summary_df = summary_df.append({'Participant_ID': sub_id, 'Model_Type': 'All_Features',
                                     #'R2_mood': r2, 'MAE_mood': mae}, ignore_index=True)

    # Now, train the model without watch ema features
    # set X2 to columns from index 3 to the location of 'Q1_SAD' col
    selected_columns = [
        'IN_VEHICLE', 'ON_BIKE', 'ON_FOOT', 'RUNNING', 'STILL', 
        'TILTING', 'WALKING', 'UNKNOWN', 'MIMS_SUM_WEAR', 
        'SLEEP_MINUTES', 'SCREEN_ON_SECONDS', 'UNLOCK_EVENTS_NUM', 
        'USAGE_DURATION_MIN'
    ]

    # Filter the DataFrame to include only selected columns
    X2 = df[selected_columns]
    y_mood = df['mood_score'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X2, y_mood, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("test2")
    r2, mae = evaluate_and_print_metrics(y_test, y_pred, sub_id)

    # Now, train model for Q7_STRESS
    y_stress = df['Q7_STRESS'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X2, y_stress, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    r2_stress, mae_stress = evaluate_and_print_metrics(y_test, y_pred, sub_id)

    summary_df = pd.concat([summary_df, pd.DataFrame([{'Participant_ID': sub_id, 'Model_Type': 'No_Watch_EMAs',
                                     'R2_mood': r2, 'MAE_mood': mae, 'R2_stress': r2_stress, 'MAE_stress': mae_stress}])], ignore_index=True)

    print("test3")
    # set X3 to columns from index of 'Q1_SAD' to index 25
    selected_columns = [ 'excite_14', 'focus_3', 'frust_12', 
        'happy_8', 'int_exer_3', 'nervous_4', 'proc_4', 'relax_10', 
        'sad_7', 'stress_2', 'tense_11'
    ]

    # Filter the DataFrame to include only selected columns
    X3 = df[selected_columns]
    #X3 = X3.drop(columns=['Q7_STRESS'])
    #print(X3.columns)
    X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    r2, mae = evaluate_and_print_metrics(y_test, y_pred, sub_id)

    # now, test stress
    y = df['Q7_STRESS'] 
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    r2_stress, mae_stress = evaluate_and_print_metrics(y_test, y_pred, sub_id)

    summary_df = pd.concat([summary_df, pd.DataFrame([{'Participant_ID': sub_id, 'Model_Type': 'Watch_EMAs_Only',
                                     'R2_mood': r2, 'MAE_mood': mae, 'R2_stress': r2_stress, 'MAE_stress': mae_stress}])], ignore_index=True)
    return summary_df
    


def evaluate_and_print_metrics(y_test, y_pred, sub_id):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("\n\n" + "=" * 50)
    print(f"Report for {sub_id}")
    print(f"R2: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print("=" * 50 + "\n\n")
    return r2, mae


watch_metrics_path = '/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Umberto/data'
# create an empty dataframe to combine the ML performance summary
summary_df = pd.DataFrame(columns=['Participant_ID', 'Model_Type', 'R2_mood', 'MAE_mood', 'R2_stress', 'MAE_stress'])

for file in mood_scores_files:
    df = pd.read_csv(os.path.join(mood_scores_path, file))
    sub_id = df.Participant_ID[0]
    print(sub_id)

    merge_and_save_csvfile(file)

    # check if merged csv already exists
    if f"{sub_id}_merged.csv" in os.listdir('/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/merged_for_training'):
        # train models
        merged_path = os.path.join(merged_folder_path, f"{sub_id}_merged.csv")
        summary_df = train_and_evaluate_models(merged_path, summary_df)

    # Round all float columns to the nearest thousandth
    summary_df = summary_df.round(decimals=3)

    # drop all rows with Participant_ID = 'groinunratedbattery'
    summary_df = summary_df[summary_df['Participant_ID'] != 'groinunratedbattery']

    # calculate the maximum of R2_mood and R2_stress
    max_r2_mood = summary_df['R2_mood'].max()
    max_r2_stress = summary_df['R2_stress'].max()
    print("Max R2 Mood: ", max_r2_mood)
    print("Max R2 Stress: ", max_r2_stress)

    # calculate the minimum of MAE_mood and MAE_stress
    min_mae_mood = summary_df['MAE_mood'].min()
    min_mae_stress = summary_df['MAE_stress'].min()
    print("Min MAE Mood: ", min_mae_mood)
    print("Min MAE Stress: ", min_mae_stress)
    
    summary_df.to_csv('/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/summary_df.csv', index=False)