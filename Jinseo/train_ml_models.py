import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
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
    #print(merged.head())
    merged.to_csv(f'/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/merged_for_training/{sub_id}_merged.csv') 
    print(f"Saved {sub_id}_merged.csv")

def train_and_evaluate_models(merged_csv_path):
    # load merged csv
    df = pd.read_csv(merged_csv_path)
    sub_id = df.Participant_ID[0]

    # set the independent variables
    X = df[df.columns[3:25]]
    # set the dependent variable
    y = df['mood_score']
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("test1")
    evaluate_and_print_metrics(y_test, y_pred, sub_id)

    # Now, train the model without watch ema features
    # set X2 to columns from index 3 to the location of 'Q1_SAD' col
    X2 = df[df.columns[3:df.columns.get_loc('Q1_SAD')]]
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("test2")
    evaluate_and_print_metrics(y_test, y_pred, sub_id)

    print("test3")
    # set X3 to columns from index of 'Q1_SAD' to index 25
    X3 = df[df.columns[df.columns.get_loc('Q1_SAD'):25]]
    #print(X3.columns)
    X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.3, random_state=42, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    evaluate_and_print_metrics(y_test, y_pred, sub_id)
    pass

def evaluate_and_print_metrics(y_test, y_pred, sub_id):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("\n\n" + "=" * 50)
    print(f"Report for {sub_id}")
    print(f"R2: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print("=" * 50 + "\n\n")

watch_metrics_path = '/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Umberto/data'
for file in mood_scores_files:
    df = pd.read_csv(os.path.join(mood_scores_path, file))
    sub_id = df.Participant_ID[0]

    merge_and_save_csvfile(file)

    # check if merged csv already exists
    if f"{sub_id}_merged.csv" in os.listdir('/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/merged_for_training'):
        # train models
        merged_path = os.path.join(merged_folder_path, f"{sub_id}_merged.csv")
        train_and_evaluate_models(merged_path)
    
