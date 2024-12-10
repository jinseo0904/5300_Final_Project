import json
import os
import pandas as pd

# read the json file
with open('../question_answers_to_int_mapping.json') as f:
    mood_mappings = json.load(f)

questions = list(mood_mappings.keys())
print(questions)


# define custom function
# Iterate through each row
def fetch_answers(questionID, df):
    answers_raw = []
    answers_mapped = []

    for index, row in df.iterrows():
        # Convert the row to a list for easier access
        row_list = row.tolist()

        # Check if the questionID exists in the row
        if questionID in row_list:
            # Find the index of questionID
            question_index = row_list.index(questionID)

            # Check if there's a value immediately to the right
            if question_index + 2 < len(row_list):
                value_to_right = row_list[question_index + 2]
                # print(f"Row {index}: Value to the right of {questionID} is {value_to_right}")
                answers_raw.append(value_to_right)
                answers_mapped.append(mood_mappings[questionID].get(value_to_right, None))
            else:
                print(f"Row {index}: {questionID} is the last element, no value to the right.")
        else:
            print(f"Row {index}: {questionID} not found.")

    # print(f"Found {len(answers_raw)} answers for {questionID}")
    assert len(answers_raw) == len(df)
    assert len(answers_mapped) == len(df)
    colname = questionID + "_int"
    df[colname] = answers_mapped


# EDIT this for different directories
input_directory = '/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/phone-EMAs'
output_directory = '/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/phone-EMAs/mood_scores_extracted'

for file in os.listdir(input_directory):
    print(f"Processing {file}...")

    # if file is a csv, load it as dataframe
    if file.endswith('.csv'):
        print(file)
        df = pd.read_csv(os.path.join(input_directory, file))

        # filter only rows where the answer status is 'Completed' and Prompt_Type is Daily
        df = df[(df['Answer_Status'] == 'Completed') & (df['Prompt_Type'] == 'Daily')]

        # drop all columns beyond 'Question_19_Answer_Unixtime'
        df = df.iloc[:, :df.columns.get_loc('Question_19_Answer_Unixtime') + 1]

        # map all questions to int, according to mapping json
        for q in questions:
            colname = q + "_int"
            if colname not in df.columns:
                # print(f"Fetching {q}...")
                fetch_answers(q, df)

        # create a new column mood_score which is the sum of Q1_SAD_int through Q14_ROUT_int
        questions_int = [q + "_int" for q in questions]
        df['mood_score'] = df[questions_int].sum(axis=1)

        # print mean and standard deviation of mood_score
        mean = df['mood_score'].mean()
        std = df['mood_score'].std()
        print(f"Mean: {mean:.2f}")
        print(f"Standard deviation: {std:.2f}")

        # sort by Initial_Prompt_Date, from oldest to newest date
        df = df.sort_values('Initial_Prompt_Date')

        # save the dataframe to a new csv file
        file = file.replace('.csv', '_mood_scores.csv')
        df.to_csv(os.path.join(output_directory, file), index=False)
        print(f"Saved to {os.path.join(output_directory, file)}")
