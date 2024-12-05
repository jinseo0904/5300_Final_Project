import csv
import json
from pathlib import Path
import pandas as pd


def create_question_mapping(csv_content):
    # Read the CSV content into a pandas DataFrame
    df = pd.read_csv(csv_content, low_memory=False)

    # Find all columns that contain question IDs and text
    id_columns = [col for col in df.columns if col.startswith('Question_') and col.endswith('_ID')]
    text_columns = [col for col in df.columns if col.startswith('Question_') and col.endswith('_Text') and not col.endswith('_Answer_Text')]
    answer_columns = [col for col in df.columns if col.startswith('Question_') and col.endswith('_Answer_Text')]

    # Create a mapping to store unique question ID-text pairs
    question_mapping = {}

    # Process each row
    for i, row in df.iterrows():
        # For each row, look at all question ID-text pairs
        for id_col, text_col, answ_col in zip(id_columns, text_columns, answer_columns):
            if pd.isna(row[id_col]) or pd.isna(row[text_col]) or row[id_col] in ["key_next_wake_que", 'key_prev_sleep_que', 'key_prev_wake_que', 'key_next_sleep_que']:
                continue
            question_id = row[id_col]
            question_text = row[text_col]
            question_answer = row[answ_col]

            if question_answer not in ["Yes", "No", "A little", "Sometimes", "Not really", "Not at all"]:
                continue

            # Only add non-empty pairs with specified answer options
            if question_id and question_text:
                # If we find the same ID with different text, log it
                if question_id in question_mapping and question_mapping[question_id] != question_text:
                    print(f"Warning: Found different text for question ID {question_id}:")
                    print(f"Existing: {question_mapping[question_id]}")
                    print(f"New: {question_text}")
                question_mapping[question_id] = question_text

    return question_mapping


def process_directory(directory_path):
    # Combined mapping for all files
    combined_mapping = {}

    # Process each CSV file in the directory
    csv_files = list(Path(directory_path).glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return combined_mapping

    print(f"Found {len(csv_files)} CSV files to process")

    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        try:
            # Get mapping from this file
            file_mapping = create_question_mapping(csv_path)

            # Update combined mapping
            for question_id, question_text in file_mapping.items():
                if question_id in combined_mapping and combined_mapping[question_id] != question_text:
                    print(f"\nWarning: Different text found for {question_id} in {csv_path.name}")
                    print(f"Existing: {combined_mapping[question_id]}")
                    print(f"New: {question_text}")
                combined_mapping[question_id] = question_text

        except Exception as e:
            print(f"Error processing {csv_path.name}: {str(e)}")

    return combined_mapping


def save_mapping(mapping, output_path):
    # Sort the mapping by question ID
    sorted_mapping = dict(sorted(mapping.items()))

    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_mapping, f, indent=2)

    # Also save as CSV for easy viewing in spreadsheet software
    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Question ID', 'Question Text'])
        for question_id, question_text in sorted_mapping.items():
            writer.writerow([question_id, question_text])

    return sorted_mapping


if __name__ == "__main__":
    # Directory containing CSV files
    directory = "/Users/umberto/repo/5300_Final_Project/phone-EMAs"
    output_file = "question_mapping.json"

    print("Starting question mapping process...")

    # Process all CSV files and get combined mapping
    combined_mapping = process_directory(directory)

    # Save results
    if combined_mapping:
        save_mapping(combined_mapping, output_file)
        print("\nProcessing complete!")
        print(f"Found {len(combined_mapping)} unique questions with specified answer options")
        print(f"Results saved to {output_file} and {output_file.replace('.json', '.csv')}")

        # Print first few entries as example
        print("\nExample entries:")
        for i, (qid, text) in enumerate(list(combined_mapping.items())[:5]):
            print(f"{qid}: {text}")
    else:
        print("No questions found in the CSV files with the specified answer options")
