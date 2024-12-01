import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from glob import glob
from typing import Dict, List, Tuple

def process_subject_data(subject_path: str, subject_id: str) -> pd.DataFrame:
    """
    Process and combine stress, physical activity, and phone usage data for one subject.
    Returns hourly-level dataframe with all metrics.
    """
    
    def get_stress_data() -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract stress responses from phone EMAs
        """
        ema_files = glob(os.path.join(subject_path, "*phone_promptresponse*.csv"))
        if not ema_files:
            return pd.DataFrame()
        
        dfs = []
        for file in ema_files:
            df = pd.read_csv(file)
            # Filter for completed responses
            df = df[df['Answer_Status'] == 'Completed']
            
            # Get stress question responses
            # Note: You'll need to adjust the actual column name based on your data
            stress_cols = [col for col in df.columns if 'stress' in col.lower()]
            if not stress_cols:
                continue
                
            # Take relevant columns
            df_stress = df[['Initial_Prompt_Local_Time'] + stress_cols].copy()
            df_stress['datetime'] = pd.to_datetime(df_stress['Initial_Prompt_Local_Time'])
            df_stress['hour'] = df_stress['datetime'].dt.floor('H')
            
            dfs.append(df_stress)
        
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs), stress_cols
    
    def get_physical_activity() -> pd.DataFrame:
        """
        Get hourly physical activity from MIMS data
        """
        activity_files = glob(os.path.join(subject_path, "*watch_accelerometer_mims_hour*.csv"))
        if not activity_files:
            return pd.DataFrame()
            
        dfs = []
        for file in activity_files:
            df = pd.read_csv(file)
            # Create datetime column from components
            df['hour'] = pd.to_datetime(df['YEAR_MONTH_DAY'] + ' ' + df['HOUR'].astype(str) + ':00:00')
            # Only include wear time activity
            df['activity_level'] = df['MIMS_SUM_WEAR']
            dfs.append(df[['hour', 'activity_level']])
            
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs)
    
    def get_phone_usage() -> pd.DataFrame:
        """
        Get hourly phone usage duration
        """
        usage_files = glob(os.path.join(subject_path, "*phone_app_usage_duration*.csv"))
        if not usage_files:
            return pd.DataFrame()
            
        dfs = []
        for file in usage_files:
            df = pd.read_csv(file)
            df['hour'] = pd.to_datetime(df['LOG_TIME']).dt.floor('H')
            # Sum up usage duration per hour
            hourly_usage = df.groupby('hour')['USAGE_DURATION_MIN'].sum().reset_index()
            dfs.append(hourly_usage)
            
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs)
    
    # Get all data
    stress_df, stress_columns = get_stress_data()
    activity_df = get_physical_activity()
    usage_df = get_phone_usage()
    
    # Merge all dataframes on hour
    if not all([len(df) > 0 for df in [stress_df, activity_df, usage_df]]):
        raise ValueError("Missing one or more required datasets")
    
    # Average stress by hour if multiple responses
    stress_hourly = stress_df.groupby('hour')[stress_columns].mean().reset_index()
    
    # Merge all data
    final_df = stress_hourly.merge(activity_df, on='hour', how='inner')\
                           .merge(usage_df, on='hour', how='inner')
    
    return final_df

def analyze_relationships(df: pd.DataFrame) -> Dict:
    """
    Calculate correlations and basic statistics
    """
    # Calculate correlations
    corr_matrix = df[['stress_level', 'activity_level', 'USAGE_DURATION_MIN']].corr()
    
    # Basic statistics
    stats = {
        'total_hours': len(df),
        'avg_stress': df['stress_level'].mean(),
        'avg_activity': df['activity_level'].mean(),
        'avg_phone_usage': df['USAGE_DURATION_MIN'].mean(),
        'correlations': corr_matrix.to_dict()
    }
    
    return stats

if __name__ == "__main__":
    # Replace with your data path and subject ID
    SUBJECT_PATH = "/path/to/subject/data"
    SUBJECT_ID = "example_subject"
    
    try:
        # Process data
        combined_df = process_subject_data(SUBJECT_PATH, SUBJECT_ID)
        
        # Analyze relationships
        results = analyze_relationships(combined_df)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Total hours analyzed: {results['total_hours']}")
        print(f"Average stress level: {results['avg_stress']:.2f}")
        print(f"Average activity level: {results['avg_activity']:.2f}")
        print(f"Average phone usage (minutes/hour): {results['avg_phone_usage']:.2f}")
        print("\nCorrelations:")
        print(pd.DataFrame(results['correlations']))
        
        # Save processed data
        combined_df.to_csv(f"{SUBJECT_ID}_processed_data.csv", index=False)
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")