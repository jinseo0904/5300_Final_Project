import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


def prepare_features(df):
    # Extract hour from timestamp
    df['hour'] = pd.to_datetime(df['hour']).dt.hour

    # Create binary features from activities
    activity_features = ['Used phone/tablet/computer', 'TV/videos/video games', 'Work',
                         'Exercise/physical activity', 'Socialized', 'Ate/drank']

    for activity in activity_features:
        df[f'activity_{activity.lower().replace("/", "_")}'] = df['Question_15_Answer_Text'].fillna('').str.contains(activity).astype(int)
        df[f'activity_{activity.lower().replace("/", "_")}'] += df['Question_16_Answer_Text'].fillna('').str.contains(activity).astype(int)

    # Create features from other emotions/states
    emotion_map = {'Not at all': 0, 'A little': 1, 'Moderately': 2, 'Quite a bit': 3, 'Extremely': 4}

    emotion_columns = {
        'sad': 'Q1_SAD',
        'happy': 'Q2_HAPP',
        'fatigued': 'Q3_FATIG',
        'energetic': 'Q4_EN',
        'relaxed': 'Q5_REL',
        'tense': 'Q6_TEN',
        'frustrated': 'Q8_FRUST',
        'nervous': 'Q9_NERV'
    }

    for emotion, col in emotion_columns.items():
        mask = df[f'Question_{col}_Text'].str.contains('Right now, how|Over the past day, how', na=False)
        df[f'feeling_{emotion}'] = df[mask][f'Question_{col}_Answer_Text'].map(emotion_map)

    # Create stress target (binary classification: stressed vs not stressed)
    stress_q = df['Question_7_Answer_Text'].map(emotion_map)
    df['stress_level'] = (stress_q > 1).astype(int)  # Threshold at "Moderately" stressed

    # Select features for model
    feature_cols = ['hour'] + \
                   [col for col in df.columns if col.startswith('activity_')] + \
                   [col for col in df.columns if col.startswith('feeling_')]

    return df[feature_cols].fillna(0), df['stress_level']


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))

    return model, scaler, X_test, y_test


def save_model(model, scaler, filename_prefix):
    joblib.dump(model, f'{filename_prefix}_model.joblib')
    joblib.dump(scaler, f'{filename_prefix}_scaler.joblib')


def predict_stress(model, scaler, new_data):
    scaled_data = scaler.transform(new_data)
    return model.predict_proba(scaled_data)


# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/afflictedrevenueepilepsy_processed_data.csv')

    # Prepare features and target
    X, y = prepare_features(df)

    # Train and evaluate model
    model, scaler, X_test, y_test = train_model(X, y)

    # Save model
    save_model(model, scaler, 'stress_prediction')

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
