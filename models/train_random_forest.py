import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

def train():
    print("Loading UCI dataset...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    with open(os.path.join(data_dir, 'uci_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
        X_train, y_train = train_data['X_train'], train_data['y_train']

    with open(os.path.join(data_dir, 'uci_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
        X_test, y_test = test_data['X_test'], test_data['y_test']

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    feature_names = X_train.columns.tolist()
    models_dir = os.path.dirname(os.path.abspath(__file__))
    feature_names_path = os.path.join(models_dir, 'rf_feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Saved feature column order to {feature_names_path}")

    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    print("Evaluating on the test set...")
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy:  {acc:.4f}")
    print("\nClassification Report (Precision, Recall, F1):")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(models_dir, 'rf_model.pkl')
    print(f"\nSaving model to {model_path} using joblib...")
    joblib.dump(rf_model, model_path)

    if acc < 0.85:
        print("\nTarget accuracy of >85% was NOT reached.")
        print("Suggested adjustments: Increase n_estimators (e.g., 200), try bootstrap=False, or tune min_samples_split.")
    else:
        print("\nTarget accuracy of >85% REACHED successfully!")

if __name__ == '__main__':
    train()
