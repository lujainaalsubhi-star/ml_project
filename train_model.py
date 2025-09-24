import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from preprocess_data import preprocess_data
import joblib
import numpy as np

if __name__ == '__main__':
    # Preprocess data
    X, y_label, y_type = preprocess_data('train_test_network.csv')

    # For this example, let's predict 'type' of attack
    y = y_type

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize and train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Model Evaluation for 'type' of attack:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-validation accuracies: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard deviation of CV accuracy: {np.std(cv_scores):.4f}")

    # Save the trained model and scaler
    joblib.dump(model, 'random_forest_model.pkl')
    print("\nModel saved as random_forest_model.pkl")


