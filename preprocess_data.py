import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop columns with too many missing values or irrelevant for initial model
    # For simplicity, let's drop columns that are mostly empty or unique identifiers
    # This needs more careful analysis in a real scenario
    df = df.drop(columns=['dns_query', 'ssl_version', 'ssl_cipher', 'ssl_subject', 'ssl_issuer', 'http_uri', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types', 'weird_name', 'weird_addl', 'weird_notice'], errors='ignore')

    # Handle missing values: fill numerical with median, categorical with mode
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['label', 'type']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Separate features (X) and target (y)
    X = df.drop(columns=['label', 'type'])
    y_label = df['label']
    y_type = df['type']

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y_label, y_type

if __name__ == '__main__':
    X, y_label, y_type = preprocess_data('train_test_network.csv')
    print("Features shape:", X.shape)
    print("Label target shape:", y_label.shape)
    print("Type target shape:", y_type.shape)
    print("First 5 rows of preprocessed features:\n", X.head())
    print("First 5 rows of label target:\n", y_label.head())
    print("First 5 rows of type target:\n", y_type.head())

