import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
file_path = 'ATPBetting/Data/2023.xlsx'
df = pd.read_excel(file_path)

# Drop unnecessary columns and handle missing values
df = df.drop(['ATP', 'Comment'], axis=1)
df = df.dropna()

# Convert categorical features to numerical using Label Encoding
label_encoder = LabelEncoder()
df['Winner'] = label_encoder.fit_transform(df['Winner'])
categorical_columns = ['Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Convert date to numerical format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y%m%d').astype(int)

# Define features and target
features = ['Location', 'Tournament', 'Date', 'Series', 'Court', 'Surface',
            'WRank', 'LRank', 'WPts', 'LPts', 'AvgW', 'AvgL', 'Wsets', 'Lsets']

X = df[features]
y = df['Winner']  # Assuming 'Winner' column indicates the winner (1 for Winner, 0 for Loser)

# Define numerical features for standardization
numerical_features = ['Date', 'WRank', 'LRank', 'WPts', 'LPts', 'AvgW', 'AvgL', 'Wsets', 'Lsets']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
    ],
    remainder='passthrough'  # Pass through non-numeric columns as is
)

# Create a pipeline with preprocessing and logistic regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=5000, class_weight='balanced', solver='newton-cg', C=0.1)),
])

# Grid search for hyperparameter tuning with KFold
param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'classifier__solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']}

# Define KFold
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=kfold)
grid_search.fit(X, y)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best model for prediction
best_model = grid_search.best_estimator_

# If you still want to split into training and testing sets after hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train set labels distribution:", y_train.value_counts())
print("Test set labels distribution:", y_test.value_counts())

# Train the best model on the training set
best_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = best_model.predict(X_test)

# Examine predictions and actual labels
print("Actual Labels:", y_test.values)
print("Predictions:", predictions)

# Evaluate the best model
accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions, zero_division=1)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_result)
