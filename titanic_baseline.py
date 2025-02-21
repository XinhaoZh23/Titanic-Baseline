import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path='train.csv'):
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # 2. Data preprocessing
    print("\nPreprocessing data...")
    
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Select features
    features = ['Age', 'Pclass', 'Sex']
    X = df[features].values
    y = df['Survived'].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    return X_train, y_train, X_val, y_val, scaler

class TitanicModel(nn.Module):
    def __init__(self, input_size=3):
        super(TitanicModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\nTraining the model...")
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.view(-1, 1))
            val_pred = (val_outputs > 0.5).float()
            val_accuracy = (val_pred.view(-1) == y_val).float().mean()
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Training Loss: {loss.item():.4f}")
            print(f"Validation Loss: {val_loss.item():.4f}")
            print(f"Validation Accuracy: {val_accuracy.item():.4f}\n")

def evaluate_test_set(model, scaler, test_file='test.csv', submission_file='submission.csv'):
    print("\nEvaluating on test set...")
    # Load test data
    test_df = pd.read_csv(test_file)
    print(f"Test set shape: {test_df.shape}")
    
    # Preprocess test data
    test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
    
    # Select features
    features = ['Age', 'Pclass', 'Sex']
    X_test = test_df[features].values
    
    # Standardize features using the same scaler
    X_test = scaler.transform(X_test)
    X_test = torch.FloatTensor(X_test)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predictions = (test_outputs > 0.5).float().numpy()
    
    # Create submission file
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions.flatten().astype(int)
    })
    submission_df.to_csv(submission_file, index=False)
    print(f"\nSubmission file created: {submission_file}")
    print("\nPreview of predictions:")
    print(submission_df.head())
    
    # Compare with gender_submission.csv as a baseline
    gender_submission = pd.read_csv('gender_submission.csv')
    matching_predictions = (submission_df['Survived'] == gender_submission['Survived']).mean()
    print(f"\nAgreement with gender_submission baseline: {matching_predictions:.2%}")

def main():
    # Load and preprocess data
    X_train, y_train, X_val, y_val, scaler = load_and_preprocess_data()
    
    # Initialize model
    model = TitanicModel()
    
    # Train model
    train_model(model, X_train, y_train, X_val, y_val)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_pred = (val_outputs > 0.5).float()
        final_accuracy = (val_pred.view(-1) == y_val).float().mean()
        print(f"\nFinal Validation Accuracy: {final_accuracy.item():.4f}")
    
    # Evaluate on test set and create submission
    evaluate_test_set(model, scaler)

if __name__ == "__main__":
    main()
