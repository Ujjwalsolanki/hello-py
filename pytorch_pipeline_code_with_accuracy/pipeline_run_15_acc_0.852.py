import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================
df = pd.read_csv('heart_disease.csv')
print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# ============================================
# 2. HANDLE MISSING VALUES
# ============================================
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Handle missing values for numeric columns (use median)
numeric_imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns)

print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")

# ============================================
# 3. TRAIN-TEST SPLIT (80/20)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================
# 4. NORMALIZATION (StandardScaler)
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

print(f"X_train_tensor shape: {X_train_tensor.shape}")
print(f"y_train_tensor shape: {y_train_tensor.shape}")

# ============================================
# 5. CREATE PYTORCH MODEL WITH 3 HIDDEN LAYERS AND RELU ACTIVATION
# ============================================
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        # 3 hidden layers with ReLU activation
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        
        # Output layer for binary classification
        self.fc_out = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc_out(x)
        return x

# Initialize model
input_size = X_train_scaled.shape[1]
model = BinaryClassifier(input_size)
print(f"\nModel created with input size: {input_size}")

# ============================================
# 6. TRAINING SETUP
# ============================================
criterion = nn.BCEWithLogitsLoss()  # Correct loss function for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ============================================
# 7. TRAIN THE MODEL
# ============================================
num_epochs = 100
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# ============================================
# 8. EVALUATE ON TEST SET
# ============================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    # Convert logits to probabilities
    test_predictions = torch.sigmoid(test_outputs)
    # Convert probabilities to binary predictions (0.5 threshold)
    test_predictions_binary = (test_predictions > 0.5).float()
    
    # Calculate accuracy
    accuracy = (test_predictions_binary == y_test_tensor).float().mean()
    test_accuracy = accuracy.item()

print(f"\nFinal Accuracy: {test_accuracy}")