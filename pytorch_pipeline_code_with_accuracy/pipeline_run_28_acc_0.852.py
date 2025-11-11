import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============= LOAD AND PREPARE DATA =============
# Load the dataset
df = pd.read_csv('heart_disease.csv')

# Handle missing values
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna('missing', inplace=True)

# ============= SEPARATE FEATURES AND TARGET =============
# Assuming 'target' is the target column
X = df.drop('target', axis=1).values
y = df['target'].values

# ============= TRAIN-TEST SPLIT =============
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============= NORMALIZATION =============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# ============= CREATE PYTORCH MODEL =============
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        
        # 3 hidden layers with ReLU activation
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        
        # Output layer (no activation because we use BCEWithLogitsLoss)
        self.fc_out = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc_out(x)
        return x

# Initialize model
input_size = X_train_scaled.shape[1]
model = BinaryClassificationModel(input_size)

# ============= TRAINING =============
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    
    # Mini-batch training
    indices = torch.randperm(len(X_train_tensor))
    for i in range(0, len(X_train_tensor), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train_tensor[batch_indices]
        y_batch = y_train_tensor[batch_indices]
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ============= EVALUATION ON TEST SET =============
model.eval()
with torch.no_grad():
    # Get predictions
    logits = model(X_test_tensor)
    # Convert logits to probabilities using sigmoid
    predictions = torch.sigmoid(logits)
    # Convert to binary predictions (threshold = 0.5)
    y_pred = (predictions > 0.5).float().numpy()

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Final Accuracy: {test_accuracy}")