import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler  # For normalization
import matplotlib.pyplot as plt  # Ensure this import is added

# Assuming 'fetch_ucirepo' is not necessary with direct CSV loading
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# Separate features and target
X = data.drop('quality', axis=1)
y = data[['quality']]

# Normalize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Convert to tensors
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y_normalized, dtype=torch.float32).view(-1, 1)  # Ensure correct shape for y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the model
class RegressionMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()  # Simplified call to super
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # One output neuron

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # Streamlined forward method

# Model, criterion, and optimizer
model = RegressionMLP(X_train.shape[1], 10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:  # Print every 10 epochs for clarity
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluation
with torch.no_grad():
    predicted = model(X_test)
    rmse = torch.sqrt(criterion(predicted, y_test))

# RMSE and Plot
print(f'RMSE: {rmse.item()}')
plt.scatter(y_test.numpy(), predicted.numpy())
plt.xlabel('Actual Quality (Normalized)')
plt.ylabel('Predicted Quality (Normalized)')
plt.title('Actual vs Predicted Quality')
plt.show()
