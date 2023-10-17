
import pandas as pd
# Path to the file
!url = 'https://raw.githubusercontent.com/vinaytukka/Files_Repo/main/Boston%20housing%20Price.csv'

column_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing = pd.read_csv(url, header=None,names=column_names,delim_whitespace=True)
housing.head()

X = housing.drop('MEDV', axis=1).copy()
y = housing.pop('MEDV').copy()

# Split the data into Train and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# import the necessary libraries

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define hyperparameters
hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.01
num_epochs = 1000
batch_size = 32

# Scale your data using StandardScaler
def scale_datasets(x_train, x_test):
    standard_scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(standard_scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(standard_scaler.transform(x_test), columns=x_test.columns)
    return x_train_scaled, x_test_scaled


# Load your dataset, perform data scaling, and define x_train, y_train

x_train_scaled, x_test_scaled = scale_datasets(X_train, X_test)

# Convert your Pandas DataFrames to NumPy arrays
x_train = x_train_scaled.values
y_train = y_train.values

#Split the data into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

# Convert your data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


x_train.shape, X_test.shape, x_val.shape,  y_train.shape, y_test.shape, y_val.shape

#Initialize the weights and biases

W1 = torch.randn(hidden_units1, x_train.shape[1]) * 0.1
b1 = torch.randn(hidden_units1) * 0.01
W2 = torch.randn(hidden_units2, hidden_units1) * 0.01
b2 = torch.randn(hidden_units2) * 0.01
W3 = torch.randn(hidden_units3, hidden_units2) * 0.01
b3 = torch.randn(hidden_units3) * 0.01
W4 = torch.randn(1, hidden_units3) * 0.01
b4 = torch.randn(1) * 0.01

# Create DataLoader for batching
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

parameters = [W1, b1, W2, b2, W3, b3, W4, b4]

print(sum(p.nelement() for p in parameters)) #number of parameters in total

for p in parameters:
  p.requires_grad = True

#Optimization
learning_rates = [learning_rate, learning_rate, learning_rate, learning_rate, learning_rate, learning_rate, learning_rate, learning_rate]
optimizer = optim.Adam([{'params': param, 'lr': lr} for param, lr in zip(parameters, learning_rates)])

# Training loop
train_losses = []
val_losses = []

for epoch in range(num_epochs):
  # print('epoch', epoch)

  for inputs, targets in train_loader:

      # print(inputs.shape, targets.shape)
      optimizer.zero_grad()

      # Forward pass
      x = inputs
      x = F.relu(x.mm(W1.t()) + b1)
      x = F.relu(x.mm(W2.t()) + b2)
      x = F.relu(x.mm(W3.t()) + b3)
      outputs = x.mm(W4.t()) + b4

      # # Calculate loss (Mean Squared Logarithmic Error)
      tr_loss = F.mse_loss(outputs, targets.view(-1, 1))
      # # print(loss.item())

      # Backpropagation
      tr_loss.backward()
      # # Update weights
      optimizer.step()

  train_losses.append(tr_loss.log10().item())

      #validation step
  with torch.no_grad():
    for inputs, targets in val_loader:

      x = inputs
      x = F.relu(x.mm(W1.t()) + b1)
      x = F.relu(x.mm(W2.t()) + b2)
      x = F.relu(x.mm(W3.t()) + b3)
      outputs = x.mm(W4.t()) + b4
      v_loss = F.mse_loss(outputs, targets.view(-1, 1))
    val_losses.append(v_loss.log10().item())

  if epoch % 100 == 0:
    print(f'{epoch}/{num_epochs}: Train loss: {tr_loss.item():.2f}, Val loss:{v_loss:.2f}')

  # break

#Plot the Train and Test Loss

plt.plot(train_losses, 'blue')
plt.plot(val_losses, 'orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Val Loss Curve")
plt.legend(['Train','val'])
plt.show()


# Perform inference using the trained weights and biases
with torch.no_grad():
    x = x_val_tensor
    x = F.relu(x.mm(W1.t()) + b1)
    x = F.relu(x.mm(W2.t()) + b2)
    x = F.relu(x.mm(W3.t()) + b3)
    outputs = x.mm(W4.t()) + b4

# Convert the PyTorch tensor to a NumPy array
predictions = outputs

# check the Predicitions
predictions[:10]

y_val = y_val_tensor.reshape(-1,1)
y_val[:10]

#Measure the R Squared on the model predictions with actual y
import numpy as np
def r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6 ))
    return r2.item()  # Return the R-squared value as a scalar

r_squared(y_val, predictions)

# 0.9121
