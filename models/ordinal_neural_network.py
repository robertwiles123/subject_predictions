# this may work, but needs more data.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from grade_packages import encoding
import numpy as np

# Define the architecture of the Ordinal Neural Network
class OrdinalNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OrdinalNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the data and split into features and target
data = pd.read_csv('../csv_clean/ordinal_combined_undersample.csv')
X_ordinal, y = encoding.ordinal_encoder_combined(data)

# convert bool to np array
PP_array = data['PP'].values.reshape(-1, 1)
SEN_bool_array = data['SEN bool'].values.reshape(-1, 1)
' combined features'
X = np.concatenate((X_ordinal, PP_array, SEN_bool_array), axis=1)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58, stratify=y)

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the target variable to tensor
y_train = torch.tensor(y_train.astype(float)).unsqueeze(1)
y_test = torch.tensor(y_test.astype(float)).unsqueeze(1)

# Define the neural network classifier
input_dim = X_train.shape[1]
output_dim = y_train.max().item() + 1  # Number of unique classes
hidden_dim = 128
model = NeuralNetClassifier(
    module=OrdinalNeuralNetwork,
    module__input_dim=input_dim,
    module__hidden_dim=hidden_dim,
    module__output_dim=output_dim,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    max_epochs=100,
    verbose=1,
)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = (y_pred == y_test).float().mean().item()
print("Accuracy:", accuracy)
