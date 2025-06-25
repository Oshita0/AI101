# Task 1

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Step 2: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Perform feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the test data based on the training data

# Step 4: One-hot encode the target labels
y_train_encoded = to_categorical(y_train, num_classes=3)  # One-hot encode training labels
y_test_encoded = to_categorical(y_test, num_classes=3)  # One-hot encode test labels

# Display the shapes of the results
print("Training features shape:", X_train_scaled.shape)
print("Test features shape:", X_test_scaled.shape)
print("One-hot encoded training labels shape:", y_train_encoded.shape)
print("One-hot encoded test labels shape:", y_test_encoded.shape)

# Task 2

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Define the neural network model
model = Sequential()

# Input Layer: Automatically inferred from input shape (4 features)
# Hidden Layer: 8 neurons, ReLU activation
model.add(Dense(8, input_dim=4, activation='relu'))

# Output Layer: 3 neurons for 3 classes (Iris species), Softmax activation
model.add(Dense(3, activation='softmax'))

# Step 2: Compile the model
model.compile(loss='categorical_crossentropy',  # Loss function for multi-class classification
              optimizer='adam',                 # Adam optimizer
              metrics=['accuracy'])             # Accuracy as the evaluation metric

# Step 3: Train the model
history = model.fit(X_train_scaled, y_train_encoded, 
                    epochs=50, 
                    batch_size=10, 
                    validation_data=(X_test_scaled, y_test_encoded))

# Step 4: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

#Task 3

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Define the neural network model
model = Sequential()

# Input Layer: Automatically inferred from input shape (4 features)
# Hidden Layer: 8 neurons, ReLU activation
model.add(Dense(8, input_dim=4, activation='relu'))

# Output Layer: 3 neurons for 3 classes (Iris species), Softmax activation
model.add(Dense(3, activation='softmax'))

# Step 2: Compile the model with Adam optimizer and categorical cross-entropy loss function
model.compile(loss='categorical_crossentropy',  # Loss function for multi-class classification
              optimizer='adam',                 # Adam optimizer
              metrics=['accuracy'])             # Accuracy as the evaluation metric

# Step 3: Train the model on the training data for 100 epochs with batch size 5
history = model.fit(X_train_scaled, y_train_encoded, 
                    epochs=100,               # Number of epochs
                    batch_size=5,             # Batch size
                    validation_data=(X_test_scaled, y_test_encoded))  # Use validation data for tracking

# Task 4

# Step 4: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')
