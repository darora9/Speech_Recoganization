import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, BatchNormalization, InputLayer

# Define the TDNN model architecture
def build_tdnn_model(input_shape, num_classes):
    model = Sequential()
    
    # Input layer
    model.add(InputLayer(input_shape=input_shape))
    
    # First Conv1D layer with padding
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Second Conv1D layer with padding
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Third Conv1D layer with padding
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Fourth Conv1D layer with padding
    model.add(Conv1D(256, kernel_size=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Fifth Conv1D layer with padding
    model.add(Conv1D(256, kernel_size=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Dense layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Ensure X_scaled has the correct shape for Conv1D
X_scaled = X_scaled.reshape(-1, 13, 1)  # Assuming 13 timesteps and 1 feature per timestep

# Encode the labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check shapes
print(f"Input shape: {X_scaled.shape}")  # Should be (samples, 13, 1)
print(f"Label shape: {y_encoded.shape}")  # Should be (samples,)

# Split data (Optional but recommended)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build the TDNN model
input_shape = (X_train.shape[1], 1)  # Adjust input shape based on features
num_classes = len(np.unique(y_encoded))  # Determine the number of classes

model = build_tdnn_model(input_shape, num_classes)
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
