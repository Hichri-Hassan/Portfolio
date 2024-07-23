import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
def preprocess_image(image_path):
    img = image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to 0-1 range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the MNIST dataset from the CSV file
mnist_data = pd.read_csv("mnist_train.csv")

# Split the dataset into features (X) and labels (y)
X = mnist_data.drop("label", axis=1).values
y = mnist_data["label"].values

# Normalize the pixel values to the range [0, 1]
X = X / 255.0

# Reshape the data to fit a CNN (batch, height, width, channels)
X = X.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y = to_categorical(y, 10)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
# Define the neural network model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Use softmax for multiclass classification
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save("mnist_cnn_model.h5")
"""

model = load_model("mnist_cnn_model.h5")

# Load and preprocess an image
"""for i in range(100):
    input_image = X_test[i].reshape(1, 28, 28, 1)  # Reshape to match the model's input shape
    #image_array = preprocess_image(input_image)

    # Make predictions on the preprocessed input image
    predictions = model.predict(input_image)

    # Get the predicted class index
    predicted_class = np.argmax(predictions)

    # Print the true label and predicted class
    print(f'True label: {np.argmax(y_test[i])}')
    print(f'Predicted class: {predicted_class}')"""
image = preprocess_image("3.png")
# Make predictions on the reshaped input image
predictions = model.predict(image)

# Get the predicted class index
predicted_class = np.argmax(predictions)

# Print the true label and predicted class
print(f'Predicted class: {predicted_class}')
