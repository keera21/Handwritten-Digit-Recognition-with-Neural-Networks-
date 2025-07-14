from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps

# Load trained model once
model = keras.models.load_model('handwritten_digit_model.h5')

# Loop through image files named 1.png to 9.png
for i in range(1, 10):
    try:
        filename = f"{i}.png"
        img = Image.open(filename).convert("L")     # Convert to grayscale
        img = ImageOps.invert(img)                  # Invert image (white background)
        img = img.resize((28, 28))                  # Resize to 28x28
        img = np.array(img) / 255.0                 # Normalize
        img = img.reshape(1, 28, 28)                # Reshape for model input

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        print(f"{filename} → Predicted Digit: {digit}")

    except FileNotFoundError:
        print(f"{filename} not found. Skipping.")
