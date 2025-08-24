# Handwritten_Digit_Recognition

This project implements a handwritten digit recognition system using a neural network trained on the MNIST dataset. The trained model predicts digits from grayscale images.

## Features

- Uses a simple feed-forward neural network built with TensorFlow/Keras.
- Loads a pre-trained model for digit classification.
- Processes images from a folder, resizing and normalizing them.
- Predicts and displays the digit along with the input image.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV (`cv2`)
- NumPy
- Matplotlib

Install dependencies with:

   bash
pip install tensorflow opencv-python numpy matplotlib
Usage
Place your digit images (named digits1.png, digits2.png, ...) inside the digits folder.

Ensure the trained model file handwritten.keras is in the project root directory.

Run the script:

bash
Copy code
python main.py
The script will process each image, predict the digit, print the result, and display the image.

Project Structure
nginx
Copy code
Handwritten Digit Recognition/
│
├── digits/                 # Folder containing digit images (digits1.png, digits2.png, ...)
├── handwritten.keras       # Trained Keras model file
├── main.py                 # Prediction script
└── README.md
Training the Model
If you want to train the model yourself, use the MNIST dataset and the following example code snippet:

python
Copy code
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save("handwritten.keras")
License
This project is open-source and available under the MIT License.
