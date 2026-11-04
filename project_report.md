# Project Report: Handwritten Digit Recognition

## 1. Aim

Build a program that recognizes handwritten digits (0–9) from a drawing or image using a convolutional neural network. For short, single-line numbers, separate the digit regions and classify them left to right.

## 2. Dataset

The project uses MNIST, available through `keras.datasets.mnist`. It contains 70,000 grayscale images: 60,000 for training and 10,000 for final testing. Every image is 28 × 28 pixels and has a correct digit label from 0 to 9.

## 3. Image preparation

- Convert pixel values from 0–255 to the 0–1 range.
- Add a channel dimension so images have the shape 28 × 28 × 1.
- During prediction, convert the drawing or uploaded image to grayscale, isolate foreground, and split a single-line number at vertical gaps. Resize and center each digit on its own 28 × 28 black image before classification.

## 4. CNN model

The network has the following layers:

1. 32-filter convolution layer with a 3 × 3 kernel and ReLU activation
2. Max pooling
3. 64-filter convolution layer with a 3 × 3 kernel and ReLU activation
4. Max pooling
5. Flatten the feature maps
6. Dropout (0.25)
7. Dense layer with 128 units and ReLU activation
8. Dropout (0.25)
9. Dense output layer with 10 units and softmax activation

The model uses the Adam optimizer and sparse categorical cross-entropy loss. It trains for five epochs by default, reserving 10% of the training data for validation. Dropout helps reduce overfitting.

## 5. Evaluation

After training, the model is evaluated on the separate 10,000-image MNIST test set. The command-line script prints test loss and test accuracy and saves training and validation accuracy/loss curves in `training_history.png`.

Record the measured result here after running `python train.py`:

- Test accuracy: __________________
- Test loss: __________________

## 6. How to run

On Windows, double-click `run_windows.bat`. The app uses the saved `mnist_cnn.keras` model and does not train when visitors open it. To train or retrain separately, run `python train.py` from PowerShell, then include the resulting model file in the deployed GitHub repository.

## 7. Conclusion

This project demonstrates how a CNN can learn visual patterns from labeled digit images and use those patterns to classify new examples. The web app supports individual digits and separated digits in a single-line number. Its predictions may be less reliable for handwriting that differs from MNIST, especially when digits touch or overlap.
