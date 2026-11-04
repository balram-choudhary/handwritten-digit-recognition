# Handwritten Digit Recognition

This project uses a convolutional neural network (CNN) to recognize a single handwritten digit from 0 to 9. It uses the MNIST dataset built into Keras and has a small web app where you can draw or upload a digit.

## Easiest way to start on Windows

1. Double-click `run_windows.bat` in this folder.
2. The first time, it creates a Python environment in a short folder and installs the required packages. This avoids the Windows path length error that can happen when TensorFlow is installed inside this deeply nested project folder.
3. A browser tab opens with the app. If the model has not been trained yet, click **Train / retrain model** in the sidebar and wait for training to finish.
4. Draw one digit on the canvas and click **Predict drawing**, or upload a PNG/JPG image and click **Predict uploaded digit**.

The first setup needs an internet connection to download the packages and the MNIST dataset. Later launches reuse the environment and saved model.

## Start it from PowerShell instead

Open PowerShell in this folder and run:

```powershell
python -m venv "$env:LOCALAPPDATA\mnist-digit-recognition-venv"
& "$env:LOCALAPPDATA\mnist-digit-recognition-venv\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
streamlit run app.py
```

If you already made a virtual environment in this project folder and package installation failed with a Windows long path error, use `run_windows.bat` instead. It installs to the shorter environment path above.

## Train from the command line

You can also train the model directly:

```powershell
python train.py
```

Change the number of training passes through the MNIST training data with `--epochs`:

```powershell
python train.py --epochs 8
```

The script evaluates the trained model on MNIST's test set and saves `mnist_cnn.keras` and `training_history.png` beside the scripts. The Streamlit app loads the saved model when it is present.

## Files

- `app.py` — web app for drawing, uploading, and predicting digits; can train the model from the sidebar
- `train.py` — command-line training, evaluation, and learning-curve plotting
- `requirements.txt` — Python packages needed by the project
- `run_windows.bat` — Windows setup and app launcher
- `project_report.md` — short project explanation suitable for a presentation or submission

## How it works

MNIST provides 60,000 training images and 10,000 test images. Each image is a 28 × 28 grayscale picture with a label from 0 to 9. The project scales pixel values to 0–1, trains a CNN with two convolution and pooling stages, and uses the final ten output values as probabilities for the ten digits. The app converts your drawing or upload to a centered 28 × 28 grayscale image before prediction.

See `project_report.md` for the model and method in more detail. Predictions are estimates; clear images containing one digit tend to work best.
