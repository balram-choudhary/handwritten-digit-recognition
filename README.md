# Handwritten Digit Recognition

A polished Streamlit app that recognizes one handwritten digit (0–9) with a CNN trained on Keras's MNIST dataset. Draw on the canvas or upload a PNG/JPG image.

## Run on Windows

Double-click `run_windows.bat`. It creates a Python environment in a short path, installs the pinned dependencies, and starts the app. The first install takes longer because TensorFlow is a large package.

The saved `mnist_cnn.keras` model is used for predictions. If you need to create or retrain it, install the training-only plotting dependency with `python -m pip install -r requirements-training.txt`, then run `python train.py`. Visitors to the app cannot start training.

## Run from PowerShell

Open PowerShell in this folder:

```powershell
python -m venv "$env:LOCALAPPDATA\mnist-digit-recognition-venv"
& "$env:LOCALAPPDATA\mnist-digit-recognition-venv\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Put these project files at the root of a GitHub repository, including `app.py`, `requirements.txt`, and the trained `mnist_cnn.keras` file. Do not upload `.venv`. In Streamlit Community Cloud, create an app from the repository and choose `app.py` on the `main` branch.

`mnist_cnn.keras` is intentionally included in Git. Without it the deployed app displays a setup message and predictions stay disabled until you train and add the model.

## Project files

- `app.py` — responsive dark UI, drawing and upload inputs, preprocessing, and prediction results
- `train.py` — separate MNIST training, evaluation, and training-curve generation
- `mnist_cnn.keras` — saved CNN used by the app
- `requirements.txt` — pinned app dependencies used by Streamlit Community Cloud
- `requirements-training.txt` — extra plotting dependency for local training
- `project_report.md` — explanation of the dataset, CNN, and evaluation
- `run_windows.bat` — Windows setup and launcher

## Notes

The app imports TensorFlow and loads the model only when a prediction is requested, then caches the model for the running session. The image is converted to grayscale, foreground is isolated, and the digit is resized and centered to 28 × 28 before prediction. App code can reduce its own initialization work, but it cannot prevent Streamlit Community Cloud from sleeping or cold-starting an app.
