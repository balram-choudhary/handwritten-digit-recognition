# Handwritten Digit Recognition

A polished Streamlit app that recognizes single digits and short, single-line numbers with a CNN trained on Keras's MNIST dataset. It separates digits from left to right, then classifies each with the same model. Draw on the canvas or upload a PNG/JPG image.

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

**Choose Python 3.13 in the deployment's Advanced settings.** TensorFlow 2.21.0 has wheels through Python 3.13, but not Python 3.14. The current Cloud app was created with Python 3.14, so simply pushing code or restarting it will not fix dependency installation. Streamlit Community Cloud does not let you change an app's Python version after deployment; deploy a new app with Python 3.13, or delete and redeploy the existing app with the same custom URL and Python 3.13 selected.

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

The app imports TensorFlow and loads the model only when a prediction is requested, then caches the model for the running session. It segments a single-line number at clear vertical gaps, converts each digit to grayscale, and centers each one in a 28 × 28 input. Leave space between handwritten digits; touching or unusually connected digits can be ambiguous. App code cannot prevent Streamlit Community Cloud from sleeping or cold-starting an app.
