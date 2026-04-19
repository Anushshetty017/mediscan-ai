# MediScan AI — Setup & Run Guide (Windows + VS Code)

A Flask web app for Brain Tumor MRI and Skin Cancer classification using EfficientNet-B0.

---

## Folder Structure

```
project/
├── app.py                  ← Flask backend
├── requirements.txt        ← Python dependencies
├── models/
│   ├── brain_model.pth     ← Your trained brain tumor model
│   └── skin_model.pth      ← Your trained skin cancer model
├── templates/
│   └── index.html          ← Frontend UI
└── static/
    └── style.css           ← Stylesheet
```

---

## Step-by-Step Instructions (Windows + VS Code)

### STEP 1 — Install Python (if not installed)

1. Go to https://www.python.org/downloads/
2. Download Python 3.10 or higher
3. During installation, **check "Add Python to PATH"**
4. Click Install Now

Verify installation — open a terminal and run:
```
python --version
```

---

### STEP 2 — Open Project in VS Code

1. Open VS Code
2. Go to **File → Open Folder**
3. Select your `project/` folder
4. VS Code will open the project

---

### STEP 3 — Open the Integrated Terminal

In VS Code:
- Press **Ctrl + `** (backtick key, below Escape)
- OR go to **Terminal → New Terminal**

---

### STEP 4 — Create a Virtual Environment

In the terminal, run:

```bash
python -m venv venv
```

This creates a `venv/` folder inside your project.

---

### STEP 5 — Activate the Virtual Environment

```bash
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

> If you get a permissions error, run this first:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> Then try activating again.

---

### STEP 6 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, PyTorch, EfficientNet, Pillow, etc.

> If you have an NVIDIA GPU and want GPU acceleration:
> Visit https://pytorch.org/get-started/locally/ to get the correct CUDA install command.

---

### STEP 7 — Add Your Trained Models

Place your saved `.pth` files in the `models/` folder:

```
project/
└── models/
    ├── brain_model.pth   ← from your watershed.py training
    └── skin_model.pth    ← your skin cancer model
```

> The model must have been saved using `torch.save(model.state_dict(), ...)`.
> The number of output classes must match:
>   - Brain: 4 classes (glioma, meningioma, notumor, pituitary)
>   - Skin : 7 classes (ISIC categories)
> Edit `BRAIN_CLASSES` and `SKIN_CLASSES` in `app.py` if your labels differ.

---

### STEP 8 — Run the Flask App

```bash
python app.py
```

You should see output like:
```
Running on: cpu
[INFO] Loaded model from models/brain_model.pth
 * Running on http://127.0.0.1:5000
```

---

### STEP 9 — Open in Browser

Open your browser and go to:

```
http://127.0.0.1:5000
```

You'll see the MediScan AI interface!

---

### STEP 10 — Use the App

1. Select **Brain Tumor** or **Skin Cancer** from the toggle
2. Upload a medical image (PNG or JPG)
3. Click **Analyse Image**
4. See the predicted class and confidence score

---

## Stopping the Server

In the terminal, press **Ctrl + C** to stop Flask.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure `(venv)` is active and run `pip install -r requirements.txt` |
| Model not loaded warning | Check that `.pth` files are in the `models/` folder |
| Port already in use | Change `port=5000` to `port=5001` in `app.py` |
| Slow on CPU | Normal — EfficientNet inference on CPU takes 1–3 seconds |
| Wrong number of classes | Edit `BRAIN_CLASSES` / `SKIN_CLASSES` lists in `app.py` |

---

## Customising Class Labels

Open `app.py` and edit these lists to match your training labels exactly:

```python
BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SKIN_CLASSES  = ["actinic keratosis", "basal cell carcinoma", ...]
```

The order must match the folder order used by `ImageFolder` during training.

---

## Notes

- This app is for **educational/research purposes only**
- Always consult a qualified medical professional for diagnosis
- GPU (CUDA) is automatically used if available
