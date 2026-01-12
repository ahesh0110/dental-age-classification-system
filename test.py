import os
import cv2
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from skimage.feature import hog, graycomatrix, graycoprops
from dataclasses import dataclass
import pickle
import io

# ---------------------------
# Define Config class
# ---------------------------
@dataclass
class Config:
    img_size: int = 128
    crop_frac: float = 0.75
    random_crop_train: bool = True
    random_crop_size: float = 0.88

    hog_orientations: int = 9
    hog_pixels_per_cell: int = 16
    hog_cells_per_block: int = 2

    glcm_distances = (1, 3, 5)
    glcm_angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)

    cluster_pca_components: int = 30
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 3
    cluster_sample_limit: int = 1500

    use_pca: bool = True
    pca_components: int = 25
    use_selectkbest: bool = True
    selectkbest_k: int = 35

    max_adult_samples: int = 10000
    max_child_samples: int = 10000
    test_ratio: float = 0.30
    random_state: int = 42

# ---------------------------
# Custom unpickler that handles Config class
# ---------------------------
class ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Config':
            return Config
        return super().find_class(module, name)

def safe_load_model(filepath):
    """Load model with custom unpickler to handle Config class"""
    try:
        # Try standard joblib load first
        return joblib.load(filepath)
    except AttributeError as e:
        if 'Config' in str(e):
            # Use custom unpickler for Config class
            with open(filepath, 'rb') as f:
                return ConfigUnpickler(f).load()
        else:
            raise

# ---------------------------
# Helper to find the model file
# ---------------------------
def find_model_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "best_model.pkl"),
        os.path.join(script_dir, "dental_classification_results", "best_model.pkl"),
        os.path.join(os.path.dirname(script_dir), "dental_classification_results", "best_model.pkl"),
        r"C:\Users\zrela\OneDrive\Desktop\Codes\projects\dental_classification_results\best_model.pkl",
        os.path.join(os.path.expanduser("~"), "Desktop", "dental_classification_results", "best_model.pkl"),
    ]
    for path in possible_paths:
        if path and os.path.exists(path):
            return path

    # ask user
    root_temp = tk.Tk()
    root_temp.withdraw()
    selected = filedialog.askopenfilename(
        title="Select best_model.pkl",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    root_temp.destroy()
    return selected if selected else None

# ---------------------------
# Load model (safe)
# ---------------------------
MODEL_PATH = find_model_file()
if MODEL_PATH is None:
    messagebox.showerror("Error", "Model file not found. Please locate best_model.pkl")
    raise SystemExit("Model file not found")

try:
    model_data = safe_load_model(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model:\n{e}\n\nTry running the training script first.")
    raise

# Extract model components
model = model_data.get("model")
scaler = model_data.get("scaler")
pca = model_data.get("pca")
selector = model_data.get("selector")
class_names = model_data.get("class_names", ["class0", "class1"])
cfg_obj = model_data.get("config", None)

# Handle config object - convert to our Config class
if cfg_obj is None:
    CFG = Config()
    print("Warning: No config found in model, using defaults")
elif isinstance(cfg_obj, dict):
    # Create Config instance from dict
    CFG = Config()
    for k, v in cfg_obj.items():
        if hasattr(CFG, k):
            setattr(CFG, k, v)
elif isinstance(cfg_obj, Config):
    # Already the right type
    CFG = cfg_obj
else:
    # It's some other object - try to extract attributes
    CFG = Config()
    try:
        for attr in ['img_size', 'crop_frac', 'hog_orientations', 'hog_pixels_per_cell', 
                     'hog_cells_per_block', 'glcm_distances', 'glcm_angles']:
            if hasattr(cfg_obj, attr):
                setattr(CFG, attr, getattr(cfg_obj, attr))
    except Exception as e:
        print(f"Warning: Could not extract all config attributes: {e}")

# Print model info
print("="*60)
print("Model loaded:", MODEL_PATH)
try:
    best_result = model_data.get("best_result", {})
    if isinstance(best_result, dict):
        print("Classifier:", best_result.get("name", "N/A"))
    else:
        print("Classifier:", "N/A")
except Exception:
    print("Classifier:", "N/A")
print("Classes:", class_names)
print("="*60)

# ---------------------------
# Preprocessing (same as training)
# ---------------------------
def aggressive_crop(image, fraction):
    height, width = image.shape
    cropped_height = int(height * fraction)
    cropped_width = int(width * fraction)
    y_start = (height - cropped_height) // 2
    x_start = (width - cropped_width) // 2
    return image[y_start:y_start+cropped_height, x_start:x_start+cropped_width]

def preprocess_img(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image or empty: {file_path}")
    image = aggressive_crop(image, CFG.crop_frac)
    image = cv2.resize(image, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
    image = cv2.fastNlMeansDenoising(image, None, h=8, templateWindowSize=5, searchWindowSize=15)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

# ---------------------------
# Feature extraction (same as training)
# ---------------------------
def extract_hog(image):
    features = hog(
        image,
        orientations=CFG.hog_orientations,
        pixels_per_cell=(CFG.hog_pixels_per_cell, CFG.hog_pixels_per_cell),
        cells_per_block=(CFG.hog_cells_per_block, CFG.hog_cells_per_block),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    ).astype(np.float32)
    return features

def extract_glcm(image):
    quantized_image = (image / 16).astype(np.uint8)
    glcm_matrix = graycomatrix(
        quantized_image, 
        distances=CFG.glcm_distances, 
        angles=CFG.glcm_angles,
        levels=16,
        symmetric=True, 
        normed=True
    )
    properties = ["contrast", "homogeneity", "energy", "correlation"]
    features = []
    for property_name in properties:
        property_values = graycoprops(glcm_matrix, property_name)
        features.append(property_values.mean())
        features.append(property_values.std())
    return np.array(features, dtype=np.float32)

def extract_statistical_features(image):
    features = []
    features.append(np.mean(image))
    features.append(np.std(image))
    features.append(np.median(image))
    for p in [10,25,75,90]:
        features.append(np.percentile(image, p))
    features.append(np.min(image))
    features.append(np.max(image))
    hist, _ = np.histogram(image.ravel(), bins=16, range=(0, 256))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    for i in range(10):
        features.append(hist[i] if i < len(hist) else 0.0)
    hist_normalized = hist[hist > 0]
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized)) if hist_normalized.size > 0 else 0.0
    features.append(entropy)
    return np.array(features, dtype=np.float32)

def extract_edge_features(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edges = cv2.Canny(image, 50, 150)
    features = []
    features.append(np.mean(sobel_magnitude))
    features.append(np.std(sobel_magnitude))
    features.append(np.sum(edges > 0) / edges.size)
    return np.array(features, dtype=np.float32)

def build_feature_vector(image):
    hog_features = extract_hog(image)
    glcm_features = extract_glcm(image)
    stat_features = extract_statistical_features(image)
    edge_features = extract_edge_features(image)
    all_features = np.concatenate([hog_features, glcm_features, stat_features, edge_features], axis=0)
    return all_features

# ---------------------------
# Prediction function
# ---------------------------
def predict_image(img_path):
    image = preprocess_img(img_path)
    features = build_feature_vector(image).reshape(1, -1).astype(np.float32)

    if scaler is not None:
        features = scaler.transform(features)
    if pca is not None:
        features = pca.transform(features)
    if selector is not None:
        features = selector.transform(features)

    pred = model.predict(features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(features)
        proba = float(proba_arr[0, int(pred)]) if proba_arr is not None else None

    return class_names[int(pred)], proba

# ---------------------------
# GUI
# ---------------------------
root = tk.Tk()
root.title("Dental Age Classification System")
root.geometry("700x520")
root.resizable(False, False)

title_label = tk.Label(root, text="Dental Age Classifier", font=("Arial", 16, "bold"), fg="#2E86AB")
title_label.pack(pady=10)

info_text = f"Model loaded: {os.path.basename(MODEL_PATH)}"
tk.Label(root, text=info_text, font=("Arial", 10)).pack()

results_frame = tk.Frame(root)
results_frame.pack(padx=12, pady=8, fill=tk.BOTH, expand=True)

tk.Label(results_frame, text="Prediction Results:", font=("Arial", 11, "bold")).pack(anchor='w')

scrollbar = tk.Scrollbar(results_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

results_box = tk.Text(results_frame, height=18, width=85, yscrollcommand=scrollbar.set, font=("Courier", 9))
results_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=results_box.yview)

progress = ttk.Progressbar(root, length=650, mode='determinate')
progress.pack(pady=8)

status_label = tk.Label(root, text="Ready", font=("Arial", 9), fg="gray")
status_label.pack()

def classify_files(file_list):
    results_box.delete(1.0, tk.END)
    progress["value"] = 0

    valid_files = [
        f for f in file_list
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]

    total = len(valid_files)
    if total == 0:
        messagebox.showerror("Error", "No valid images found.")
        status_label.config(text="No valid images")
        return

    results_box.insert(tk.END, f"{'Filename':<45} {'Prediction':<20} {'Confidence':<10}\n")
    results_box.insert(tk.END, "="*95 + "\n")

    success_count = 0
    for i, path in enumerate(valid_files):
        status_label.config(text=f"Processing {i+1}/{total}: {os.path.basename(path)}")
        root.update_idletasks()
        try:
            label, prob = predict_image(path)
            filename = os.path.basename(path)
            if len(filename) > 42:
                filename = filename[:39] + "..."
            prob_str = f"{prob:.1%}" if prob is not None else "N/A"
            results_box.insert(tk.END, f"{filename:<45} {label:<20} {prob_str:<10}\n")
            success_count += 1
        except Exception as e:
            results_box.insert(tk.END, f"{os.path.basename(path):<45} ERROR: {e}\n")
        progress["value"] = ((i + 1) / total) * 100
        root.update_idletasks()

    results_box.insert(tk.END, "="*95 + "\n")
    results_box.insert(tk.END, f"Successfully processed: {success_count}/{total} images\n")
    status_label.config(text=f"Done: {success_count}/{total} processed")

def select_images():
    files = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
    )
    if files:
        classify_files(files)

def select_folder():
    folder = filedialog.askdirectory(title="Select Folder")
    if not folder:
        return
    files = []
    for root_dir, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                files.append(os.path.join(root_dir, f))
    if files:
        classify_files(files)
    else:
        messagebox.showwarning("Warning", "No image files found in the selected folder")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=12)

btn_style = {"width": 20, "height": 2, "font": ("Arial", 10, "bold"), "bg": "#2E86AB", "fg": "white", "relief": tk.RAISED}
tk.Button(btn_frame, text="📁 Select Images", command=select_images, **btn_style).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="📂 Select Folder", command=select_folder, **btn_style).grid(row=0, column=1, padx=10)

root.mainloop()