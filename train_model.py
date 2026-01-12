import os
import cv2
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from dataclasses import dataclass
from collections import defaultdict
from hashlib import sha1

from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import DBSCAN

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)

import warnings
warnings.filterwarnings('ignore')

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
    test_ratio: float = 0.40  # Changed from 0.30 to 0.40
    random_state: int = 42

CFG = Config()

def aggressive_crop(image, fraction):
    height, width = image.shape
    cropped_height = int(height * fraction)
    cropped_width = int(width * fraction)
    y_start = (height - cropped_height) // 2
    x_start = (width - cropped_width) // 2
    return image[y_start:y_start+cropped_height, x_start:x_start+cropped_width]

def random_crop(image, fraction):
    height, width = image.shape
    cropped_height = int(height * fraction)
    cropped_width = int(width * fraction)
    y_start = np.random.randint(0, height - cropped_height + 1)
    x_start = np.random.randint(0, width - cropped_width + 1)
    return image[y_start:y_start+cropped_height, x_start:x_start+cropped_width]

def preprocess_img(file_path, is_training=False):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read: {file_path}")
    
    if image.size == 0:
        raise ValueError(f"Empty image: {file_path}")
    
    image = aggressive_crop(image, CFG.crop_frac)
    
    if is_training and CFG.random_crop_train:
        image = random_crop(image, CFG.random_crop_size)
    
    image = cv2.resize(image, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
    image = cv2.fastNlMeansDenoising(image, None, h=8, templateWindowSize=5, searchWindowSize=15)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

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
    
    num_features = len(features)
    feature_names = [f"HOG_feature_{i+1}" for i in range(num_features)]
    
    return features, feature_names

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
    feature_names = []
    
    for property_name in properties:
        property_values = graycoprops(glcm_matrix, property_name)
        features.append(property_values.mean())
        features.append(property_values.std())
        feature_names.append(f"GLCM_{property_name}_mean")
        feature_names.append(f"GLCM_{property_name}_std")
    
    return np.array(features, dtype=np.float32), feature_names

def extract_statistical_features(image):
    features = []
    feature_names = []
    
    features.append(np.mean(image))
    feature_names.append("Intensity_mean")
    
    features.append(np.std(image))
    feature_names.append("Intensity_std")
    
    features.append(np.median(image))
    feature_names.append("Intensity_median")
    
    percentiles = [10, 25, 75, 90]
    for p in percentiles:
        features.append(np.percentile(image, p))
        feature_names.append(f"Intensity_percentile_{p}")
    
    features.append(np.min(image))
    feature_names.append("Intensity_min")
    
    features.append(np.max(image))
    feature_names.append("Intensity_max")
    
    hist, _ = np.histogram(image.ravel(), bins=16, range=(0, 256))
    hist = hist.astype(np.float32) / hist.sum()
    
    for i in range(10):
        features.append(hist[i])
        feature_names.append(f"Histogram_bin_{i+1}")
    
    hist_normalized = hist[hist > 0]
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
    features.append(entropy)
    feature_names.append("Intensity_entropy")
    
    return np.array(features, dtype=np.float32), feature_names

def extract_edge_features(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    edges = cv2.Canny(image, 50, 150)
    
    features = []
    feature_names = []
    
    features.append(np.mean(sobel_magnitude))
    feature_names.append("Edge_gradient_mean")
    
    features.append(np.std(sobel_magnitude))
    feature_names.append("Edge_gradient_std")
    
    features.append(np.sum(edges > 0) / edges.size)
    feature_names.append("Edge_density")
    
    return np.array(features, dtype=np.float32), feature_names

def build_feature_vector(image):
    hog_features, hog_names = extract_hog(image)
    glcm_features, glcm_names = extract_glcm(image)
    stat_features, stat_names = extract_statistical_features(image)
    edge_features, edge_names = extract_edge_features(image)
    
    all_features = np.concatenate([hog_features, glcm_features, stat_features, edge_features], axis=0)
    all_names = hog_names + glcm_names + stat_names + edge_names
    
    return all_features, all_names

def sha1_of_file(file_path):
    hash_obj = sha1()
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def collect_image_paths(dataset_dir):
    classes = ["child", "adult"]
    image_paths = []
    image_labels = []
    
    print("\n[1] Collecting images from dataset...")
    print(f"    Dataset directory: {dataset_dir}")
    
    for class_index, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_dir, class_name)
        
        if not os.path.isdir(class_folder):
            raise FileNotFoundError(f"Missing folder: {class_folder}")
        
        valid_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', 
                           '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIF', '*.TIFF']
        
        found_images = []
        for ext in valid_extensions:
            pattern = os.path.join(class_folder, '**', ext)
            found_images.extend(glob(pattern, recursive=True))
        
        found_images = list(set(found_images))
        
        if len(found_images) == 0:
            raise FileNotFoundError(f"No valid images in {class_folder}")
        
        max_samples = CFG.max_child_samples if class_name == "child" else CFG.max_adult_samples
        if len(found_images) > max_samples:
            np.random.seed(CFG.random_state)
            found_images = list(np.random.choice(found_images, max_samples, replace=False))
        
        print(f"    {class_name}: {len(found_images)} images")
        
        image_paths.extend(found_images)
        image_labels.extend([class_index] * len(found_images))
    
    print(f"    Total: {len(image_paths)} images")
    return np.array(image_paths), np.array(image_labels), classes

def remove_exact_duplicates(image_paths, image_labels):
    seen_hashes = {}
    keep_mask = np.ones(len(image_paths), dtype=bool)
    removed_count = 0
    
    for index, path in enumerate(image_paths):
        try:
            file_hash = sha1_of_file(path)
            if file_hash in seen_hashes:
                keep_mask[index] = False
                removed_count += 1
            else:
                seen_hashes[file_hash] = path
        except Exception:
            continue
    
    if removed_count > 0:
        print(f"    Removed {removed_count} exact duplicates")
    
    return image_paths[keep_mask], image_labels[keep_mask]

def cluster_near_duplicates(image_paths, sample_limit=None):
    total_images = len(image_paths)
    selected_indices = np.arange(total_images)
    
    if sample_limit and sample_limit < total_images:
        np.random.seed(CFG.random_state)
        selected_indices = np.random.choice(total_images, sample_limit, replace=False)
    
    hog_features_list = []
    selected_paths = []
    
    for index in selected_indices:
        try:
            image = preprocess_img(image_paths[index], is_training=False)
            features, _ = extract_hog(image)
            hog_features_list.append(features)
            selected_paths.append(image_paths[index])
        except Exception:
            continue
    
    if len(hog_features_list) == 0:
        return {path: f"single_{idx}" for idx, path in enumerate(image_paths)}
    
    feature_matrix = np.vstack(hog_features_list)
    num_components = min(CFG.cluster_pca_components, feature_matrix.shape[1])
    pca_reducer = PCA(n_components=num_components, random_state=CFG.random_state)
    reduced_features = pca_reducer.fit_transform(feature_matrix)
    
    dbscan_clusterer = DBSCAN(eps=CFG.dbscan_eps, min_samples=CFG.dbscan_min_samples, n_jobs=-1)
    cluster_labels = dbscan_clusterer.fit_predict(reduced_features)
    
    path_to_cluster = {}
    for path, cluster_label in zip(selected_paths, cluster_labels):
        if cluster_label == -1:
            path_to_cluster[path] = f"noise_{hash(path) & 0xffffffff}"
        else:
            path_to_cluster[path] = f"cluster_{cluster_label}"
    
    for path in image_paths:
        if path not in path_to_cluster:
            path_to_cluster[path] = f"single_{hash(path) & 0xffffffff}"
    
    return path_to_cluster

def per_class_group_split(image_paths, image_labels, cluster_map, test_ratio=0.30, random_state=42):
    random_generator = np.random.RandomState(random_state)
    unique_classes = np.unique(image_labels)
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    for class_value in unique_classes:
        class_specific_paths = [path for path, label in zip(image_paths, image_labels) if label == class_value]
        
        path_groups = defaultdict(list)
        for path in class_specific_paths:
            group_id = cluster_map.get(path, f"solo_{hash(path)}")
            path_groups[group_id].append(path)
        
        total_class_images = len(class_specific_paths)
        target_train_count = int(total_class_images * (1.0 - test_ratio))
        
        group_items_list = list(path_groups.items())
        random_generator.shuffle(group_items_list)
        group_items_list.sort(key=lambda item: len(item[1]), reverse=True)
        
        selected_train_groups = set()
        current_train_count = 0
        
        for group_id, path_list in group_items_list:
            if current_train_count + len(path_list) <= target_train_count:
                selected_train_groups.add(group_id)
                current_train_count += len(path_list)
        
        for group_id, path_list in group_items_list:
            if group_id in selected_train_groups:
                train_paths.extend(path_list)
                train_labels.extend([class_value] * len(path_list))
            else:
                test_paths.extend(path_list)
                test_labels.extend([class_value] * len(path_list))
    
    return (np.array(train_paths), np.array(train_labels),
            np.array(test_paths), np.array(test_labels))

def build_features(image_paths, image_labels, is_training=False):
    feature_vectors = []
    valid_labels = []
    failed_count = 0
    feature_names = None
    
    total = len(image_paths)
    for idx, (path, label) in enumerate(zip(image_paths, image_labels)):
        if (idx + 1) % 100 == 0:
            print(f"    Progress: {idx + 1}/{total} images processed")
        try:
            image = preprocess_img(path, is_training=is_training)
            feature_vector, names = build_feature_vector(image)
            feature_vectors.append(feature_vector)
            valid_labels.append(label)
            if feature_names is None:
                feature_names = names
        except Exception:
            failed_count += 1
    
    if failed_count > 0:
        print(f"    Warning: {failed_count} images failed during feature extraction")
    
    if len(feature_vectors) == 0:
        raise RuntimeError("No features extracted")
    
    return np.vstack(feature_vectors).astype(np.float32), np.array(valid_labels, dtype=np.int64), feature_names

def perform_feature_selection(X_train, y_train, X_test, feature_names):
    results = {
        'selected_train': None,
        'selected_test': None,
        'scaler': None,
        'pca': None,
        'selector': None,
        'feature_names': feature_names.copy()
    }
    
    print("\n[4] Feature scaling and selection...")
    
    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X_test)
    results['scaler'] = scaler
    
    current_train = scaled_train
    current_test = scaled_test
    current_names = feature_names.copy()
    
    if CFG.use_pca:
        num_components = min(CFG.pca_components, current_train.shape[1])
        pca = PCA(n_components=num_components, random_state=CFG.random_state)
        pca_train = pca.fit_transform(current_train)
        pca_test = pca.transform(current_test)
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"    PCA: {num_components} components ({variance_explained*100:.2f}% variance)")
        results['pca'] = pca
        current_train = pca_train
        current_test = pca_test
        current_names = [f"PCA_component_{i+1}" for i in range(num_components)]
    
    if CFG.use_selectkbest:
        k = min(CFG.selectkbest_k, current_train.shape[1])
        selector = SelectKBest(f_classif, k=k)
        selected_train = selector.fit_transform(current_train, y_train)
        selected_test = selector.transform(current_test)
        print(f"    SelectKBest: selected top {k} features")
        results['selector'] = selector
        
        selected_indices = selector.get_support(indices=True)
        current_names = [current_names[i] for i in selected_indices]
        
        current_train = selected_train
        current_test = selected_test
    
    results['selected_train'] = current_train
    results['selected_test'] = current_test
    results['final_feature_names'] = current_names
    
    print(f"    Final feature dimension: {current_train.shape[1]}")
    
    return results

def get_classifiers():
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=CFG.random_state,
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=CFG.random_state
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=CFG.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=CFG.random_state,
            probability=True
        )
    }
    
    return classifiers

def evaluate_classifier(clf, clf_name, X_train, y_train, X_test, y_test, class_names):
    start_time = time.time()
    
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = 0.0
    except:
        roc_auc = 0.0
    
    results = {
        'classifier': clf,
        'name': clf_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'train_time': train_time,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"    {clf_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, Time={train_time:.2f}s")
    
    return results

def plot_confusion_matrices(results_list, class_names, output_dir):
    n_classifiers = len(results_list)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results_list):
        ax = axes[idx]
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names, cbar=True)
        ax.set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.4f}", fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_classifier_comparison(results_list, output_dir):
    names = [r['name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    f1_scores = [r['f1'] for r in results_list]
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    axes[0, 0].bar(names, accuracies, color=colors)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylim(0, 1.0)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
    
    axes[0, 1].bar(names, f1_scores, color=colors)
    axes[0, 1].set_ylabel('F1 Score', fontsize=11)
    axes[0, 1].set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.0)
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
    
    axes[1, 0].bar(names, precisions, color=colors)
    axes[1, 0].set_ylabel('Precision', fontsize=11)
    axes[1, 0].set_title('Precision Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylim(0, 1.0)
    for i, v in enumerate(precisions):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
    
    axes[1, 1].bar(names, recalls, color=colors)
    axes[1, 1].set_ylabel('Recall', fontsize=11)
    axes[1, 1].set_title('Recall Comparison', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0, 1.0)
    for i, v in enumerate(recalls):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classifier_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curves(results_list, y_test, output_dir):
    plt.figure(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, result in enumerate(results_list):
        if result['y_pred_proba'] is not None and result['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{result['name']} (AUC={result['roc_auc']:.3f})", 
                    linewidth=2.5, color=colors[idx])
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Classifier Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(best_result, feature_names, output_dir):
    clf = best_result['classifier']
    
    if not hasattr(clf, 'feature_importances_'):
        print("    Classifier does not support feature importance visualization")
        return
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(14, 8))
    plt.title(f'Top 20 Most Important Features - {best_result["name"]}', fontsize=14, fontweight='bold')
    plt.bar(range(len(indices)), importances[indices], color='#2E86AB')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()

def display_feature_importance(best_result, feature_names):
    clf = best_result['classifier']
    
    if not hasattr(clf, 'feature_importances_'):
        print("    Feature importance analysis not available for this classifier")
        return
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT FEATURES FOR CLASSIFICATION")
    print("="*80)
    print(f"\n{'Rank':<6} {'Feature Name':<40} {'Importance':<15} {'Impact':<10}")
    print("-"*80)
    
    for rank, idx in enumerate(indices[:20], 1):
        importance = importances[idx]
        feature_name = feature_names[idx]
        
        if importance > 0.10:
            impact = "Very High"
        elif importance > 0.05:
            impact = "High"
        elif importance > 0.02:
            impact = "Medium"
        else:
            impact = "Low"
        
        print(f"{rank:<6} {feature_name:<40} {importance:<15.6f} {impact:<10}")

def train_and_compare_classifiers(dataset_dir, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("DENTAL AGE CLASSIFICATION SYSTEM")
    print("="*80)
    
    image_paths, image_labels, class_names = collect_image_paths(dataset_dir)
    
    print("\n[2] Removing duplicates and clustering...")
    image_paths, image_labels = remove_exact_duplicates(image_paths, image_labels)
    cluster_map = cluster_near_duplicates(image_paths, CFG.cluster_sample_limit)
    
    print(f"\n[3] Train/test split ({100*(1-CFG.test_ratio):.0f}%/{100*CFG.test_ratio:.0f}%)...")
    train_paths, train_labels, test_paths, test_labels = per_class_group_split(
        image_paths, image_labels, cluster_map, CFG.test_ratio, CFG.random_state
    )
    print(f"    Training samples: {len(train_paths)}")
    print(f"    Testing samples: {len(test_paths)}")
    
    print("\n[5] Extracting features (HOG, GLCM, Statistical, Edge)...")
    train_features, train_labels, feature_names = build_features(train_paths, train_labels, is_training=True)
    test_features, test_labels, _ = build_features(test_paths, test_labels, is_training=False)
    print(f"    Raw feature dimension: {train_features.shape[1]}")
    print(f"    Raw feature names: {len(feature_names)}")
    
    feature_results = perform_feature_selection(train_features, train_labels, test_features, feature_names)
    X_train = feature_results['selected_train']
    X_test = feature_results['selected_test']
    final_feature_names = feature_results['final_feature_names']
    
    print("\n[6] Training and evaluating classifiers...")
    print("    Classifiers: Random Forest, Decision Tree, XGBoost, SVM")
    
    classifiers = get_classifiers()
    all_results = []
    
    for clf_name, clf in classifiers.items():
        result = evaluate_classifier(
            clf, clf_name, X_train, train_labels, X_test, test_labels, class_names
        )
        all_results.append(result)
    
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'Classifier':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    for result in all_results:
        print(f"{result['name']:<20} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} {result['f1']:<12.4f}")
    
    best_result = all_results[0]
    print("\n" + "="*80)
    print(f"BEST CLASSIFIER: {best_result['name']}")
    print("="*80)
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"  Precision:   {best_result['precision']:.4f}")
    print(f"  Recall:      {best_result['recall']:.4f}")
    print(f"  F1-Score:    {best_result['f1']:.4f}")
    print(f"  Sensitivity: {best_result['sensitivity']:.4f}")
    print(f"  Specificity: {best_result['specificity']:.4f}")
    print(f"  ROC AUC:     {best_result['roc_auc']:.4f}")
    print(f"  Training Time: {best_result['train_time']:.2f} seconds")
    
    cm = best_result['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()

    total = cm.sum()
    cm_percent = cm / total * 100

    print("\n" + "="*80)
    print("CONFUSION MATRIX (Counts and Percentages)")
    print("="*80)
    print(f"{'':>18}Predicted {class_names[0]:<10} Predicted {class_names[1]:<10}")
    print("-"*80)
    print(f"Actual {class_names[0]:<10} "
        f"{tn:6d} ({cm_percent[0,0]:5.1f}%)   "
        f"{fp:6d} ({cm_percent[0,1]:5.1f}%)")
    print(f"Actual {class_names[1]:<10} "
        f"{fn:6d} ({cm_percent[1,0]:5.1f}%)   "
        f"{tp:6d} ({cm_percent[1,1]:5.1f}%)")
    print("-"*80)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("="*80)

    
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, best_result['y_pred'], target_names=class_names))
    
    display_feature_importance(best_result, final_feature_names)
    
    print("\n[7] Generating visualizations...")
    plot_confusion_matrices(all_results, class_names, output_dir)
    plot_classifier_comparison(all_results, output_dir)
    plot_roc_curves(all_results, test_labels, output_dir)
    plot_feature_importance(best_result, final_feature_names, output_dir)
    print(f"    Saved visualizations to: {output_dir}/")
    
    print("\n[8] Saving model and results...")
    save_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump({
        "model": best_result['classifier'],
        "scaler": feature_results['scaler'],
        "pca": feature_results['pca'],
        "selector": feature_results['selector'],
        "config": CFG,
        "class_names": class_names,
        "feature_names": final_feature_names,
        "all_results": all_results,
        "best_result": best_result
    }, save_path)
    print(f"    Model saved to: {save_path}")
    
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DENTAL AGE CLASSIFICATION - COMPLETE RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration Parameters:\n")
        f.write(f"  Image size: {CFG.img_size}x{CFG.img_size}\n")
        f.write(f"  HOG orientations: {CFG.hog_orientations}\n")
        f.write(f"  GLCM angles: {len(CFG.glcm_angles)}\n")
        f.write(f"  GLCM distances: {len(CFG.glcm_distances)}\n")
        f.write(f"  PCA components: {CFG.pca_components}\n")
        f.write(f"  SelectKBest features: {CFG.selectkbest_k}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Total images: {len(image_paths)}\n")
        f.write(f"  Training samples: {len(train_paths)}\n")
        f.write(f"  Testing samples: {len(test_paths)}\n")
        f.write(f"  Class distribution:\n")
        for class_index, class_name in enumerate(class_names):
            count = int(np.sum(image_labels == class_index))
            f.write(f"    {class_name}: {count}\n")
        f.write("\n")
        
        f.write("Feature Engineering:\n")
        f.write(f"  Raw features: {train_features.shape[1]}\n")
        f.write(f"  Final features: {X_train.shape[1]}\n")
        f.write(f"  Feature extraction methods: HOG, GLCM, Statistical, Edge\n")
        f.write(f"  Feature selection: PCA + SelectKBest\n\n")
        
        f.write("Final Feature Names:\n")
        for i, name in enumerate(final_feature_names, 1):
            f.write(f"  {i}. {name}\n")
        f.write("\n")
        
        f.write("Classifier Performance:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Classifier':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time(s)':<10}\n")
        f.write("-"*80 + "\n")
        for result in all_results:
            f.write(f"{result['name']:<20} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
                   f"{result['recall']:<12.4f} {result['f1']:<12.4f} {result['train_time']:<10.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Best Classifier: {best_result['name']}\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy:    {best_result['accuracy']:.4f}\n")
        f.write(f"Precision:   {best_result['precision']:.4f}\n")
        f.write(f"Recall:      {best_result['recall']:.4f}\n")
        f.write(f"F1-Score:    {best_result['f1']:.4f}\n")
        f.write(f"Sensitivity: {best_result['sensitivity']:.4f}\n")
        f.write(f"Specificity: {best_result['specificity']:.4f}\n")
        f.write(f"ROC AUC:     {best_result['roc_auc']:.4f}\n")
        f.write(f"Training Time: {best_result['train_time']:.2f} seconds\n\n")
        
        f.write("Confusion Matrix:\n")
        cm = best_result['confusion_matrix']
        f.write(f"                    Predicted\n")
        f.write(f"                 {class_names[0]:<8} {class_names[1]:<8}\n")
        f.write(f"  Actual {class_names[0]:<8} {cm[0,0]:6d}   {cm[0,1]:6d}\n")
        f.write(f"         {class_names[1]:<8} {cm[1,0]:6d}   {cm[1,1]:6d}\n\n")
        
        if hasattr(best_result['classifier'], 'feature_importances_'):
            f.write("="*80 + "\n")
            f.write("TOP 20 MOST IMPORTANT FEATURES\n")
            f.write("="*80 + "\n")
            importances = best_result['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1]
            f.write(f"{'Rank':<6} {'Feature Name':<40} {'Importance':<15} {'Impact':<10}\n")
            f.write("-"*80 + "\n")
            for rank, idx in enumerate(indices[:20], 1):
                importance = importances[idx]
                feature_name = final_feature_names[idx]
                if importance > 0.10:
                    impact = "Very High"
                elif importance > 0.05:
                    impact = "High"
                elif importance > 0.02:
                    impact = "Medium"
                else:
                    impact = "Low"
                f.write(f"{rank:<6} {feature_name:<40} {importance:<15.6f} {impact:<10}\n")
    
    print(f"    Report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    return all_results, best_result

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\zrela\OneDrive\Desktop\dental_dataset"
    OUTPUT_DIR = "dental_classification_results"
    
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        exit(1)
    
    child_dir = os.path.join(DATASET_DIR, "child")
    adult_dir = os.path.join(DATASET_DIR, "adult")
    
    if not os.path.exists(child_dir) or not os.path.exists(adult_dir):
        print(f"ERROR: Missing child/ or adult/ folders in dataset")
        exit(1)
    
    try:
        all_results, best_result = train_and_compare_classifiers(DATASET_DIR, OUTPUT_DIR)
    except Exception as error:
        print(f"ERROR: {error}")
        import traceback
        traceback.print_exc()
        exit(1)