import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing

def plot_train_classwise_pca_correlation(X_train_pca, y_train, label_map, n_components=20):
    """
    Plots a correlation matrix of PCA features for each class.

    Args:
        X_train_pca (np.ndarray): Array of shape (N, T, n_components).
        y_train (np.ndarray): Labels for training samples.
        label_map (dict): Mapping from class names to numeric labels.
        n_components (int): Number of PCA components per time step.
    """
    inv_label_map = {v: k for k, v in label_map.items()}

    for class_id in sorted(np.unique(y_train)):
        class_name = inv_label_map.get(class_id, f"Class {class_id}")
        class_samples = X_train_pca[y_train == class_id]

        if class_samples.size == 0:
            print(f"No samples found for class {class_name}")
            continue

        flat = class_samples.reshape(-1, n_components)

        df = pd.DataFrame(flat, columns=[f"PCA{i+1}" for i in range(n_components)])
        corr = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=True,
                    cbar_kws={"label": "Correlation Coefficient"})
        plt.title(f"Correlation Matrix (Train PCA) — {class_name}")
        plt.tight_layout()
        plt.show()

def extract_top_correlated_features_per_class(X_train_pca, y_train, label_map, n_components=20, top_n=5):
    """
    Extracts the most correlated PCA feature pairs per class in training data.

    Args:
        X_train_pca (np.ndarray): Shape (N, T, n_components).
        y_train (np.ndarray): Class labels for X_train_pca.
        label_map (dict): Mapping from class names to numeric labels.
        n_components (int): Number of PCA components.
        top_n (int): Number of top correlated feature pairs to return per class.

    Returns:
        dict: Maps each class name to a list of tuples.
    """
    inv_label_map = {v: k for k, v in label_map.items()}
    result = {}

    for class_id in sorted(np.unique(y_train)):
        class_name = inv_label_map.get(class_id, f"Class {class_id}")
        class_samples = X_train_pca[y_train == class_id]

        if class_samples.size == 0:
            print(f"No samples for {class_name}")
            continue

        flat = class_samples.reshape(-1, n_components)
        df = pd.DataFrame(flat, columns=[f"PCA{i+1}" for i in range(n_components)])
        corr = df.corr().values

        pairs = []
        for i in range(n_components):
            for j in range(i + 1, n_components):
                value = corr[i, j]
                pairs.append((f"PCA{i+1}", f"PCA{j+1}", value))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        result[class_name] = pairs[:top_n]

        print(f"\nTop {top_n} correlated pairs in {class_name}:")
        for a, b, val in result[class_name]:
            print(f"  {a} - {b} = {val:.2f}")

    return result

def save_correlated_features(correlated_features, output_path="Applied-ML-Group-7/project_name/data/manually_extracted_features/most_correlated_features.csv"):
    """
    Saves the most correlated feature pairs per class to a CSV file.

    Args:
        correlated_features (dict): Output from the function extract_top_correlated_features_per_class.
        output_path (str): Path to the CSV to save the most correlated features.
    """
    rows = []
    for class_name, pairs in correlated_features.items():
        for feature1, feature2, correlation in pairs:
            rows.append({
                "class": class_name,
                "feature_1": feature1,
                "feature_2": feature2,
                "correlation": round(correlation, 4)

            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Correlated features saved to: {output_path}")

if __name__ == "__main__":

    label_map = {
        "Airplane": 0,
        "Bics": 1,
        "Cars": 2,
        "Helicopter": 3,
        "Motocycles": 4,
        "Train": 5,
        "Truck": 6,
        "bus": 7
    }
    
    p = Preprocessing(0.7, 0.15, 0.15, 48000)

    train_spec = "Applied-ML-Group-7/project_name/data/spectograms/train"
    train_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/train"
    X_spec_train, X_manual_train, y_spec_train, y_manual_train = p.load_dual_inputs(train_spec, train_manual)

    valid_spec = "Applied-ML-Group-7/project_name/data/spectograms/valid"
    valid_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/valid"
    X_spec_valid, X_manual_valid, y_spec_valid, y_manual_valid = p.load_dual_inputs(valid_spec, valid_manual)

    test_spec = "Applied-ML-Group-7/project_name/data/spectograms/test"
    test_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/test"
    X_spec_test, X_manual_test, y_spec_test, y_manual_test = p.load_dual_inputs(test_spec, test_manual) 
    	
    X_train_pca, X_valid_pca, X_test_pca = p.apply_pca(X_manual_train, X_manual_valid, X_manual_test)

    plot_train_classwise_pca_correlation(X_train_pca, y_manual_train, label_map)
    corrrelated_features = extract_top_correlated_features_per_class(
        X_train_pca, y_manual_train, label_map, n_components=20, top_n=5
    )
    save_correlated_features(corrrelated_features)