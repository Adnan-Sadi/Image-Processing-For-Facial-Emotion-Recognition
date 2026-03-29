import os
import numpy as np
import json
import time
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import config, RFConfig, SVMConfig

def main():
    # load the extracted features from the npz file
    features_path = os.path.join(config.feature_dir, "dataset_features.npz")
    data = np.load(features_path, allow_pickle=True)

    # prepare the feature matrix X and label vector y
    selected_features = [data[feature_name] for feature_name in config.features_to_use]
    X = np.concatenate(selected_features, axis=1)
    y = data["labels"]
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")

    K = config.num_features_to_select
    if K:
        selector = SelectKBest(k=K)
        X = selector.fit_transform(X, y)


    scoring_metrics = {
    "accuracy": "accuracy",
    "precision": "precision_macro", # macro average score for multi-class classification
    "recall": "recall_macro", # macro average score for multi-class classification
    "f1": "f1_macro", # macro average score for multi-class classification
    }
    start_time = time.time()

    # train a random forest classifier
    if config.model == "random_forest":
        rf_config = RFConfig()
        clf = RandomForestClassifier(
            n_estimators=rf_config.n_estimators,
            max_features=rf_config.max_features,
            max_depth=rf_config.max_depth,
            min_samples_split=rf_config.min_samples_split,
            criterion=rf_config.criterion,
            random_state=rf_config.random_state
        )
    elif config.model == "svm":
        svm_config = SVMConfig()
        clf = Pipeline([
            ("scaler", StandardScaler()), # scaling is needed for SVM
            ("svc", SVC(
                kernel=svm_config.kernel,
                C=svm_config.C,
                gamma=svm_config.gamma,
                random_state=svm_config.random_state,
            )),
        ])
    else:
        raise ValueError(f"Unsupported model type: {config.model}")

    scores = cross_validate(clf, X, y, cv=5,  scoring=scoring_metrics, return_train_score=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print and save results as json
    results = {
        "model": config.model,
        "input_size": config.input_size,
        "hog": {
            "input_size": config.hog_input_size,
            "orientations": config.hog_orientations,
            "pixels_per_cell": config.hog_pixels_per_cell,
            "cells_per_block": config.hog_cells_per_block,
        },
        "features_used": config.features_to_use,
        "scores": {
            "train_accuracy": scores["train_accuracy"].mean(),
            "test_accuracy": scores["test_accuracy"].mean(),
            "train_precision": scores["train_precision"].mean(),
            "test_precision": scores["test_precision"].mean(),
            "train_recall": scores["train_recall"].mean(),
            "test_recall": scores["test_recall"].mean(),
            "train_f1": scores["train_f1"].mean(),
            "test_f1": scores["test_f1"].mean(),
        },
        "training_time_seconds": elapsed_time,
    }

    # add model hyperparameters to results
    if config.model == "random_forest":
        results["model_hyperparameters"] = {
            "n_estimators": rf_config.n_estimators,
            "max_features": rf_config.max_features,
            "max_depth": rf_config.max_depth,
            "min_samples_split": rf_config.min_samples_split,
            "criterion": rf_config.criterion,
        }
    elif config.model == "svm":
        results["model_hyperparameters"] = {
            "kernel": svm_config.kernel,
            "C": svm_config.C,
            "gamma": svm_config.gamma,
        }

    if not os.path.exists(config.ml_result_save_path):
        os.makedirs(config.ml_result_save_path)

    with open(os.path.join(config.ml_result_save_path, config.ml_result_filename), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Crross-validation completed in {elapsed_time:.2f} seconds.")
    print(f"Model Test Accuracy: {results['scores']['test_accuracy']:.4f}")
    print(f"Model Test F1 Score: {results['scores']['test_f1']:.4f}")
    print(f"Results saved to: {os.path.join(config.ml_result_save_path, config.ml_result_filename)}")

    # save the trained model using joblib
    clf.fit(X, y) # fit the model on the entire dataset
    model_save_path = os.path.join(config.ml_result_save_path, f"{config.model}_model.joblib")
    joblib.dump(clf, model_save_path)
    print(f"Trained model saved to: {model_save_path}")

if __name__ == "__main__":
    main()