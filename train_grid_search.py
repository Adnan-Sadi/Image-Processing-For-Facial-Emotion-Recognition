import os
from matplotlib.pyplot import clf
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
from config import config, RFConfig, SVMConfig, GridSearchConfig

def get_model():
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
        return clf, rf_config
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
        return clf, svm_config
    else:
        raise ValueError(f"Unsupported model type: {config.model}")

def main():
    # load the extracted features from the npz file
    features_path = os.path.join(config.feature_dir, "dataset_features.npz")
    data = np.load(features_path, allow_pickle=True)

    # get the model and its config
    clf, model_config = get_model()
    # define scoring metrics for cross-validation
    scoring_metrics = {
    "accuracy": "accuracy",
    "precision": "precision_macro", # macro average score for multi-class classification
    "recall": "recall_macro", # macro average score for multi-class classification
    "f1": "f1_macro", # macro average score for multi-class classification
    }


    grid_search_config = GridSearchConfig()
    selected_feature_combinations = grid_search_config.features_combinations
    num_feat_to_select_combinations = grid_search_config.num_features_to_select_combinations
    start_time = time.time()

    # store results for all combinations of features, best-to-worst
    results = []

    for K in num_feat_to_select_combinations:
        for feature_combination in selected_feature_combinations:
            print(f"Training model with features: {feature_combination} and K={K} features selected")
             # prepare the feature matrix X and label vector y
            selected_features = [data[feature_name] for feature_name in feature_combination]
            X = np.concatenate(selected_features, axis=1)
            y = data["labels"]

            if K:
                selector = SelectKBest(k=K)
                X = selector.fit_transform(X, y)

            scores = cross_validate(clf, X, y, cv=5,  scoring=scoring_metrics, return_train_score=True)
            results.append({
                "features": feature_combination,
                "K": K,
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
            })


    # sort results by test accuracy score in descending order
    results.sort(key=lambda x: x["scores"]["test_accuracy"], reverse=True)
    elapsed_time = time.time() - start_time

    best_model = results[0]

    # make results directory if it doesn't exist
    if not os.path.exists(config.ml_result_save_path):
        os.makedirs(config.ml_result_save_path)

    # save results as json
    with open(os.path.join(config.ml_result_save_path, config.ml_result_filename), "w") as f:
        json.dump(results, f, indent=4)

    # save model config in a separate json file
    model_config = {
        "model": config.model,
        "model_config": model_config.__dict__, # convert the class object to dict
    }
    with open(os.path.join(config.ml_result_save_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    print(f"Grid-Search Cross-validation completed in {elapsed_time:.2f} seconds.")
    print(f"Model Test Accuracy: {best_model['scores']['test_accuracy']:.4f}")
    print(f"Model Test F1 Score: {best_model['scores']['test_f1']:.4f}")
    print(f"Results saved to: {os.path.join(config.ml_result_save_path, config.ml_result_filename)}")

    # save the best model using joblib
    best_features = best_model["features"]
    X = np.concatenate([data[feature_name] for feature_name in best_features], axis=1)
    y = data["labels"]
    if best_model["K"]:
        selector = SelectKBest(k=best_model["K"])
        X = selector.fit_transform(X, y)
    clf.fit(X, y) # fit the model on the entire dataset
    model_save_path = os.path.join(config.ml_result_save_path, f"{config.model}_best_model.joblib")
    joblib.dump(clf, model_save_path)
    print(f"Best model saved to: {model_save_path}")

if __name__ == "__main__":
    main()