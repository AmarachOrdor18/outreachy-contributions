import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, make_scorer
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from tabulate import tabulate
import logging
from featurise import FeatureExtractor
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTraining:
    def __init__(self,
                 featurizer_id: str,
                 model_type: str,
                 cv_strategy: str,
                 target_col: str = "Y",
                 non_feature_cols: list = None):
        self.featurizer_id = featurizer_id
        self.target_col = target_col
        self.model_type = model_type
        self.cv_strategy = cv_strategy
        self.non_feature_cols = non_feature_cols or []

        # Get the parent directory of the current working directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True) 

        # Construct paths to the data files located in the parent directory
        self.train_file = os.path.join(parent_dir, f"data/bbb_train_{featurizer_id}_features.csv")
        self.test_file = os.path.join(parent_dir, f"data/bbb_test_{featurizer_id}_features.csv")
        self.val_file = os.path.join(parent_dir, f"data/bbb_valid_{featurizer_id}_features.csv")

        # Load the data
        self.X_train, self.y_train = self.load_data(self.train_file)
        self.X_test, self.y_test = self.load_data(self.test_file)
        self.X_val, self.y_val = self.load_data(self.val_file)

        # List the feature names
        self.feature_names = self.X_train.columns.tolist()

        # Preprocess the features
        self.preprocess_features()

    def load_data(self, file_path: str):
        df = pd.read_csv(file_path)
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col], errors="ignore")
        return X, y

    def preprocess_features(self):
        def clean(df):
            # Select only numeric columns
            df = df.select_dtypes(include=["number"])

            # Replace inf values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Clip values to avoid extreme values
            df = np.clip(df, -1e6, 1e6)

            # Fill NaN values with the median of each column
            if df.isnull().sum().sum() > 0:
                df.fillna(df.median(), inplace=True)
            return df

        # Clean training, validation, and test data
        self.X_train = clean(self.X_train)
        self.X_val = clean(self.X_val)
        self.X_test = clean(self.X_test)

        # Identify and remove zero-variance columns
        zero_columns = self.X_train.columns[self.X_train.nunique() == 1].tolist()
        self.X_train.drop(columns=zero_columns, inplace=True)
        self.X_val.drop(columns=zero_columns, inplace=True)
        self.X_test.drop(columns=zero_columns, inplace=True)

        logger.info(f"Removed {len(zero_columns)} zero-variance columns.")

        # Get standardized filenames
        filenames = self.generate_filenames(self.featurizer_id, self.model_type, self.cv_strategy)

        # Save feature columns (after zero-variance removal)
        feature_columns_path = os.path.join(self.models_dir, filenames["feature_columns"])
        with open(feature_columns_path, "w") as f:
            for col in self.X_train.columns.tolist():
                f.write(f"{col}\n")
        logger.info(f"Feature columns saved to: {feature_columns_path}")

        # Save zero-variance columns
        zero_variance_path = os.path.join(self.models_dir, filenames["zero_variance_columns"])
        with open(zero_variance_path, "w") as f:
            for col in zero_columns:
                f.write(f"{col}\n")
        logger.info(f"Zero-variance columns saved to: {zero_variance_path}")

        # Apply StandardScaler to the training, validation, and test sets
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        # Use standardized filenames
        filenames = self.generate_filenames(self.featurizer_id, self.model_type, self.cv_strategy)

        # Save the scaler
        scaler_path = os.path.join(self.models_dir, filenames["scaler"])
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ Scaler saved to: {scaler_path}")


    def apply_resampling(self):
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        logger.info("SMOTE resampling completed.")

        oversampler = RandomOverSampler(random_state=42)
        X_train_over, y_train_over = oversampler.fit_resample(self.X_train, self.y_train)
        logger.info("Random Oversampling completed.")

        smote_enn = SMOTEENN(random_state=42)
        X_train_hybrid, y_train_hybrid = smote_enn.fit_resample(self.X_train, self.y_train)
        logger.info("SMOTE + ENN Hybrid Resampling completed.")

        return X_train_smote, y_train_smote, X_train_over, y_train_over, X_train_hybrid, y_train_hybrid

    def tune_model(self, X_train_balanced, y_train_balanced):
        if self.model_type == "xgboost":
            model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                objective='binary:logistic',
            )
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }

        elif self.model_type == "random_forest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

        else:
            raise ValueError("Invalid model type. Choose 'xgboost' or 'random_forest'.")

        if self.cv_strategy == "grid_search":
            logger.info(f"🔍 Running GridSearchCV for {self.model_type.upper()}...")
            scorer = make_scorer(precision_score, pos_label=0, greater_is_better=True)

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scorer,
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(X_train_balanced, y_train_balanced)

            print(f"Best Hyperparameters: {grid_search.best_params_}")
            print(f"Best Class 0 Precision (CV): {grid_search.best_score_:.4f}")

            return grid_search.best_estimator_

        elif self.cv_strategy == "stratified_kfold":
            logger.info(f"🔁 Running manual StratifiedKFold for {self.model_type.upper()}...")
            best_score = 0
            best_model = None
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for params in ParameterGrid(param_grid):
                fold_precisions = []
                for train_idx, val_idx in skf.split(X_train_balanced, y_train_balanced):
                    X_fold_train, X_fold_val = X_train_balanced[train_idx], X_train_balanced[val_idx]
                    y_fold_train, y_fold_val = y_train_balanced[train_idx], y_train_balanced[val_idx]

                    model_clone = clone(model).set_params(**params)
                    model_clone.fit(X_fold_train, y_fold_train)
                    preds = model_clone.predict(X_fold_val)
                    fold_precisions.append(precision_score(y_fold_val, preds, pos_label=0))

                avg_precision = np.mean(fold_precisions)
                if avg_precision > best_score:
                    best_score = avg_precision
                    best_model = clone(model).set_params(**params)
                    best_model.fit(X_train_balanced, y_train_balanced)

            print(f"Best Class 0 Precision (manual CV): {best_score:.4f}")
            return best_model

        else:
            raise ValueError("Invalid CV strategy. Choose 'grid_search' or 'stratified_kfold'.")

    def compute_metrics(self, y_true, y_pred, y_proba, dataset_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else None

        print(f"\n{dataset_name} Metrics:")
        table = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["Precision (Weighted)", f"{precision:.4f}"],
            ["Recall (Weighted)", f"{recall:.4f}"],
            ["F1 Score (Weighted)", f"{f1:.4f}"]
        ]
        if roc_auc is not None:
            table.append(["ROC AUC", f"{roc_auc:.4f}"])

        print(tabulate(table, headers=["Metric", "Score"], tablefmt="fancy_grid"))
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_true, y_pred)
        }


    def make_predictions(self, X_raw, featurizer_id, model_type):
        try:
            # Load trained model, scaler, and feature columns
            filenames = self.generate_filenames(self.featurizer_id, self.model_type, self.cv_strategy)

            model_path = os.path.join(self.models_dir, filenames["model"])
            feature_file = os.path.join(self.models_dir, filenames["feature_columns"])
            scaler_path = os.path.join(self.models_dir, filenames["scaler"])
            zero_columns_file = os.path.join(self.models_dir, filenames["zero_variance_columns"])

            if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_file, zero_columns_file]):
                logger.error("❌ Model, scaler, or feature columns file is missing.")
                return None

            # Load the trained model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Load expected feature columns and zero-variance columns
            with open(feature_file, 'r') as f:
                expected_columns = [line.strip() for line in f.readlines()]

            with open(zero_columns_file, 'r') as f:
                zero_columns = [line.strip() for line in f.readlines()]

            # Preprocess input data (clean and align columns)
            X = X_raw.select_dtypes(include=["number"]).copy()
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = np.clip(X, -1e6, 1e6)
            X.fillna(X.median(), inplace=True)

            # Remove zero-variance columns (align with training)
            X.drop(columns=zero_columns, inplace=True)

            # Ensure columns match the training feature set
            input_columns = set(X.columns)
            expected_columns_set = set(expected_columns)

            missing_cols = list(expected_columns_set - input_columns)
            extra_cols = list(input_columns - expected_columns_set)

            if missing_cols:
                logger.warning(f"⚠️ Missing columns: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0  # Add missing columns with default value of 0

            if extra_cols:
                logger.warning(f"⚠️ Extra columns not used in training: {extra_cols}")
                X.drop(columns=extra_cols, inplace=True)

            # Reorder columns to match the training set's column order
            X = X[expected_columns]

            # Scale the features using the saved scaler
            X_scaled = scaler.transform(X)

            # Make prediction
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1]

            permeability = "Permeable" if y_pred[0] == 1 else "Not Permeable"
            probability = f"{y_proba[0]:.4f}"

            print("\n🧪 Prediction Result:")
            print(tabulate([
                ["Prediction", permeability],
                ["Probability", probability]
            ], headers=["Metric", "Value"], tablefmt="fancy_grid"))

            return {"prediction": permeability, "probability": y_proba[0]}

        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            return None

    def evaluate_model(self, model, threshold=0.75):
        y_train_pred = model.predict(self.X_train)
        y_train_proba = model.predict_proba(self.X_train)[:, 1]
        self.compute_metrics(self.y_train, y_train_pred, y_train_proba, "Train")

        y_val_pred = model.predict(self.X_val)
        y_val_proba = model.predict_proba(self.X_val)[:, 1]
        self.compute_metrics(self.y_val, y_val_pred, y_val_proba, "Validation")

        y_test_pred = model.predict(self.X_test)
        y_test_proba = model.predict_proba(self.X_test)[:, 1]
        self.compute_metrics(self.y_test, y_test_pred, y_test_proba, "Test")

    def save_model(self, model, featurizer_id, model_type, cv_strategy):
        # Get the absolute path to the models directory in the parent folder
        base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Use standardized filenames
        filenames = self.generate_filenames(featurizer_id, model_type, cv_strategy)

        # Save the trained model
        model_file_path = os.path.join(models_dir, filenames["model"])
        joblib.dump(model, model_file_path)
        print(f"✅ Trained model saved to: {model_file_path}")

        # Save the feature column names
        feature_file_path = os.path.join(models_dir, filenames["feature_columns"])
        with open(feature_file_path, 'w') as f:
            for col in self.feature_names:
                f.write(f"{col}\n")
        print(f"✅ Feature columns saved to: {feature_file_path}")


    def run(self):
        X_train_smote, y_train_smote, _, _, _, _ = self.apply_resampling()
        best_model = self.tune_model(X_train_smote, y_train_smote)
        self.evaluate_model(best_model)
        self.save_model(best_model, self.featurizer_id, self.model_type, self.cv_strategy)

    # Generate consistent filenames based on featurizer, model type, and CV strategy
    def generate_filenames(self, featurizer_id, model_type, cv_strategy):
        base_name = f"{featurizer_id}_{model_type}_{cv_strategy}"
        filenames = {
            "model": f"best_{base_name}_model.pkl",
            "scaler": f"{base_name}_scaler.pkl",
            "feature_columns": f"{base_name}_feature_columns.txt",
            "zero_variance_columns": f"{base_name}_zero_variance_columns.txt"
        }
        return filenames


if __name__ == "__main__":
    featurizer_id = "eos5axz"  # Change if running training
    model_type = "random_forest"
    cv_strategy = "stratified_kfold"

    # Initialize the ModelTraining class (training part)
    trainer = ModelTraining(
        featurizer_id=featurizer_id,
        model_type=model_type,
        cv_strategy=cv_strategy
    )
    trainer.run()

    # INFERENCE (prediction part)
    smiles = "CC(C)CCOCC"  # Example SMILES string
    featuriser = 'eos8a4x'
    
    # Use the FeatureExtractor to get features from the SMILES string
    featurizer = FeatureExtractor(featuriser)
    X = featurizer.featurize_smiles(smiles)

    if X is not None:
        # Initialize ModelTraining for prediction
        modelling = ModelTraining(featurizer_id=featuriser)

        # Now use the already trained scaler and model
        prediction = modelling.make_predictions(X, featuriser)

