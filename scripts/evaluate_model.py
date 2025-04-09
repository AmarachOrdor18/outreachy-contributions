import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from featurise import FeatureExtractor  # Assuming this is the correct import path

# Import ModelTraining and its method
from train_model import ModelTraining

class ModelEvaluation:
    def __init__(self, featurizer_id, model_type, cv_strategy, logger=None):
        # Get the parent directory (models folder is outside of the notebook folder)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Initialize the ModelTraining class and generate filenames using it
        model_training = ModelTraining(featurizer_id, model_type=model_type, cv_strategy=cv_strategy)
        filenames = model_training.generate_filenames(featurizer_id, model_type, cv_strategy)

        # Construct file paths relative to the 'models' folder in the parent directory
        self.featurizer_id = featurizer_id
        self.model_type = model_type
        self.cv_strategy = cv_strategy
        self.model_file = os.path.join(parent_dir, 'models', filenames["model"])
        self.valid_file = os.path.join(parent_dir, 'data', f"bbb_valid_{featurizer_id}_features.csv")  # Assuming this is still constant
        self.feature_file = os.path.join(parent_dir, 'models', filenames["feature_columns"])
        self.scaler_file = os.path.join(parent_dir, 'models', filenames["scaler"])

        # Initialize model and other objects
        self.model = self.load_model(self.model_file)
        self.scaler = joblib.load(self.scaler_file)
        self.feature_columns = self.load_feature_columns()

        self.valid_df = pd.read_csv(self.valid_file)
        self.target_col = 'Y'

        # Drop non-numeric columns before scaling
        self.valid_df = self.valid_df.select_dtypes(include=[np.number])

        # Filter out only feature columns present in both list and dataframe
        valid_columns = [col for col in self.feature_columns if col in self.valid_df.columns]
        missing_cols = set(self.feature_columns) - set(self.valid_df.columns)
        if missing_cols:
            print(f"⚠️ Missing feature columns in validation data: {missing_cols}")

        self.X_valid = self.valid_df[valid_columns]
        self.y_valid = self.valid_df[self.target_col]

        self.figure_dir = os.path.join(parent_dir, 'data', 'figures')
        os.makedirs(self.figure_dir, exist_ok=True)

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def load_model(self, model_file):
        try:
            model = joblib.load(model_file)
            print(f"✅ Model loaded from {model_file}")
            return model
        except FileNotFoundError:
            print(f"❌ Error: Model file {model_file} not found.")
            raise

    def save_plot(self, filename):
        # Modify the plot saving to include featurizer_id, model_type, and cv_strategy in the filename
        save_path = os.path.join(self.figure_dir, f"{self.featurizer_id}_{self.model_type}_{self.cv_strategy}_{filename}")
        plt.savefig(save_path)
        plt.show()  # Display the plot inline
        plt.close()
        self.logger.info(f"✅ Plot saved to {save_path}")

    def process_features(self, features_df):
        # Drop non-numeric columns to ensure consistent feature set
        numeric_features_df = features_df.select_dtypes(include=['float64', 'int64'])

        # Check if the processed DataFrame is empty
        if numeric_features_df.empty:
            self.logger.error("❌ No numeric features found in the data!")
            return None

        return numeric_features_df

    def run_evaluation(self):
        try:
            # Ensure the validation data has only numeric features (drop non-numeric columns)
            self.X_valid = self.process_features(self.X_valid)
            if self.X_valid is None:
                return

            # Scale features
            X_scaled = self.scaler.transform(self.X_valid)
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)[:, 1]

            accuracy = accuracy_score(self.y_valid, y_pred)
            print(f"🎯 Validation Accuracy: {accuracy:.4f}")

            report = classification_report(self.y_valid, y_pred)
            print("📋 Classification Report:\n", report)

            self.plot_confusion_matrix(y_pred)
            self.plot_roc_curve(y_proba)
            self.plot_precision_recall_curve(y_proba)
            self.plot_feature_importance()
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")

    def plot_confusion_matrix(self, y_pred):
        conf_matrix = confusion_matrix(self.y_valid, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        self.save_plot("confusion_matrix.png")
        plt.close()

    def plot_roc_curve(self, y_proba):
        fpr, tpr, _ = roc_curve(self.y_valid, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        self.save_plot("roc_curve.png")
        plt.show()

    def plot_precision_recall_curve(self, y_proba):
        precision, recall, _ = precision_recall_curve(self.y_valid, y_proba)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color="green", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        self.save_plot("precision_recall_curve.png")
        plt.show()

    def plot_feature_importance(self):
        try:
            # Check if the model supports feature importances
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_

                # Only include feature columns actually present in X_valid
                valid_feature_columns = [col for col in self.feature_columns if col in self.X_valid.columns]

                if len(importances) != len(valid_feature_columns):
                    raise ValueError(
                        f"Length mismatch: feature_importances_ = {len(importances)}, valid features = {len(valid_feature_columns)}"
                    )

                feat_imp_df = pd.DataFrame({
                    "Feature": valid_feature_columns,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False).head(10)  # Top 10

                plt.figure(figsize=(10, 6))
                sns.barplot(y=feat_imp_df["Feature"], x=feat_imp_df["Importance"], palette="viridis")
                plt.xlabel("Importance Score")
                plt.ylabel("Feature")
                plt.title("Top 10 Feature Importances")
                self.save_plot("feature_importance_top10.png")
                plt.show()

            else:
                self.logger.warning("⚠️ Model does not support feature importances.")

        except Exception as e:
            self.logger.error(f"Error while plotting feature importances: {str(e)}")

    def load_feature_columns(self):
        with open(self.feature_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    featurizer_id = "eos5axz"  # Change if running training
    model_type = "random_forest"
    cv_strategy = "stratified_kfold" # Example cross-validation strategy (e.g., kfold)

    # Step 1: Initialize the evaluator
    evaluator = ModelEvaluation(featurizer_id, model_type, cv_strategy)

    # Step 2: Run the evaluation
    evaluator.run_evaluation()
