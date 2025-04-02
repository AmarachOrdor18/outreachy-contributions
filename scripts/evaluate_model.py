import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

class ModelEvaluation:
    def __init__(self, model_file, valid_file, logger=None):
        self.model = self.load_model(model_file)
        self.valid_df = pd.read_csv(valid_file)
        self.target_col = 'Y'
        self.non_feature_cols = ['Drug_ID', 'Drug', 'key', 'input', 'Y']
        self.X_valid = self.valid_df.drop(columns=self.non_feature_cols)
        self.y_valid = self.valid_df[self.target_col]
        self.figure_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'figures')
        os.makedirs(self.figure_dir, exist_ok=True)  # Ensure figures directory exists
        
        # Setup logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def load_model(self, model_file):
        try:
            model = joblib.load(model_file)
            print(f"Model loaded from {model_file}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file {model_file} not found.")
            raise

    def save_plot(self, filename):
        """Save the plot to the figures folder."""
        save_path = os.path.join(self.figure_dir, filename)
        plt.savefig(save_path)
        plt.close()  # Close the plot after saving to avoid memory issues
        self.logger.info(f"✅ Plot saved to {save_path}")
    
    def run_evaluation(self):
        # Making predictions
        y_pred = self.model.predict(self.X_valid)

        # Accuracy
        accuracy = accuracy_score(self.y_valid, y_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Classification Report
        report = classification_report(self.y_valid, y_pred)
        print("Classification Report:\n", report)

        # Confusion Matrix
        self.plot_confusion_matrix(y_pred)

        # ROC Curve
        self.plot_roc_curve()

        # Precision-Recall Curve
        self.plot_precision_recall_curve()

        # Feature Importance Plot
        self.plot_feature_importance()

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

    def plot_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y_valid, self.model.predict_proba(self.X_valid)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        self.save_plot("roc_curve.png")

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_valid, self.model.predict_proba(self.X_valid)[:, 1])
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color="green", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        self.save_plot("precision_recall_curve.png")

    def plot_feature_importance(self):
        # Extract feature importances from the model
        feature_importances = self.model.feature_importances_
        feature_names = self.X_valid.columns

        # Convert to DataFrame for better visualization
        feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
        feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(20)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(y=feat_imp_df["Feature"], x=feat_imp_df["Importance"], palette="viridis")
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Feature")
        plt.title("Top 20 Most Important Features in XGBoost Model")
        self.save_plot("feature_importance_top20.png")

# Run the evaluation
if __name__ == "__main__":
    model_file = 'models/best_xgboost_model.pkl'
    valid_file = 'data/bbb_valid_features.csv'
    
    model_evaluator = ModelEvaluation(model_file, valid_file)
    model_evaluator.run_evaluation()
