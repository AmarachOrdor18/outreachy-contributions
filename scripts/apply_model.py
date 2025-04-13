import os
import pandas as pd
import numpy as np
import joblib
import logging
import subprocess
import time
from tabulate import tabulate
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


class ApplyTrainedModel:
    def __init__(self, data_path, featurizer_id, model_type, cv_strategy):
        self.featurizer_id = featurizer_id
        self.model_type = model_type
        self.cv_strategy = cv_strategy

        # Dynamically set base directory based on script location
        self.base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

        # Set model and data directories relative to the base directory
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "data")

        # Now that self.data_dir exists, we can safely resolve self.data_path
        self.data_path = os.path.join(self.data_dir, os.path.basename(data_path))

        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.filenames = self._generate_filenames()

        # File paths
        self.model_file = os.path.join(self.models_dir, self.filenames["model"])
        self.scaler_file = os.path.join(self.models_dir, self.filenames["scaler"])
        self.feature_columns_file = os.path.join(self.models_dir, self.filenames["feature_columns"])
        self.zero_variance_file = os.path.join(self.models_dir, self.filenames["zero_variance_columns"])

        # Load model components
        self.model = joblib.load(self.model_file)
        self.scaler = joblib.load(self.scaler_file)
        self.expected_columns = self._load_txt(self.feature_columns_file)
        self.zero_variance_columns = self._load_txt(self.zero_variance_file)

    def _generate_filenames(self):
        base = f"{self.featurizer_id}_{self.model_type}_{self.cv_strategy}"
        return {
            "model": f"best_{base}_model.pkl",
            "scaler": f"{base}_scaler.pkl",
            "feature_columns": f"{base}_feature_columns.txt",
            "zero_variance_columns": f"{base}_zero_variance_columns.txt"
        }

    def _load_txt(self, filepath):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _clean_features(self, X):
        # Track dropped columns
        dropped_columns = []

        # Drop columns with zero variance
        X.drop(columns=self.zero_variance_columns, errors="ignore", inplace=True)
        dropped_columns.extend(self.zero_variance_columns)

        X = X.select_dtypes(include=["number"])
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = np.clip(X, -1e6, 1e6)
        X.fillna(X.median(), inplace=True)

        # Add missing columns
        for col in self.expected_columns:
            if col not in X.columns:
                X[col] = 0

        # Drop extra columns and track dropped columns
        extra_columns = [col for col in X.columns if col not in self.expected_columns]
        X.drop(columns=extra_columns, inplace=True)
        dropped_columns.extend(extra_columns)

        # Ensure columns are in the same order as expected during training
        X = X[self.expected_columns]

        # Log the number of columns dropped and columns added
        self.logger.info(f"📉 Dropped {len(dropped_columns)} columns: {', '.join(dropped_columns)}")
        self.logger.info(f"➕ Added missing columns: {', '.join([col for col in self.expected_columns if col not in X.columns])}")

        return X

    def _generate_features(self):
        """
        Featurizes the input data using the CLI method (Ersilia CLI).
        """
        try:
            output_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(output_dir, exist_ok=True)

            self.logger.info(f"Fetching model '{self.featurizer_id}' from Ersilia...")
            # Run command to fetch the featurizer model
            fetch_command = f"ersilia -v fetch {self.featurizer_id}"
            fetch_result = subprocess.run(fetch_command, shell=True, check=True)
            if fetch_result.returncode != 0:
                self.logger.error(f"Error fetching model: {fetch_result.stderr}")
                return None

            self.logger.info("Serving the model in the background...")
            # Run command to serve the featurizer model
            serve_command = f"ersilia -v serve {self.featurizer_id}"
            serve_result = subprocess.run(serve_command, shell=True, check=True)
            if serve_result.returncode != 0:
                self.logger.error(f"Error serving model: {serve_result.stderr}")
                return None

            self.logger.info("Waiting 20 seconds for the model to fully start...")
            time.sleep(20)

            self.logger.info(f"Running feature extraction on {self.data_path}...")
            output_file = self.data_path.replace(".csv", "_featurized.csv")
            run_command = f"ersilia -v run -i {self.data_path} -o {output_file}"
            run_result = subprocess.run(run_command, shell=True, check=True)

            if run_result.returncode != 0:
                self.logger.error(f"Error during feature extraction: {run_result.stderr}")
                return None

            self.logger.info(f"Feature extraction complete! Data saved to {output_file}")
            return pd.read_csv(output_file)

        except Exception as e:
            self.logger.error(f"Error during feature extraction: {e}", exc_info=True)
            return None

    def run(self):
        self.logger.info(f"📁 Loading input data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        if "Drug" not in df.columns:
            self.logger.error("❌ 'Drug' column not found in input data.")
            return None

        self.logger.info("🔬 Featurizing unseen data...")
        featurized_data = self._generate_features()

        if featurized_data is None:
            self.logger.error("❌ Feature extraction failed.")
            return None

        self.logger.info("🧹 Cleaning and scaling features...")
        X_cleaned = self._clean_features(featurized_data)
        X_scaled = self.scaler.transform(X_cleaned)

        self.logger.info("🤖 Running predictions...")
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        results = df.copy()
        results["Prediction"] = ["Permeable" if p == 1 else "Not Permeable" for p in y_pred]
        results["Probability"] = y_proba

        # Define the output file path dynamically
        dataset_name = os.path.splitext(os.path.basename(self.data_path))[0]
        result_filename = os.path.join(self.data_dir, f"{dataset_name}_predictions_{self.featurizer_id}_{self.model_type}_{self.cv_strategy}.csv")

        self.logger.info(f"✅ Prediction complete. Saving results to {result_filename}")
        results.to_csv(result_filename, index=False)

        # Optional: Compute classification report if true labels are available
        if "Y" in df.columns:
            y_true = df["Y"]

            # Classification report
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            self.logger.info("📊 Classification Report:")
            print(tabulate(report_df, headers='keys', tablefmt='fancy_grid', floatfmt=".3f"))

            # Summary metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            summary_metrics = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Score": [accuracy, precision, recall, f1]
            })

            self.logger.info("📌 Summary Metrics:")
            print(tabulate(summary_metrics, headers='keys', tablefmt='fancy_grid', floatfmt=".3f"))

        else:
            self.logger.warning("⚠️ 'Y' column not found. Skipping classification report.")

        return results
    
