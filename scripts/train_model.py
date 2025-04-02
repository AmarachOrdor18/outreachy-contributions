import os
import pandas as pd
import warnings
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.utils.validation import check_X_y

# Suppress warnings (especially FutureWarnings)
warnings.simplefilter(action='ignore', category=FutureWarning)

class ModelTraining:
    def __init__(self, train_file, test_file, val_file, target_col="Y", non_feature_cols=None):
        if non_feature_cols is None:
            non_feature_cols = ["Drug_ID", "Drug", "key", "input", "Y"]  # Default non-feature columns
        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file
        self.target_col = target_col
        self.non_feature_cols = non_feature_cols
        self.X_train, self.y_train = self.load_data(train_file)
        self.X_test, self.y_test = self.load_data(test_file)
        self.X_val, self.y_val = self.load_data(val_file)

    def load_data(self, file_path):
        """Load dataset and separate features and target variable."""
        df = pd.read_csv(file_path)
        X = df.drop(columns=self.non_feature_cols)
        y = df[self.target_col]
        return X, y

    def apply_smote(self):
        """Apply SMOTE to balance the training dataset."""
        X_train_valid, y_train_valid = check_X_y(self.X_train, self.y_train)
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_valid, y_train_valid)
        return X_train_balanced, y_train_balanced

    def tune_model(self, X_train_balanced, y_train_balanced):
        """Perform hyperparameter tuning using GridSearchCV."""
        xgb_model = xgb.XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 9],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                                   scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test, X_val, y_val):
        """Evaluate the model on test and validation datasets."""
        y_pred_test = model.predict(X_test)
        y_pred_val = model.predict(X_val)
        
        test_acc = accuracy_score(y_test, y_pred_test)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("Test Classification Report:\n", classification_report(y_test, y_pred_test))

    def save_model(self, model, model_filename="best_xgboost_model.pkl"):
        """Save the trained model to the 'models' directory."""
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(project_root, 'models')  

        # Ensure the models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Define full model path
        model_file_path = os.path.join(models_dir, model_filename)

        # Save the model
        joblib.dump(model, model_file_path)
        print(f"Trained model saved to {model_file_path}")

    def load_model(self, model_filename="best_xgboost_model.pkl"):
        """Load a saved model from the 'models' directory."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_file_path = os.path.join(project_root, 'models', model_filename)

        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        return joblib.load(model_file_path)

    def run(self):
        """Run the full training, tuning, and evaluation process."""
        X_train_balanced, y_train_balanced = self.apply_smote()
        best_model = self.tune_model(X_train_balanced, y_train_balanced)
        self.evaluate_model(best_model, self.X_test, self.y_test, self.X_val, self.y_val)
        self.save_model(best_model)

if __name__ == "__main__":
    # Set environment variable for SCIPY_ARRAY_API
    os.environ["SCIPY_ARRAY_API"] = "1"

    # Initialize and run the model training
    model_trainer = ModelTraining(
        train_file="data/bbb_train_features.csv", 
        test_file="data/bbb_valid_features.csv", 
        val_file="data/bbb_valid_features.csv"
    )
    model_trainer.run()
