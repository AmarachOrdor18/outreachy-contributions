import os
import logging
from eda import ExploratoryDataAnalysis
from featurise import FeatureExtractor
from data_loader import Dataloader
from train_model import ModelTraining
from evaluate_model import ModelEvaluation  # Ensure this is correctly imported

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# File paths
DATA_PATH = "data/bbb.csv"
TRAIN_FEATURES_PATH = "data/bbb_train_features.csv"
VALID_FEATURES_PATH = "data/bbb_valid_features.csv"
TEST_FEATURES_PATH = "data/bbb_test_features.csv"
MODEL_PATH = "models/best_xgboost_model.pkl"
RESULTS_PATH = "results/evaluation_metrics.json"


def main():
    """Automates the entire BBB dataset processing pipeline."""

    # Step 1: Download dataset if not present
    if not os.path.exists(DATA_PATH):
        logger.info("\U0001F4E5 Dataset not found. Downloading now...")
        downloader = Dataloader()
        downloader.download_data()
    else:
        logger.info("✅ Dataset found. Skipping download.")

    # Step 2: Perform Exploratory Data Analysis
    logger.info("📊 Running Exploratory Data Analysis...")
    eda = ExploratoryDataAnalysis()
    eda.generate_visuals()

    # Step 3: Extract Features (train, validation, and test features) if not already done
    if not (os.path.exists(TRAIN_FEATURES_PATH) and os.path.exists(VALID_FEATURES_PATH) and os.path.exists(TEST_FEATURES_PATH)):
        logger.info("🔬 Extracting molecular descriptors...")
        extractor = FeatureExtractor()
        extractor.generate_features()  # Ensure this handles creating the three separate files
    else:
        logger.info("✅ Feature extraction already done. Skipping.")

    # Step 4: Train Model
    if not os.path.exists(MODEL_PATH):
        logger.info("🤖 Training the XGBoost model...")
        model_trainer = ModelTraining(
            train_file=TRAIN_FEATURES_PATH,
            test_file=VALID_FEATURES_PATH,
            val_file=VALID_FEATURES_PATH
        )
        model_trainer.run()
    else:
        logger.info("✅ Model already trained. Skipping training.")

    # Step 5: Evaluate Model
    if os.path.exists(MODEL_PATH) and os.path.exists(VALID_FEATURES_PATH):
        logger.info("📈 Evaluating the model...")
        model_evaluator = ModelEvaluation(MODEL_PATH, VALID_FEATURES_PATH)
        model_evaluator.run_evaluation()
    else:
        logger.warning("⚠️ Model or validation dataset not found. Skipping evaluation.")
    
    logger.info("🚀 Full pipeline completed successfully!")


if __name__ == "__main__":
    main()
