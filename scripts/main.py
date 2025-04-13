import os
import logging
from pathlib import Path
from eda import ExploratoryDataAnalysis
from featurise import FeatureExtractor
from data_loader import Dataloader
from train_model import ModelTraining
from evaluate_model import ModelEvaluation
from apply_model import ApplyTrainedModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths (using formatted strings for dynamic IDs)
DATA_PATH = "data/bbb.csv"

def main():
    """Automates the entire BBB dataset processing pipeline."""
    
    # Set identifiers (these could be passed as arguments or read from a config)
    featurizer_id = "eos8a4x"  # Example featurizer ID
    model_type = "random_forest"  # Example model type
    cv_strategy = "stratified_kfold"  # Example CV strategy

    # Feature paths dynamically using featurizer_id
    train_features_path = f"data/bbb_train_{featurizer_id}_features.csv"
    valid_features_path = f"data/bbb_valid_{featurizer_id}_features.csv"
    test_features_path = f"data/bbb_test_{featurizer_id}_features.csv"

    # Step 1: Download dataset if not present
    if not os.path.exists(DATA_PATH):
        logger.info("\U0001F4E5 Dataset not found. Downloading now...")
        downloader = Dataloader()
        downloader.fetch_bbb_dataset()  # Download and split dataset
    else:
        logger.info("✅ Dataset found. Skipping download.")

    # Step 2: Perform Exploratory Data Analysis (EDA)
    logger.info("📊 Running Exploratory Data Analysis...")
    eda = ExploratoryDataAnalysis(data_dir="data/")
    eda.generate_visuals()

    # Step 3: Extract Features
    if not (os.path.exists(train_features_path) and os.path.exists(valid_features_path) and os.path.exists(test_features_path)):
        logger.info("🔬 Extracting molecular descriptors...")
        feature_extractor = FeatureExtractor(featurizer_id=featurizer_id)
        feature_extractor.generate_features()
    else:
        logger.info("✅ Feature extraction already done. Skipping.")

    # Step 4: Train Model
    model_filename = f"best_{featurizer_id}_{model_type}_{cv_strategy}_model.pkl"
    model_path = os.path.join("models", model_filename)

    if not os.path.exists(model_path):
        logger.info("🤖 Training the model...")
        trainer = ModelTraining(
            featurizer_id=featurizer_id,
            model_type=model_type,
            cv_strategy=cv_strategy,
            train_file=train_features_path,
            val_file=valid_features_path,
            test_file=test_features_path
        )
        trainer.run()  # This trains and saves the model
    else:
        logger.info(f"✅ Model found at {model_path}. Skipping training.")

    # Step 5: Apply Model to Unseen Data Before Evaluation
    unseen_data_path = "data/unseen_data.csv"  # Change this path as needed
    if os.path.exists(model_path) and os.path.exists(unseen_data_path):
        logger.info("🔎 Applying trained model to unseen data...")
        predictor = ApplyTrainedModel(
            data_path=unseen_data_path,
            featurizer_id=featurizer_id,
            model_type=model_type,
            cv_strategy=cv_strategy
        )
        output = predictor.run()
    else:
        logger.warning("⚠️ Model or unseen data not found. Skipping prediction step.")

    # Step 6: Evaluate Model
    if os.path.exists(model_path) and os.path.exists(valid_features_path):
        logger.info("📈 Evaluating the model...")
        evaluator = ModelEvaluation(
            featurizer_id=featurizer_id,
            model_type=model_type,
            cv_strategy=cv_strategy
        )
        evaluator.run_evaluation()
    else:
        logger.warning("⚠️ Model or validation dataset not found. Skipping evaluation.")

    logger.info("🚀 Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
