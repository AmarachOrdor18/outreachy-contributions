import os
import time
import pandas as pd
import logging
import subprocess
from pathlib import Path
from rdkit import Chem
import numpy as np

# Patch deprecated attributes for compatibility with Mordred
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

class FeatureExtractor:
    def __init__(self, featurizer_id, log_file="ersilia_output.log"):
        """
        Initializes the FeatureExtractor with a specified featurizer model from Ersilia.
        Sets up paths for train, validation, and test datasets.
        """
        self.featurizer_id = featurizer_id
        notebook_dir = Path(__file__).resolve().parent
        base_path = notebook_dir.parent / "data"

        self.datasets = {
            "train": {
                "input": base_path / "bbb_train.csv",
                "output": base_path / f"bbb_train_{featurizer_id}_features.csv",
            },
            "valid": {
                "input": base_path / "bbb_valid.csv",
                "output": base_path / f"bbb_valid_{featurizer_id}_features.csv",
            },
            "test": {
                "input": base_path / "bbb_test.csv",
                "output": base_path / f"bbb_test_{featurizer_id}_features.csv",
            },
        }

        # Set up logger
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def run_command(self, command):
        """Run a shell command and return its output or None if it fails."""
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            self.logger.info(result.stdout)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {command}")
            self.logger.error(e.stderr)
            return None

    def generate_features(self):
        """
        Extracts features from SMILES in the train, validation, and test datasets
        using the specified Ersilia featurizer.
        """
        try:
            output_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(output_dir, exist_ok=True)

            self.logger.info(f"Fetching model '{self.featurizer_id}' from Ersilia...")
            if self.run_command(f"ersilia -v fetch {self.featurizer_id}") is None:
                return

            self.logger.info("Serving the model in the background...")
            if self.run_command(f"ersilia -v serve {self.featurizer_id}") is None:
                return

            self.logger.info("Waiting 20 seconds for the model to fully start...")
            time.sleep(20)

            for dataset, paths in self.datasets.items():
                input_file = paths["input"]
                output_file = paths["output"]
                # Customize the descriptor file name using featurizer_id
                output_file = Path(paths["output"])  # Ensure it's a Path object
                descriptors_file = output_file.parent / f"{output_file.stem}_descriptors.csv"

                if not os.path.exists(input_file):
                    self.logger.error(f"Dataset not found: {input_file}")
                    continue

                self.logger.info(f"Running feature extraction for {dataset}...")
                command = f"ersilia -v run -i {input_file} -o {descriptors_file}"
                if self.run_command(command) is None:
                    continue

                if not os.path.exists(descriptors_file):
                    self.logger.error(f"Feature extraction failed for {dataset}! Check Ersilia logs.")
                    continue

                self.logger.info(f"Merging extracted features with {dataset} dataset...")
                descriptors = pd.read_csv(descriptors_file)
                df = pd.read_csv(input_file)

                # Merge excluding duplicate SMILES if present
                df = pd.concat([df, descriptors.drop(columns=["SMILES"], errors='ignore')], axis=1)
                df.to_csv(output_file, index=False)

                self.logger.info(f"Feature extraction complete! Data saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error during feature extraction: {e}", exc_info=True)
            return None

    def featurize_smiles(self, smiles):
        """
        Featurizes a single SMILES string using the specified Ersilia model.

        Args:
            smiles: A SMILES string representing a molecule.

        Returns:
            pd.DataFrame or None: A dataframe with the features, or None if featurization failed.
        """
        try:
            t1 = time.time()

            self.logger.info(f"Featurizing SMILES string with model '{self.featurizer_id}'...")

            # Fetch and serve the model if it hasn't been done already
            if self.run_command(f"ersilia -v fetch {self.featurizer_id}") is None:
                return None

            self.logger.info("Serving the model in the background...")
            if self.run_command(f"ersilia -v serve {self.featurizer_id}") is None:
                return None

            self.logger.info("Waiting 20 seconds for the model to fully start...")
            time.sleep(20)

            # Create a customized temporary input file using the featurizer_id
            input_file = f'../data/{self.featurizer_id}_temp_smiles.csv'
            with open(input_file, 'w') as f:
                f.write('SMILES\n')
                f.write(f'{smiles}\n')

            # Customize the descriptors file name based on featurizer_id
            descriptors_file = f'../data/{self.featurizer_id}_placeholder.csv'
            command = f"ersilia -v run -i {input_file} -o {descriptors_file}"
            if self.run_command(command) is None:
                return None

            if not os.path.exists(descriptors_file):
                self.logger.error(f"Feature extraction failed for SMILES: {smiles}")
                return None

            # Read and return the features
            X = pd.read_csv(descriptors_file)
            X.drop(columns=['key', 'input'], inplace=True)

            time_taken = time.time() - t1
            self.logger.info(f"Featurization completed in {time_taken:.2f}s for {smiles}")

            return X

        except Exception as e:
            self.logger.error(f"Error during featurization for SMILES: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    # Replace with the actual featurizer ID you want to use
    featurizer_id = "eos8a4x"
    
    # Initialize the FeatureExtractor object
    extractor = FeatureExtractor(featurizer_id)

    # Example SMILES string (Ethanol)
    smiles_string = "CCO"

    # Call the featurize_smiles method to process the single SMILES string
    features_df = extractor.featurize_smiles(smiles_string)

    if features_df is not None:
        print("Featurization successful! Features extracted:")
        print(features_df)
    else:
        print("Featurization failed.")
