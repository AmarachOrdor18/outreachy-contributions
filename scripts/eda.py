import os
import pandas as pd
import seaborn as sns
import logging
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

class ExploratoryDataAnalysis:
    def __init__(self, data_path="data/bbb.csv"):
        self.data_path = data_path
        self.figure_dir = os.path.join("data", "figures")  # Ensures all visuals are saved inside data/figures/
        os.makedirs(self.figure_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def generate_visuals(self):
        """
        Generates and saves three key EDA visualizations:
        1. Label Distribution (Does the drug cross the BBB?)
        2. SMILES String Length Distribution (Molecular Complexity)
        3. Molecular Structure Visualization (First 6 Drugs)
        """
        try:
            if not os.path.exists(self.data_path):
                self.logger.error(f"Dataset not found: {self.data_path}")
                return None

            df = pd.read_csv(self.data_path)

            # ✅ Label Distribution (0 = Non-permeable, 1 = Permeable)
            plt.figure(figsize=(6, 4))
            ax = sns.countplot(x=df["Y"], palette="coolwarm")

            # Add numbers on top of bars
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha='center', va='bottom', fontsize=12, fontweight='bold')

            plt.xlabel("BBB Permeability (0 = No, 1 = Yes)")
            plt.ylabel("Count")
            plt.title("Label Distribution - Blood Brain Barrier Permeability")

            # Save figure
            plt.savefig(os.path.join(self.figure_dir, "label_distribution.png"))
            plt.close()
            self.logger.info("✅ Label distribution saved.")


            # ✅ SMILES Length Distribution (Molecular Complexity)
            df["SMILES_Length"] = df["Drug"].apply(len)
            plt.figure(figsize=(8, 5))
            sns.histplot(df["SMILES_Length"], bins=30, kde=True, color="purple")
            plt.xlabel("SMILES String Length")
            plt.ylabel("Frequency")
            plt.title("Distribution of Molecular Complexity (SMILES Length)")
            plt.savefig(os.path.join(self.figure_dir, "smiles_length_distribution.png"))
            plt.close()
            self.logger.info("✅ SMILES length distribution saved.")

            # ✅ Visualizing Molecular Structures (First 6)
            smiles_list = df["Drug"].iloc[:6].tolist()
            drug_ids = df["Drug_ID"].iloc[:6].astype(str).tolist()  # Convert IDs to strings
            drug_names = df["Drug"].iloc[:6].tolist()  # Get drug names

            # Convert SMILES to molecular structures
            mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

            # Create labels using Drug ID and Name
            labels = [f"{drug_id}: {drug_name}" for drug_id, drug_name in zip(drug_ids, drug_names)]

            # Generate grid image with labels
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), legends=labels)

            # Save image
            img_path = os.path.join(self.figure_dir, "molecule_visualization.png")
            img.save(img_path)
            self.logger.info("✅ Molecular visualization saved.")

            self.logger.info("EDA visualizations completed and saved in 'data/figures' folder.")

        except Exception as e:
            self.logger.error(f"Error during EDA: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    eda = ExploratoryDataAnalysis()
    eda.generate_visuals()
