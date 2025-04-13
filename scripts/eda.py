import os
import pandas as pd
import seaborn as sns
import logging
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from IPython.display import display
from io import BytesIO

class ExploratoryDataAnalysis:
    def __init__(self, data_dir="data/"):
        notebook_dir = Path(__file__).resolve().parent
        base_path = notebook_dir.parent / data_dir

        self.data_path = base_path / "bbb.csv"
        self.figure_dir = base_path / "figures"
        os.makedirs(self.figure_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

        print(f"Data path is: {self.data_path}")

    def generate_visuals(self):
        try:
            if not os.path.exists(self.data_path):
                self.logger.error(f"Dataset not found: {self.data_path}")
                return None

            df = pd.read_csv(self.data_path)

            # Label Distribution
            plt.figure(figsize=(6, 4))
            ax = sns.countplot(x=df["Y"], palette="coolwarm")
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha='center', va='bottom', fontsize=12, fontweight='bold')
            plt.xlabel("BBB Permeability (0 = No, 1 = Yes)")
            plt.ylabel("Count")
            plt.title("Label Distribution - Blood Brain Barrier Permeability")
            plt.savefig(os.path.join(self.figure_dir, "label_distribution.png"))
            plt.show()
            plt.close()
            self.logger.info("✅ Label distribution saved.")

            # SMILES Length Distribution
            df["SMILES_Length"] = df["Drug"].apply(len)
            plt.figure(figsize=(8, 5))
            sns.histplot(df["SMILES_Length"], bins=30, kde=True, color="purple")
            plt.xlabel("SMILES String Length")
            plt.ylabel("Frequency")
            plt.title("Distribution of Molecular Complexity (SMILES Length)")
            plt.savefig(os.path.join(self.figure_dir, "smiles_length_distribution.png"))
            plt.show()
            plt.close()
            self.logger.info("✅ SMILES length distribution saved.")

            # Molecular Structures
            smiles_list = df["Drug"].iloc[:6].tolist()
            drug_ids = df["Drug_ID"].iloc[:6].astype(str).tolist()
            drug_names = df["Drug"].iloc[:6].tolist()

            mols = []
            valid_labels = []

            for smiles, drug_id, drug_name in zip(smiles_list, drug_ids, drug_names):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
                    valid_labels.append(f"{drug_id}: {drug_name}")
                else:
                    self.logger.warning(f"❗ Invalid SMILES string skipped: {smiles}")

            if mols:
                img = Draw.MolsToGridImage(
                    mols, molsPerRow=3, subImgSize=(200, 200), legends=valid_labels, useSVG=False
                )

                # Saving the image to a file using BytesIO
                bio = BytesIO()
                img.save(bio, format="PNG")  # Save to buffer as PNG
                bio.seek(0)  # Rewind the buffer to the beginning
                
                img_path = os.path.join(self.figure_dir, "molecule_visualization.png")
                with open(img_path, "wb") as f:
                    f.write(bio.getvalue())  # Write buffer to file

                display(img)  # Display the image in Jupyter notebook
                self.logger.info("✅ Molecular visualization saved.")
            else:
                self.logger.warning("❗ No valid molecules to visualize.")

            self.logger.info("EDA visualizations completed and saved in 'data/figures' folder.")

        except Exception as e:
            self.logger.error(f"Error during EDA: {e}", exc_info=True)
            return None
        
