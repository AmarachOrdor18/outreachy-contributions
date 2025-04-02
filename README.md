# **Blood Brain Barrier (BBB) Permeability Prediction**

## Getting Started

This project aims to predict whether molecules can cross the **Blood Brain Barrier (BBB)** using **machine learning** and **AI-based models**. The goal is to automate dataset loading, feature extraction, model training, and evaluation to accelerate drug discovery for **neurological diseases** such as **stroke, epilepsy, and meningitis**.

## Tech Stack

- **Language**: Python 3.12
- **Machine Learning**: Scikit-Learn, XGBoost
- **Molecular Featurization**: RDKit, Ersilia
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Automation**: Shell scripting, Python scripts

## Project Structure

```
/project-root
├── data/                    # Raw & processed datasets
│   ├── figures/             # Visualization images
│   ├── bbb_martins.tab      # Raw BBB dataset
│   ├── bbb_test_features.csv # Test features for model evaluation
│   ├── bbb_test.csv         # Test data for model evaluation
│   ├── bbb_train_features.csv # Training features for model
│   ├── bbb_train.csv        # Training data for model
│   ├── bbb_valid_features.csv # Validation features
│   ├── bbb_valid.csv        # Validation data
│   ├── test_descriptors     # Molecular descriptors for testing
│   ├── train_descriptors    # Molecular descriptors for training
│   ├── valid_descriptors    # Molecular descriptors for validation
│   ├── bbb.csv              # Full dataset
│
├── models/                  # Trained models
│   └── best_xgboost_model.pkl # Best XGBoost model
│
├── notebooks/               # Jupyter Notebooks for testing
│   ├── Data Exploration.ipynb # Exploratory data analysis notebook
│   ├── Featurisation and Model Training.ipynb # Feature extraction and model training notebook
│   ├── Model Evaluation.ipynb # Model evaluation notebook
│
├── scripts/                 # Automation scripts
│   ├── data_loader.py       # Data loading script
│   ├── eda.py               # Exploratory data analysis
│   ├── featurise.py         # Molecular featurization
│   ├── train_model.py       # Model training
│   ├── evaluate_model.py    # Model evaluation
│   └── main.py              # Full automation pipeline
│
├── README.md                # This file
├── requirements.txt         # Dependencies
```

---

## ⚠️ Important: Dependencies

**Ensure that you have Conda and Ersilia installed before running the project.**

- Use **Conda** to manage dependencies
- **Ersilia CLI** is required for molecular descriptor extraction

### Installation

#### Setting up the environment

1. Clone the repository into the project folder:
   ```bash
   conda create --name ersilia python=3.12 -y
   conda activate ersilia

   # Clone the repository and navigate into it
   git clone https://github.com/AmarachOrdor18/outreachy-contributions.git
   cd outreachy-contribution
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Running the Automation Script

3. Execute the full automation pipeline:
   ```bash
   python scripts/main.py
   ```

#### Running the Notebooks

1. Launch Jupyter:
   ```bash
   jupyter lab
   ```

2. Open and run the notebooks:
   - **Data Exploration**: `notebooks/Data Exploration.ipynb`
   - **Featurisation and Model Training**: `notebooks/Featurisation and Model Training.ipynb`
   - **Model Evaluation**: `notebooks/Model Evaluation.ipynb`

---

## **Why This Matters**

Predicting **Blood Brain Barrier permeability** is crucial for drug discovery. This project automates the process of identifying whether drug molecules can cross the BBB, which is a key factor in the development of treatments for neurological diseases such as **meningitis, epilepsy, and stroke**. Automating this process makes it **faster** and **cheaper** to identify promising drug candidates, benefiting research and accelerating drug discovery efforts.

---

## **Troubleshooting**

### **Common Issues**

1. **Dataset Not Found**
   - Ensure the dataset is correctly downloaded.
   - Verify file paths in the `data/` folder.

2. **Ersilia Model Fetching Fails**
   - Check your internet connection.
   - Ensure the **Ersilia CLI** is installed properly and configured.

3. **Low Model Accuracy**
   - Tune hyperparameters in the `scripts/train.py`.
   - Consider increasing the dataset size or using more descriptive features.