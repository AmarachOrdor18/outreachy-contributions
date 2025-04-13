# **Blood Brain Barrier (BBB) Permeability Prediction**

This project aims to build a machine learning pipeline that predicts whether a compound can cross the blood-brain barrier (BBB). BBB permeability is a critical consideration in drug development, particularly for medications targeting the central nervous system. The pipeline includes dataset collection, exploratory data analysis (EDA), molecular featurization, model training, and performance evaluation. The goal is to automate dataset loading, feature extraction, model training, application of model and evaluation to accelerate drug discovery for **neurological diseases** such as **stroke, epilepsy, and meningitis**.
 
## Tech Stack

- **Language**: Python 3.11.11
- **Machine Learning**: Scikit-Learn, XGBoost
- **Molecular Featurization**: RDKit, Morgan counts fingerprints
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Automation**: Shell scripting, Python scripts

## Project Structure

```
/project-root
├── data/                    # Raw & processed datasets
│   ├── figures/             # Visualization images 
├── models/                  # Trained models
├── notebooks/               # Jupyter Notebooks for testing
├── scripts/                 # Automation scripts
│   ├── data_loader.py       # Data loading script
│   ├── eda.py               # Exploratory data analysis
│   ├── featurise.py         # Molecular featurization
│   ├── train_model.py       # Model training
│   ├── apply_model.py       # Applying Trained model on Unseen Data
│   ├── evaluate_model.py    # Model evaluation
│   └── main.py              # Full automation pipeline
├── README.md                # This file
├── requirements.txt         # Dependencies
```

### File Naming Convention
To ensure consistency and clarity, I use a naming convention for files based on the featurizer, model type, and cross-validation strategy. For example, when generating filenames, the base name is created by combining these components. The files include the model, scaler, feature columns, and zero-variance columns, all named with the base name followed by their respective type. This ensures uniformity and easy identification of each file across different splits and processes.
`{featurizer_id}_{model_type}_{cv_strategy}`

---

## Important: Dependencies

**Ensure that you have Conda and Ersilia installed before running the project, also make sure docker is running as the ersilia commands won't work without it**

- Use **Conda** to manage dependencies
- **Ersilia CLI** is required for molecular descriptor extraction

### Installation

#### Setting up the environment

1. Clone the repository into the project folder:
   ```bash
   git clone https://github.com/AmarachOrdor18/outreachy-contributions.git
   cd outreachy-contribution
   ```

2. Install required dependencies:
   ```bash
   python3.11 -m venv myenv
   source myenv/bin/activate
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

##  Task 1: Download the Dataset

To begin, I selected a relevant classification dataset from the Therapeutics Data Commons (TDC)https://tdcommons.ai/, specifically from the ADME (Absorption, Distribution, Metabolism, and Excretion) category. The dataset used is called `BBB_Martins`, which labels compounds based on their ability to penetrate the BBB.

The dataset was downloaded and processed using Python. It was programmatically split using a scaffold splitting method to generate training, validation, and test sets. These were saved locally in the `/data` directory for organized access. Preliminary documentation was initiated to describe installation steps, dataset format, and usage instructions.

| Field         | Description                          |
|--------------|----------------------------------|
| **Drug_ID**   | Compound name/Identifier        |
| **Drug**      | SMILES Notation of compound as a string|
| **Y**         | Binary classification, 0: Not Permeable, 1: Permeable       | 



### Exploratory Data Analysis (EDA)

To better understand the dataset, I performed exploratory analysis using visualizations. First, I examined the distribution of the target label, which showed the imbalance balance between BBB-permeable (1551) and non-permeable compounds (479). Then, I explored the complexity of molecules using SMILES string length as a proxy.

Additionally, a sample of molecules was visualized to provide a sense of chemical diversity in the dataset. These images, alongside other plots like label distribution and SMILES length histograms, were saved to the `/data/figures` directory.

| Split      |   Class 0 |   Class 1 |   Total | Class 0 (%)   | Class 1 (%)   |
|:-----------|----------:|----------:|--------:|:--------------|:--------------|
| Train      |       365 |      1056 |    1421 | 25.69%        | 74.31%        |
| Validation |        36 |       167 |     203 | 17.73%        | 82.27%        |
| Test       |        78 |       328 |     406 | 19.21%        | 80.79%        |
| Total      |       479 |      1551 |    2030 | 23.60%        | 76.40%        |

### Label Distribution
I looked at the distribution of labels and found that the dataset contains a highly imbalanced distribution of labels, with more **BBB permeable** compounds (1551) than **non-permeable** compounds (479). We need to address this imbalance because it is important before training and evaluating our model.
The label distribution can be found [here](https://github.com/ersilia-os/outreachy-contributions/blob/d8e721e67aad1959523654eaa99a5858a9595bc4/data/figures/label_distribution.png)

### SMILES Length Distribution
Also, i took a look at the length of these SMILES strings, and observed that it varies significantly. This can be found [here](https://github.com/ersilia-os/outreachy-contributions/blob/d8e721e67aad1959523654eaa99a5858a9595bc4/data/figures/smiles_length_distribution.png), where most molecules have SMILES lengths around 30 to 60 characters.


### Sample Drug Structures
To better understand the dataset, I visualized some molecular structures of drugs in the dataset. This visualization can help in analyzing the chemical properties of **BBB permeable** and **non-permeable**
The sample drug structures can be found [here](https://github.com/ersilia-os/outreachy-contributions/blob/d8e721e67aad1959523654eaa99a5858a9595bc4/data/figures/molecule_visualization.png)


---

## Task 2: Featurisation

The choice of Featuriser for this project are based on two criterions
1. The features should focus on the basic properties of molecules, like size and solubility, that help them pass through barriers like the blood-brain barrier
2. We need to look at how drug-like a compound is, meaning whether it has the right qualities to be developed into a medicine.
3. The way a compound works in the body is important because it affects how well it can cross the blood-brain barrier.

I checked the **[eos8a4x: RDKit 200 Physicochemical Descriptors](https://github.com/ersilia-os/eos8a4x)**, which provides key **physicochemical descriptors** like **molecular weight**, **solubility**, and **druggability**, essential for predicting BBB permeability. This featurizer directly relates to the factors influencing BBB crossing, making it highly relevant for our drug dataset.

I also explored **[eos5axz: Morgan counts fingerprints](https://github.com/ersilia-os/eos5axz)**, which generates **circular molecular fingerprints** capturing local chemical environments. These are effective for predicting molecular interactions with biological membranes, making it a strong choice for BBB permeability prediction.

Together, **eos8a4x** and **eos5axz** offer complementary features focusing on both **physicochemical properties** and **molecular structure**, ideal for BBB permeability prediction.

The featurisation process was automated using the Ersilia CLI. Features were generated for training, validation, and test datasets and saved with appropriate naming conventions that included the featurizer ID. This step ensured that feature generation was reproducible and easy to update if needed. All descriptors were saved in separate files in the `/data` directory.

---

## Task 3: Model Building

#### Imbalance in BBB Permeability
The dataset is **highly imbalanced**, with **1551 BBB permeable compounds** (label 1) and only **479 non-permeable compounds** (label 0). This imbalance can lead the model to be biased toward predicting **permeable compounds**, neglecting **non-permeable** ones, which is crucial in drug development.

#### Importance of Addressing the Imbalance
A biased model would struggle to detect **non-permeable compounds**, affecting its real-world applicability. To avoid this, we need to balance the dataset before training.

#### Techniques to Address Imbalance
1. **Oversampling** the minority class (**non-permeable compounds**) by duplicating instances.
2. **SMOTE (Synthetic Minority Over-sampling Technique)** to create synthetic data points for the minority class.
3. **Hybrid Approach** combining **SMOTE** and **undersampling** the majority class to balance the dataset.

#### Data Preprocessing
- **Scaling**: Used `StandardScaler` to standardize the features.
- **Data Cleaning**: Removed missing/infinite values and outliers.
- **Zero Variance Features**: Removed features with no variability.

#### Model Selection and Tuning

I tested different combinations of **model types** and **cross-validation strategies** to see which would perform best for predicting Blood Brain Barrier permeability. I compared **XGBoost** and **Random Forest**, both good at handling imbalanced data. I used **GridSearchCV** for tuning and tried **Stratified K-Fold Cross-Validation** to keep class distribution balanced during training.

The results below show how the choice of model and validation method affected performance.


#### Random Forest with Stratified K-Fold Cross-Validation

| Metric               | Train Score | Validation Score | Test Score |
|----------------------|-------------|------------------|------------|
| Accuracy             | 0.9831      | 0.8719           | 0.8793     |
| Precision (Weighted) | 0.9832      | 0.8631           | 0.8812     |
| Recall (Weighted)    | 0.9831      | 0.8719           | 0.8793     |
| F1 Score (Weighted)  | 0.9831      | 0.8654           | 0.8802     |
| ROC AUC              | 0.9991      | 0.8504           | 0.9117     |

#### XGBoost with Grid Search Cross-Validation

| Metric               | Train Score | Validation Score | Test Score |
|----------------------|-------------|------------------|------------|
| Accuracy             | 0.9930      | 0.8522           | 0.8645     |
| Precision (Weighted) | 0.9930      | 0.8388           | 0.8652     |
| Recall (Weighted)    | 0.9930      | 0.8522           | 0.8645     |
| F1 Score (Weighted)  | 0.9930      | 0.8424           | 0.8649     | 
| ROC AUC              | 0.9999      | 0.8437           | 0.9019     | 

The **Random Forest** model with **Stratified K-Fold Cross-Validation** performs better than **XGBoost** on the test set. It has higher accuracy (0.8793 vs. 0.8645), precision (0.8812 vs. 0.8652), recall (0.8793 vs. 0.8645), and F1 score (0.8802 vs. 0.8649). It also has a higher ROC AUC (0.9117 vs. 0.9019), showing it’s more reliable for classifying the test data. In order to finalize on what model is best the classification report was also used, which can be found [here](https://github.com/AmarachOrdor18/outreachy-contributions/blob/eos8a4x/notebooks/Featurisation%20and%20Model%20training%20.ipynb)

#### Prediction
For predictions, I preprocessed new data in the same way as the training data and used the trained models to ensure consistent results. The results below show how the choice of model and validation method affected performance. You can find sample predictions [here](https://github.com/AmarachOrdor18/outreachy-contributions/blob/eos8a4x/notebooks/Featurisation%20and%20Model%20training%20.ipynb)

By addressing the imbalance, I ensured that the model predicts both **BBB permeable** and **non-permeable** compounds effectively.

#### Using an Alternative Featuriser

Previously the featuriser we have been using was the **eos8a4x**, but now we will be using the **Morgan counts fingerprints (eos5axz)**

The results below show how the choice of model and validation method affected performance.

**XGBoost with Grid Search Cross-Validation**

| Metric               | Train Score | Validation Score | Test Score |
|----------------------|-------------|------------------|------------|
| Accuracy             | 0.969       | 0.8867           | 0.867      |
| Precision (Weighted) | 0.9689      | 0.8803           | 0.8623     |
| Recall (Weighted)    | 0.969       | 0.8867           | 0.867      |
| F1 Score (Weighted)  | 0.9688      | 0.8742           | 0.8642     |
| ROC AUC              | 0.997       | 0.8641           | 0.8709     |

**Random Forest with Stratified K-Fold Cross-Validation**

| Metric               | Train Score | Validation Score | Test Score |
|----------------------|-------------|------------------|------------|
| Accuracy             | 0.931       | 0.8916           | 0.8867     |
| Precision (Weighted) | 0.9334      | 0.8859           | 0.8809     |
| Recall (Weighted)    | 0.931       | 0.8916           | 0.8867     |
| F1 Score (Weighted)  | 0.9281      | 0.8807           | 0.8775     |
| ROC AUC              | 0.9829      | 0.8317           | 0.86       |

Based on the test performance, **Random Forest with Stratified K-Fold Cross-Validation using Featurizer 1(eos8a4x)** is the best model for predicting drug permeability for the Blood-Brain Barrier (BBB). This model achieved the highest test accuracy (0.8793) and performed well across all metrics, including precision (0.8812), recall (0.8793), F1 score (0.8802), and ROC AUC (0.9117). In comparison, although XGBoost with Grid Search Cross-Validation showed strong performance, particularly in train and validation scores, it had a lower test accuracy (0.8645) and slightly lower performance across other metrics. Therefore, the Random Forest model with Featurizer 1 is the most robust for this task.

### Model Evaluation Metrics

featurizer_id = "eos8a4x" 
model_type = "random_forest"
cv_strategy = "stratified_kfold"

| Metric               | Train   | Test    | Validation |
|----------------------|---------|---------|------------|
| Accuracy             | 0.9831  | 0.9831  | 0.9815     |
| Precision (Weighted) | 0.9832  | 0.9832  | 0.9820     |
| Recall (Weighted)    | 0.9831  | 0.9831  | 0.9810     |
| F1 Score (Weighted)  | 0.9831  | 0.9831  | 0.9813     |
| ROC AUC              | 0.9991  | 0.9991  | 0.9987     |

---
## Model Evaluation

To check how well the model is doing, I used a validation dataset that I had kept aside earlier. I made sure to apply the same steps for preparing the data (like scaling) before making predictions. Then, I measured the model's performance using several metrics: accuracy, precision, recall, F1 score, and ROC AUC.
The results were presented as classification reports and visualized through:

- A confusion matrix
- ROC curve
- Precision-Recall curve
- Feature importance bar chart

All plots were saved in the `/data/figures` directory, To see a proper evaluation of the model you can look into the follwing [Model Evaluation notebook](https://github.com/ersilia-os/outreachy-contributions/blob/d8e721e67aad1959523654eaa99a5858a9595bc4/notebooks/Model%20Evaluation.ipynb)

Here is a brief explanation of what each metrics mean in the prediction of BBB permeability

| **Metric**                | **Description** |
|---------------------------|-----------------|
| **Accuracy**              | Indicates the overall proportion of correct predictions made by the model. |
| **Precision**             | Measures the proportion of predicted permeable drugs that are actually permeable, helping to reduce false positives. |
| **Recall (Sensitivity)**  | Assesses the model’s ability to identify all truly permeable drugs, focusing on minimizing false negatives. |
| **F1-Score**              | Represents the balance between precision and recall, particularly valuable for imbalanced datasets. |
| **AUROC (ROC AUC)**       | Evaluates how well the model distinguishes between permeable and non-permeable drugs across all thresholds. |
| **Confusion Matrix**      | Summarizes prediction outcomes, showing true/false positives and negatives for deeper error analysis. |
| **Precision-Recall Curve**| Visualizes the trade-off between precision and recall across thresholds — especially useful when classes are imbalanced. |
| **Feature Importance**    | Highlights which molecular features most strongly influence the model’s prediction of BBB permeability. |

---

## Applying Model on Unseen data

Previously it was established that using the eos8a4x featuriser, a random forest model with a stratified k fold was the best beforming, to validate its performance, i applied both variation of model architecture and featuriser type on the unseen data, after whuch i confirmed that the eos8a4x featuriser, a random forest model with a stratified k fold was the best beforming, best redicted the permeability and non-permeabily for a drug to pass through the brain barrier. I made this process reproducible by creating a python script called apply_model.py that allows you to import the class into a note book and run the models based on any other unseen data. Here is the classification report for the Unseen Data 

### Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.621     | 0.857  | 0.720    | 21      |
| 1             | 0.962     | 0.874  | 0.916    | 87      |
| **Accuracy**  | 0.870 | 0.870| 0.870| 108 |
| **Macro avg** | 0.791     | 0.865  | 0.818    | 108     |
| **Weighted avg** | 0.896  | 0.870  | 0.878    | 108     |


## Any Improvements 

To improve the Ersilia Model Hub, we can enhance data imbalance handling by fine-tuning oversampling methods like **SMOTE**, and exploring advanced techniques such as **ADASYN (Adaptive Synthetic Sampling)**. Using ensemble methods like **XGBoost**, **Random Forest**, or model stacking with **scikit-learn's StackingClassifier** can improve prediction accuracy. Robust cross-validation using **StratifiedKFold** or **RepeatedStratifiedKFold** ensures consistency in high-dimensional datasets. Additionally, incorporating interpretability tools like **SHAP** and **LIME** is essential, especially in drug development, where understanding model predictions is critical for real-world decisions.