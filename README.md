# Predicting Heart Disease Risk from Health Survey Indicators

**Break Through Tech AI Studio | Fall 2026**  
**Industry Partner:** Accenture

---

### 👥 **Team Members**

| Name             | GitHub Handle   | Contribution                                                             |
|------------------|-----------------|--------------------------------------------------------------------------|
| Grace Williams   | [@GraceCWilliams](https://github.com/GraceCWilliams) | Data exploration, preprocessing, and model development |
| Rachel Saini     | [@rachelsaini28](https://github.com/rachelsaini28) | TBD |
| Eunjae Lee       | [@Eunjae-hash](https://github.com/Eunjae-hash) | TBD |
| Jasraj Baweja    | [@jasrajbaweja](https://github.com/jasrajbaweja) | TBD |
| Dora Liao        | [@dora-liao](https://github.com/dora-liao) | TBD |
| Chu Wen Mo       | [@jazmine366](https://github.com/jazmine366) | TBD |

---

## 🎯 **Project Highlights**

- Developed a supervised machine learning project using approximately 445,000 records from the CDC's 2022 Behavioral Risk Factor Surveillance System (BRFSS).
- Investigated demographic, lifestyle, and self-reported health indicators to predict whether an individual reports a history of coronary heart disease or heart attack.
- Explored classification approaches, including random forest and gradient boosting (XGBoost), to evaluate predictive performance.
- Addressed substantial class imbalance through data preprocessing and classification evaluation strategies focused on identifying the minority positive class.
- Evaluated model performance using metrics including precision, recall, F1 score, ROC-AUC, and PR-AUC.
- Considered model interpretability and fairness to assess the potential applications and limitations of machine learning in healthcare.

---

## 👩🏽‍💻 **Setup and Installation**

Follow these instructions to clone the repository, install dependencies, access the dataset, and run the project notebooks.

**1. Clone the repository**

```bash
git clone https://github.com/Break-Through-Tech/Healthcare-2A-predicting-heart-disease-risk-from-health-survey-indicators.git

cd Healthcare-2A-predicting-heart-disease-risk-from-health-survey-indicators
```

**2. Set up a Python environment**

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Mac/Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

**3. Install dependencies**

Install the Python libraries used for data analysis, visualization, and machine learning:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

Additional libraries may be required depending on the notebook being executed.

**4. Access the dataset**

The original dataset is available through Kaggle:

[Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

Download the CSV dataset and place it in the repository's `Data/` directory if it is not already included.

**5. Run the notebooks**

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Alternatively, open the repository in Visual Studio Code and navigate to the `Notebooks/` directory.

Open and execute the notebooks in their intended order, beginning with exploratory data analysis and preprocessing, followed by model development and evaluation.

Ensure that the dataset file paths within the notebooks match the location of the downloaded data.

---

## 🏗️ **Project Overview**

Through the Break Through Tech AI Program, we partnered with Accenture to develop a supervised machine learning solution that predicts whether an individual reports a history of coronary heart disease or heart attack using lifestyle, demographic, and self-reported health indicators.

Using data from the Centers for Disease Control and Prevention's (CDC) 2022 Behavioral Risk Factor Surveillance System (BRFSS), our project investigates how machine learning can identify patterns associated with historical heart disease diagnoses.

We explore classification approaches, including random forest and gradient boosting (XGBoost), and evaluate model performance using metrics such as PR-AUC, F1 score, and ROC-AUC. Because the dataset contains substantial class imbalance, we also investigate strategies such as SMOTE to improve identification of the minority positive class.

Our objective is to develop a reproducible proof of concept that demonstrates how machine learning can be applied to healthcare risk analysis while considering predictive performance, interpretability, and responsible AI.

This project is intended for educational and research purposes. It predicts associations with self-reported historical diagnoses and is not a clinical diagnostic tool or a predictor of future cardiovascular events.

---

## 📊 **Data Exploration**

Our project uses the **Personal Key Indicators of Heart Disease** dataset, derived from the CDC's 2022 Behavioral Risk Factor Surveillance System (BRFSS).

- **Dataset:** Personal Key Indicators of Heart Disease
- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS), 2022
- **File type:** CSV
- **File size:** Approximately 25.19 MB
- **Observations:** Approximately 445,000
- **Features:** 98 columns
- **Data types:** Boolean, categorical, string, and numerical variables
- **Target variable:** Self-reported history of coronary heart disease or heart attack

**EDA and preprocessing approaches:**

- Characterize and document feature data types and encoding decisions.
- Identify and handle missing or inconsistent data.
- Examine correlations and relationships between health indicators.
- Analyze univariate and bivariate distributions.
- Investigate relationships between demographic characteristics, lifestyle factors, and the target variable.
- Examine the distribution of the target variable to understand class imbalance.
- Prepare stratified training, validation, and test datasets using a fixed random seed.

**Insights from our Exploratory Data Analysis (EDA):**

- The dataset exhibits substantial class imbalance, with individuals reporting a history of heart disease representing the minority class.
- Demographic, lifestyle, and self-reported health indicators provide different perspectives on factors associated with historical heart disease diagnoses.
- The imbalance between classes makes accuracy alone insufficient for evaluating classification performance.
- Data quality and the limitations of self-reported survey responses must be considered when interpreting relationships between features and the target variable.

**Challenges and assumptions when working with the dataset:**

- **Class imbalance:** The minority positive class presents challenges for model training and evaluation.
- **Self-reported data:** Responses may contain recall bias, reporting inconsistencies, or measurement limitations.
- **Feature representation:** Categorical and numerical variables require appropriate preprocessing before model development.
- **Generalizability:** Findings from the 2022 survey may not necessarily generalize to other populations or survey years.

**Potential visualizations included in the project:**

- Target variable distribution.
- Univariate distributions of demographic and health indicators.
- Bivariate relationships between selected predictors and the target variable.
- Correlation heatmaps.
- Comparisons of health indicators across demographic groups.

---

## 🧠 **Model Development**

Our project approaches heart disease history prediction as a supervised binary classification problem.

**Models explored:**

- **Logistic Regression:** Establishes an interpretable baseline for comparison.
- **Random Forest:** Captures nonlinear relationships and interactions between health indicators.
- **Gradient Boosting (XGBoost):** Explores boosted decision trees to improve predictive performance.
- **K-Nearest Neighbors:** A distance-based comparison model; class imbalance handled with resampling inside the training pipeline, since KNN has no class weights.

**Feature selection and preprocessing strategies:**

- Encoding categorical and binary variables.
- Preparing numerical and categorical predictors.
- Exploring feature relationships and potential predictors.
- Identifying relevant features for model development.
- Maintaining stratified data splits to preserve class distributions.

**Addressing class imbalance:**

We investigate approaches to improve the identification of the minority positive class, including:

- Class weighting.
- Random oversampling.
- Synthetic Minority Over-sampling Technique (SMOTE).

Resampling techniques are applied exclusively to training data to avoid data leakage and preserve the validity of model evaluation.

**Training and evaluation setup:**

Models are trained and compared using stratified training, validation, and test datasets. Model selection and hyperparameter tuning are guided by classification performance, with particular emphasis on F1 score, PR-AUC, and ROC-AUC.

---

## 📈 **Results & Key Findings**

Model performance is evaluated using multiple classification metrics to understand how effectively each approach identifies individuals reporting a history of heart disease.

**Performance metrics:**

- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive observations correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes across classification thresholds.
- **PR-AUC:** Evaluates the tradeoff between precision and recall, particularly useful for imbalanced datasets.
- **Confusion Matrix:** Summarizes true positives, true negatives, false positives, and false negatives.

**Model comparison:**

All results are on the same held-out test set (88,414 rows, 5.68% positive; 60/20/20 stratified split, seed 42). Precision, recall and F1 are for the positive (heart attack) class at each model's decision threshold.

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Threshold | Imbalance handling | Notebook |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.4462 | 0.5313 | **0.4850** | **0.8904** | **0.4318** | 0.236 (max F1 on validation) | `scale_pos_weight` | `xgboost_model.ipynb` |
| Logistic Regression | 0.2228 | **0.7672** | 0.3453 | 0.8885 | 0.4147 | 0.5 (default) | balanced class weights | `milestone1_eda_starter.ipynb` |
| Random Forest | **0.5974** | 0.1008 | 0.1724 | 0.8727 | 0.3933 | 0.5 (default) | balanced class weights | `milestone1_eda_starter.ipynb` |
| K-Nearest Neighbors | 0.4052 | 0.3939 | 0.3994 | 0.8570 | 0.3432 | 0.360 (max F1 on validation) | undersampling to 1:5 | `knn_model.ipynb` |

Notes for comparing:
- ROC-AUC and PR-AUC don't depend on the threshold, so they are the fairest comparison. Precision / recall / F1 shift with the threshold: Logistic Regression and Random Forest still use the default 0.5, so their F1 would likely improve with a tuned threshold.
- PR-AUC is `average_precision_score` for XGBoost and KNN, and `auc(recall, precision)` (trapezoidal) for Logistic Regression and Random Forest; the two can differ slightly.

**Key findings:**

Our evaluation focuses on:

- Identifying the best-performing classification approach.
- Comparing model performance against the baseline.
- Examining the effect of class-balancing strategies on minority-class prediction.
- Identifying health and demographic indicators that contribute to model predictions.
- Evaluating tradeoffs between precision and recall.
- Investigating potential differences in predictive performance across demographic subgroups.

**Model interpretability and fairness:**

Model interpretability techniques, including SHAP (SHapley Additive exPlanations), can help examine the contribution of individual features to model predictions.

We also consider fairness by comparing performance across demographic subgroups, including age, sex, race/ethnicity, and income, to identify potential disparities.

**Potential visualizations included in the project:**

- Confusion matrices.
- Precision-recall curves.
- ROC curves.
- Feature importance plots.
- SHAP summary plots.
- Model performance comparisons.
- Demographic subgroup performance comparisons.

---

## 🚀 **Next Steps**

Potential areas for further development include:

- Further hyperparameter tuning and feature engineering to improve classification performance.
- Comparing additional machine learning algorithms and class-balancing strategies.
- Expanding model interpretability using SHAP.
- Conducting additional fairness evaluations across demographic subgroups.
- Investigating model generalizability using additional BRFSS survey years.
- Improving documentation and reproducibility to make the modeling workflow easier to replicate.

**Limitations:**

- The dataset relies on self-reported survey responses, which may introduce recall and reporting bias.
- Class imbalance makes identifying the minority positive class more challenging.
- The dataset represents a specific survey year and may not generalize to other populations or time periods.
- Associations identified by the model do not establish causation or predict future cardiovascular events.
- The model is not intended for clinical diagnosis or individual medical decision-making.

---

## 📝 **License**

This project is licensed under the **MIT License**.

The MIT License permits others to use, reproduce, modify, and distribute the project's code, provided that the original copyright and license notices are retained.

See the [LICENSE](LICENSE) file for additional information.

---

## 📄 **References**

1. Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System (BRFSS), 2022.*
2. Kaggle. *Personal Key Indicators of Heart Disease.* https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
3. Break Through Tech. *AI Studio Program, Fall 2026.*
4. Accenture. Healthcare AI Studio Challenge Project.

---

## 🙏 **Acknowledgements**

We would like to thank **Break Through Tech** and **Accenture** for providing the opportunity to work on this healthcare-focused machine learning challenge through the AI Studio program.

We are especially grateful to our Challenge Advisor, **Joseph Chiasson**, for his guidance, feedback, and support throughout the project. We also thank our AI Studio Coach and the broader Break Through Tech team for their continued mentorship and support.

Finally, we acknowledge the **Centers for Disease Control and Prevention (CDC)** for making the Behavioral Risk Factor Surveillance System (BRFSS) data publicly available, which made this project possible, and thank all team members for their collaboration and contributions.
