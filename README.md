# Machine Learning HW2: Binary Classification with Logistic Regression

A comprehensive binary classification project analyzing the SUPPORT2 dataset to predict in-hospital mortality using logistic regression with extensive data preprocessing and model evaluation.

## 📋 Project Overview

This project implements a supervised binary classification model to predict **in-hospital death (hospdead)** for seriously ill hospitalized patients. The analysis includes thorough exploratory data analysis, statistical feature selection, data preprocessing, transformation, and comprehensive model evaluation with multiple configurations.

## 👥 Group Members

- **Chantouch Orungrote** (ID: 66340500011)
- **Sasish Kaewsing** (ID: 66340500076)

## 📊 Dataset

**Source:** [SUPPORT2 – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/880/support2)

The SUPPORT2 dataset contains medical data from **9,105 seriously ill hospitalized patients** from five U.S. states, collected across two periods (1989-1991 and 1992-1994). The dataset includes 47 features covering patient demographics, medical conditions, laboratory values, and clinical outcomes.

### Target Variable
- **hospdead**: Binary indicator of in-hospital death (yes/no)

### Key Features
- **Demographics:** age, sex, race, education, income
- **Medical Conditions:** disease categories, comorbidities, cancer status
- **Clinical Measurements:** blood pressure, heart rate, lab values, APACHE scores
- **Treatment Information:** DNR status, hospital charges, ICU indicators
- **Functional Status:** ADL scores, disability indices

## 🔧 Technologies & Libraries

- **Python 3.12+**
- **Data Processing:** pandas, numpy
- **Statistical Analysis:** scipy, statsmodels
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Data Fetching:** ucimlrepo

## 🚀 Features & Methodology

### 1. Data Loading & Description
- Comprehensive data dictionary with 47 variables
- Duplicate detection and removal
- Initial data structure analysis

### 2. Problem Statement Definition
- **Task Type:** Supervised Binary Classification
- **Algorithm:** Logistic Regression
- **Model Assumptions:** Linearity, no multicollinearity, normality of residuals, no influential outliers, no autocorrelation, homoscedasticity

### 3. Target Distribution Analysis
- Class imbalance assessment
- Distribution visualization

### 4. Basic Data Exploration
- Statistical summaries for numerical and categorical features
- Data type identification
- Missing value assessment

### 5. Visual Exploratory Data Analysis (EDA)
- **Categorical Variables:**
  - Bar chart visualization
  - Interpretation of distributions
  
- **Continuous Variables:**
  - Histogram analysis
  - Box plot analysis
  - Outlier identification
  
- **Data Leakage Detection:** Identifying and removing features that could cause leakage

### 6. Outlier Treatment
- IQR (Interquartile Range) method for outlier detection
- Statistical outlier handling

### 7. Missing Value Treatment
- **Continuous Features:**
  - Option 1: Removal of missing values
  - Option 2: Imputation with mean
  - Option 3: Imputation with median
  
- **Categorical Features:**
  - Option 1: Removal of missing values
  - Option 2: Mode-based imputation

- Post-treatment visualization analysis

### 8. Feature Selection

#### 8.1. Categorical vs Categorical Relationships
- **Visual Exploration:** Grouped bar plots
- **Statistical Tests:**
  - Chi-square test
  - Mutual information
  - Combined statistical assessment

#### 8.2. Categorical vs Continuous Relationships
- **Visual Exploration:** Box plots by category
- **Statistical Tests:**
  - ANOVA (Analysis of Variance)
  - Point-biserial correlation
  - Combined statistical assessment

### 9. Final Feature Selection
Selected features based on statistical significance and domain knowledge:
- **Ordinal:** sfdm2 (functional disability)
- **Nominal:** dnr (do-not-resuscitate status)
- **Numerical:** d.time, avtisst, charges, hday

### 10. Data Preprocessing

#### 10.1. Ordinal Encoding
- Converting ordinal variables to numeric representation

#### 10.2. Binary Encoding
- 1/0 mapping for binary variables

#### 10.3. One-Hot Encoding
- Creating dummy variables for nominal categories

#### 10.4. Data Transformation
- **Skewness Correction:** Log transformation for highly skewed features
- **Standardization:** StandardScaler for feature scaling

### 11. Model Construction - Model 1 (Without Outlier Treatment)
- Logistic Regression with liblinear solver
- Training set: 70%
- Testing set: 30%

### 12. Model Construction - Model 2 (With Outlier Treatment)
- Complete preprocessing pipeline with outlier removal
- Same feature selection and transformation process
- Comparative analysis with Model 1

### 13. Comprehensive Model Evaluation

#### Evaluation Metrics
- **Confusion Matrix:** True positives, false positives, true negatives, false negatives
- **Accuracy:** Overall prediction correctness
- **Precision:** Positive predictive value
- **Recall (Sensitivity):** True positive rate
- **F1-Score:** Harmonic mean of precision and recall
- **Specificity:** True negative rate
- **ROC Curve:** Receiver Operating Characteristic analysis
- **AUC:** Area Under the Curve
- **Classification Report:** Detailed performance breakdown

#### Analysis Topics
1. **Overfitting Assessment:** Comparing training vs testing performance
2. **Model Accuracy Comparison:** Evaluating both models
3. **Real-World Applicability:** Determining if the model is production-ready
4. **Feature Importance:** Identifying significant predictors
5. **Settings Impact:** Analyzing how different configurations affect performance
6. **Error Analysis:** Understanding model limitations and data characteristics
7. **Model Improvement Strategies:** Recommendations for enhancement

## 📈 Key Results & Visualizations

The notebook includes extensive visualizations:
- Distribution plots (histograms and box plots)
- Grouped bar charts for categorical analysis
- Correlation heatmaps
- ROC curves with threshold markers
- Confusion matrices for both models
- Coefficient plots showing feature importance
- Residual analysis plots

## 🎯 Model Performance Highlights

- Two complete model implementations (with and without outlier treatment)
- Multiple threshold analysis (0.3, 0.5, 0.7)
- Comprehensive comparison of training and testing metrics
- Statistical validation of model assumptions
- Feature coefficient interpretation

## 📁 Project Structure

```
├── HW02_6611_6676.ipynb     # Main analysis notebook
├── README.md                # This file
└── support2_raw.pkl         # Cached raw dataset (generated)
```

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn ucimlrepo
```

### Running the Notebook
1. Clone or download the repository
2. Open `HW02_6611_6676.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells sequentially
4. Review the comprehensive analysis, visualizations, and model evaluations

## 🔍 Key Insights

- **Feature Selection:** Statistical methods (Chi-square, ANOVA, correlation) effectively identify relevant predictors
- **Data Quality:** Proper handling of missing values and outliers significantly impacts model performance
- **Transformation:** Log transformation and standardization improve model convergence and performance
- **Class Imbalance:** Important consideration for healthcare predictions
- **Model Interpretability:** Logistic regression provides transparent, interpretable coefficients
- **Threshold Selection:** Different thresholds serve different clinical objectives (sensitivity vs specificity trade-off)

## 📊 Statistical Methods Used

- **Chi-square test** for categorical independence
- **Mutual Information** for non-linear relationships
- **ANOVA** for group mean comparisons
- **Point-biserial correlation** for binary-continuous relationships
- **IQR method** for outlier detection
- **Skewness analysis** for distribution assessment

## 🎓 Learning Outcomes

This project demonstrates:
- Complete machine learning pipeline from raw data to model evaluation
- Statistical rigor in feature selection
- Proper handling of data quality issues
- Understanding of logistic regression assumptions
- Comprehensive model evaluation techniques
- Real-world considerations for healthcare ML applications

## 📝 Notes

- The notebook includes detailed Thai language comments for accessibility
- All visualizations are optimized for clarity and interpretation
- Code follows best practices for reproducibility
- Statistical tests are properly documented with interpretations

## ⚠️ Limitations & Future Work

- Limited to logistic regression (could explore ensemble methods)
- Binary classification (could extend to multi-class outcomes)
- Feature engineering opportunities (interaction terms, polynomial features)
- Cross-validation for more robust performance estimates
- Hyperparameter tuning for optimization
- External validation on independent datasets

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the group members.

## 📄 License

This project is part of academic coursework. Please respect academic integrity policies when referencing this work.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the SUPPORT2 dataset
- Course instructors and teaching assistants for guidance
- scikit-learn and scipy communities for excellent documentation
- Original SUPPORT study researchers

---

**Last Updated:** February 2026  
**Course:** Machine Learning  
**Assignment:** Homework 2 - Binary Classification
