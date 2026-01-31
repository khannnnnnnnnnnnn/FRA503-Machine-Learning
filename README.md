# Machine Learning Course Projects

A collection of three comprehensive machine learning projects analyzing the SUPPORT2 and Bank Marketing datasets, covering regression and classification techniques with systematic experimental approaches.

## 📚 Projects Overview

### 1️⃣ Homework 1: Linear Regression Analysis
**File:** `HW01_6611_6676.ipynb`

Predicting **hospital charges** using linear regression on the SUPPORT2 dataset.

**Key Features:**
- Two model implementations (with/without outlier treatment)
- Statistical feature selection (Pearson correlation, Mutual Information, ANOVA)
- Log transformation for skewed distributions
- Comprehensive residual analysis
- Evaluation: R², MAE, MSE, RMSE, MAPE

**Dataset:** 9,105 patients | 47 features | Target: charges (continuous)

---

### 2️⃣ Homework 2: Logistic Regression Classification
**File:** `HW02_6611_6676.ipynb`

Predicting **in-hospital mortality** using logistic regression on the SUPPORT2 dataset.

**Key Features:**
- Binary classification (hospdead: yes/no)
- Two model configurations with different preprocessing approaches
- Chi-square and Mutual Information for feature selection
- ROC-AUC analysis with multiple threshold evaluation
- Comprehensive confusion matrix and precision-recall analysis

**Dataset:** 9,105 patients | 47 features | Target: hospdead (binary)

---

### 3️⃣ Final Project: Multi-Model Classification Comparison
**File:** `Project_ML_6611_6635_6676.ipynb`

Predicting **term deposit subscription** with systematic comparison of four classifiers on the Bank Marketing dataset.

**Key Features:**
- **4×4 Factorial Design:** 4 models × 4 preprocessing pipelines = 16 configurations
- **Models:** Logistic Regression, KNN, Decision Tree, Random Forest
- **Pipelines:** Outlier Treatment (OT) × Feature Transformation (T) combinations
- 5-fold cross-validation with ROC-AUC as primary metric
- Feature engineering (pdays splitting into contact_flag + pdays_actual)

**Dataset:** 45,211 samples | 17 features | Target: y (binary - term deposit subscription)

---

## 🔧 Technologies

- **Python:** 3.12+
- **Core Libraries:** pandas, numpy, scipy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Data Source:** ucimlrepo

## 📊 Datasets

### SUPPORT2 (HW1 & HW2)
Medical data from seriously ill hospitalized patients across five U.S. states (1989-1994). Contains demographics, clinical measurements, lab values, and outcomes.

### Bank Marketing (Project)
Direct marketing campaign data from a Portuguese bank. Contains client demographics, financial information, and campaign contact details.

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn ucimlrepo

# Run any notebook
jupyter notebook HW01_6611_6676.ipynb
# or
jupyter notebook HW02_6611_6676.ipynb
# or
jupyter notebook Project_ML_6611_6635_6676.ipynb
```

## 📁 Repository Structure

```
├── HW01_6611_6676.ipynb              # Homework 1: Linear Regression
├── HW02_6611_6676.ipynb              # Homework 2: Logistic Regression
├── Project_ML_6611_6635_6676.ipynb   # Final Project: Multi-Model Comparison
├── README.md                         # Main README (overview)
```

## 🎯 Key Learning Outcomes

### Regression (HW1)
- Linear regression assumptions and validation
- Handling skewness with log transformations
- Outlier treatment impact on model performance
- Residual analysis and error interpretation

### Binary Classification (HW2)
- Logistic regression for healthcare predictions
- Feature selection with statistical tests
- ROC-AUC and confusion matrix analysis
- Precision-recall trade-offs

### Multi-Model Comparison (Project)
- Systematic experimental design (factorial approach)
- Comparing distance-based, linear, and tree-based models
- Impact of preprocessing on different algorithms
- Business-oriented metric selection

## 📈 Methodology Highlights

All projects follow rigorous data science workflows:
1. **EDA:** Comprehensive exploratory analysis with visualizations
2. **Feature Selection:** Statistical validation (Chi-square, ANOVA, Correlation)
3. **Preprocessing:** Missing values, outliers, encoding, scaling
4. **Model Evaluation:** Multiple metrics with cross-validation
5. **Critical Analysis:** Overfitting, limitations, real-world applicability

## 📝 Notes

- Each project has detailed documentation in its respective README file
- All notebooks include Thai language comments for accessibility
- Code follows reproducibility best practices
- Statistical methods are properly documented with interpretations

---

**Academic Year:** 2025  
**Course:** Machine Learning  
**Institution:** [KMUTT]
