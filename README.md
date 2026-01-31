# Machine Learning HW1: Linear Regression Analysis

A comprehensive linear regression project analyzing the SUPPORT2 dataset to predict hospital charges with extensive exploratory data analysis, feature selection, and model evaluation.

## 📋 Project Overview

This project implements supervised regression analysis to predict **hospital charges** for seriously ill hospitalized patients. The analysis includes thorough exploratory data analysis, statistical feature selection, data preprocessing, transformation, and comprehensive model evaluation with two different model configurations.

## 👥 Group Members

- **Chantouch Orungrote** (ID: 66340500011)
- **Sasish Kaewsing** (ID: 66340500076)

## 📊 Dataset

**Source:** [SUPPORT2 – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/880/support2)

The SUPPORT2 dataset contains medical data from **9,105 seriously ill hospitalized patients** from five U.S. states, collected across two periods (1989-1991 and 1992-1994). The dataset includes 47 features covering patient demographics, medical conditions, laboratory values, and clinical outcomes.

### Target Variable
- **charges**: Total hospital charges for each patient (continuous variable)

### Key Predictor Features
- **Demographics:** age, sex, race, education, income
- **Medical Conditions:** disease categories, comorbidities, cancer status
- **Clinical Measurements:** blood pressure, heart rate, respiratory rate, temperature, lab values
- **Hospital Stay:** length of stay (slos), study entry day (hday), follow-up days (d.time)
- **Severity Scores:** APACHE scores (aps), SUPPORT scores (sps), TISS scores (avtisst)
- **Functional Status:** ADL scores, disability indices

## 🔧 Technologies & Libraries

- **Python 3.12+**
- **Data Processing:** pandas, numpy
- **Statistical Analysis:** scipy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Data Fetching:** ucimlrepo

## 🚀 Features & Methodology

### 1. Data Loading & Description
- Comprehensive data dictionary with 47 variables
- Dataset structure analysis
- Initial exploration

### 2. Problem Statement Definition
- **Task Type:** Supervised Regression
- **Algorithm:** Linear Regression
- **Objective:** Predict hospital charges for all hospitalized patients

### 3. Linear Regression Assumptions
The project validates six key assumptions:
1. **Linearity:** Linear relationship between predictors and target
2. **Multicollinearity:** Predictors should not be highly correlated
3. **Normality of Residuals:** Errors follow normal distribution (mean=0)
4. **No Influential Outliers:** Outliers should not dominate the model
5. **No Autocorrelation:** Residuals should be independent
6. **Homoscedasticity:** Constant variance of residuals

### 4. Target Distribution Analysis
- Distribution visualization of hospital charges
- Skewness assessment
- Log transformation consideration

### 5. Basic Data Exploration
- Statistical summaries for all features
- Data type identification
- Missing value assessment
- Initial insights

### 6. Visual Exploratory Data Analysis (EDA)

#### 6.1. Categorical Variables
- Bar chart visualization for all categorical features
- Distribution interpretation
- Pattern identification

#### 6.2. Continuous Variables
- **Histogram Analysis:** Distribution shapes, skewness patterns
- **Box Plot Analysis:** Outlier detection, quartile ranges
- Comprehensive interpretation of distributions

#### 6.3. Data Leakage Detection
- Identifying features that could cause leakage
- Removing problematic features (e.g., totcst, totmcst derived from charges)

### 7. Outlier Treatment
- Outlier detection and analysis
- Decision on outlier handling strategy

### 8. Missing Value Treatment

#### 8.1. Continuous Features
- **Option 1:** Removing rows with missing values
- **Option 2:** Filling with mean
- **Option 3:** Filling with median (selected approach)
- Post-treatment validation

#### 8.2. Categorical Features
- **Option 1:** Removing rows with missing values
- **Option 2:** Filling with mode (selected approach)
- Post-treatment visualization

#### 8.3. Train-Test Split
- 70% training, 30% testing
- Distribution validation across splits

### 9. Feature Selection by Visual Correlation & Statistical Analysis

#### 9.1. Continuous vs Continuous Relationships
- **Visual Exploration:** Scatter plots for all numeric features
- **Statistical Methods:**
  - Pearson correlation coefficient
  - Mutual information
  - Correlation heatmap analysis

**Selected Features (Strong/Moderate Correlation):**
- Strong: d.time, avtisst, aps
- Weak: age, slos, edu, hday, dnrday, wblc, resp, pafi, bili, crea, glucose, bun, adlsc

#### 9.2. Continuous vs Categorical Relationships
- **Visual Exploration:** Box plots grouped by categories
- **Statistical Methods:**
  - ANOVA (Analysis of Variance)
  - Effect size analysis
  - Point-biserial correlation

**Selected Categorical Features:**
- dzgroup, race, ca

### 10. Final Feature Selection Summary
Selected features based on combined statistical and visual analysis.

### 11. Data Preprocessing for Machine Learning

#### 11.1. Feature Encoding
- **Ordinal Encoding:** For ordered categorical variables
- **Binary Encoding:** 1/0 mapping for binary variables
- **One-Hot Encoding:** Dummy variables for nominal categories

#### 11.2. Data Transformation
- **Log Transformation:** Applied to highly skewed features including target variable (charges)
- **Standardization:** StandardScaler for numeric features
- Transformation validation

### 12. Model Construction

#### Model 1: Without Outlier Treatment
- Linear regression with standardization only
- Applied log transformation on target
- Considers non-linear relationships

#### Model 2: With Outlier Treatment
- Outlier removal on doubtful predictors
- Complete preprocessing pipeline
- Improved data quality

### 13. Model Evaluation & Performance Metrics

#### Training Metrics
- **R² Score:** Coefficient of determination
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

#### Testing Metrics
- Same metrics applied to test set
- Generalization assessment

#### Visualization
- **Actual vs Predicted Scatter Plots**
- **Residual Plots:** Residual vs Actual values
- Error distribution analysis

### 14. Comprehensive Model Analysis

The project answers seven critical questions:

#### 1. Overfitting Assessment
- Comparing training vs testing R² scores
- Cost function analysis for both models
- Identifying overfitting indicators

#### 2. Model Accuracy Comparison
- Which model performs better?
- Why does one model outperform the other?
- Statistical evidence for model selection

#### 3. Real-World Applicability
- Can the model be deployed in production?
- What are the practical limitations?
- Reliability assessment

#### 4. Feature Importance
- Which features contribute most?
- Coefficient analysis
- Feature sufficiency evaluation

#### 5. Settings Impact Analysis
- How do different preprocessing steps affect performance?
- Comparison of Model 1 vs Model 2
- Impact of outlier treatment

#### 6. Error Analysis & Limitations
- What data characteristics reduce model effectiveness?
- Residual pattern analysis
- Non-linear cost escalation challenges
- Model limitations with extreme high-cost cases

#### 7. Model Improvement Strategies
- Recommendations for enhancement
- Potential advanced techniques
- Future research directions

## 📈 Key Visualizations

The notebook includes comprehensive visualizations:
- Distribution plots (histograms and box plots) for all variables
- Scatter plots showing feature-target relationships
- Correlation heatmaps
- Grouped bar charts for categorical analysis
- Actual vs Predicted plots for both models
- Residual analysis plots
- Feature coefficient comparisons

## 🎯 Model Performance Highlights

- **Two Complete Models:** With and without outlier treatment
- **Comprehensive Metrics:** R², MAE, MSE, RMSE, MAPE
- **Visual Validation:** Multiple plot types for thorough analysis
- **Statistical Rigor:** Hypothesis testing and correlation analysis
- **Practical Insights:** Real-world applicability assessment

## 📁 Project Structure

```
├── HW01_6611_6676.ipynb     # Main analysis notebook
├── README.md                # This file
└── support2_raw.pkl         # Cached raw dataset (optional)
```

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn ucimlrepo
```

### Running the Notebook
1. Clone or download the repository
2. Open `HW01_6611_6676.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells sequentially
4. Review the comprehensive analysis, visualizations, and model evaluations

## 🔍 Key Insights

- **Feature Selection Matters:** Statistical correlation analysis effectively identifies relevant predictors
- **Data Quality Impact:** Proper handling of missing values and outliers significantly affects performance
- **Transformation Benefits:** Log transformation helps address skewness in charges distribution
- **Linear Regression Limitations:** Model struggles with non-linear cost escalation, especially for extreme high-cost patients
- **Model Comparison:** Outlier treatment improves accuracy but doesn't fully resolve non-linearity issues
- **Practical Considerations:** Both models show limitations for real-world deployment due to residual patterns

## 📊 Statistical Methods Used

- **Pearson Correlation:** Linear relationship measurement
- **Mutual Information:** Non-linear relationship detection
- **ANOVA:** Group mean comparisons for categorical variables
- **Point-biserial Correlation:** Binary-continuous relationships
- **IQR Method:** Outlier detection
- **Skewness Analysis:** Distribution assessment

## 🎓 Learning Outcomes

This project demonstrates:
- Complete linear regression pipeline from raw data to model evaluation
- Rigorous statistical feature selection methodology
- Proper validation of regression assumptions
- Data quality management (missing values, outliers, transformations)
- Comprehensive model evaluation and comparison
- Critical analysis of model limitations
- Real-world applicability assessment for healthcare cost prediction

## 📝 Notes

- The notebook includes detailed interpretations and summaries throughout
- All visualizations are optimized for clarity and insight
- Code follows best practices for reproducibility
- Statistical tests are properly documented with business interpretations

## ⚠️ Limitations & Future Work

- **Non-linearity:** Linear regression struggles with extreme cost escalation
- **Model Complexity:** Could explore polynomial features or non-linear models
- **Advanced Techniques:** Tree-based models or ensemble methods might capture non-linear patterns better
- **Feature Engineering:** Interaction terms and domain-specific features could improve predictions
- **Cross-Validation:** k-fold CV for more robust performance estimates
- **Regularization:** Ridge/Lasso regression to handle multicollinearity
- **External Validation:** Testing on independent datasets from different hospitals

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

**Course:** Machine Learning  
**Assignment:** Homework 1 - Linear Regression Analysis
