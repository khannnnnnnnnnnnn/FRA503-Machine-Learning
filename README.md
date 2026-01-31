# Machine Learning Project: Bank Marketing Campaign Classification

A comprehensive comparative analysis of four machine learning classifiers on bank marketing data, evaluating the impact of outlier treatment and feature transformation across 16 unique model-data configurations.

## 📋 Project Overview

This project implements a systematic experimental study to predict **term deposit subscription** from direct marketing campaign data. The analysis features a rigorous 4×4 factorial design comparing four classification algorithms across different data preprocessing regimes, with a focus on ROC-AUC performance metrics.

## 👥 Group Members

- **Chantouch Orungrote** (ID: 66340500011)
- **Penpitcha Chobchon** (ID: 66340500035)
- **Sasish Kaewsing** (ID: 66340500076)

## 📊 Dataset

**Source:** [Bank Marketing – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The dataset contains marketing campaign data from a **Portuguese banking institution** with **45,211 samples** and **17 features** covering client demographics, financial information, and campaign contact details.

### Target Variable
- **y**: Binary indicator of term deposit subscription (yes/no)

### Feature Categories

#### Client Demographics (4 features)
- **age**: Client's age in years
- **job**: Occupation type (admin, management, technician, services, etc.)
- **marital**: Marital status (married, divorced, single)
- **education**: Education level (primary, secondary, tertiary, unknown)

#### Financial Information (3 features)
- **default**: Credit in default status (yes/no)
- **balance**: Average yearly account balance in euros
- **housing**: Housing loan status (yes/no)
- **loan**: Personal loan status (yes/no)

#### Campaign Contact Details (8 features)
- **contact**: Communication channel (cellular, telephone, unknown)
- **day**: Last contact day of the month
- **month**: Last contact month
- **duration**: Last contact duration in seconds
- **campaign**: Number of contacts during current campaign
- **pdays**: Days since last contact from previous campaign (-1 if never contacted)
- **previous**: Number of contacts before this campaign
- **poutcome**: Previous campaign outcome (success, failure, other, unknown)

## 🔧 Technologies & Libraries

- **Python 3.x**
- **Data Processing:** pandas, numpy
- **Statistical Analysis:** scipy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Feature Selection:** chi2, mutual_info_classif, f_oneway
- **Data Fetching:** ucimlrepo

## 🎯 Experimental Design

### 2.1 Problem Definition
- **Type:** Supervised Binary Classification
- **Objective:** Predict term deposit subscription
- **Approach:** Comparative analysis across multiple algorithms and data preparation strategies

### 2.2 Four Classification Algorithms

1. **Logistic Regression (LogReg)**
   - Linear decision boundary
   - Probabilistic predictions
   - Requires feature scaling

2. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Distance-sensitive (requires normalization)
   - Non-parametric approach

3. **Decision Tree**
   - Hierarchical rule-based classification
   - Handles non-linear relationships
   - Feature scaling not required

4. **Random Forest**
   - Ensemble of decision trees
   - Reduces overfitting through bagging
   - Provides feature importance

### 2.3 Data Preparation Regimes (4×4 Factorial Design)

The experiment employs a systematic comparison across four preprocessing pipelines:

| Pipeline | Outlier Treatment (OT) | Feature Scaling (T) | Description |
|----------|------------------------|---------------------|-------------|
| **OT+T** | ✅ Yes | ✅ Yes | Fully preprocessed: Outliers removed & features scaled |
| **OT+NT** | ✅ Yes | ❌ No | Outliers removed, features retain original scale |
| **NOT+T** | ❌ No | ✅ Yes | Outliers retained, features scaled |
| **NOT+NT** | ❌ No | ❌ No | Baseline: Raw data without preprocessing |

### 2.4 Evaluation Framework
- **Total Configurations:** 16 unique combinations (4 models × 4 pipelines)
- **Validation Strategy:** 5-fold cross-validation
- **Primary Metric:** Mean ROC-AUC Score
- **Additional Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 🚀 Methodology & Pipeline

### 1. Data Loading & Initial Exploration
- Dataset import from UCI repository
- Duplicate detection and removal
- Basic structure analysis

### 2. Problem Statement & Experimental Design
- Binary classification problem definition
- 4×4 factorial experimental setup
- Model assumptions documentation

### 3. Target Variable Analysis
- Distribution of term deposit subscriptions
- Class imbalance assessment

### 4. Basic Feature Exploration
- Statistical summaries for all features
- Data type identification
- Missing value assessment
- Initial insights

### 5. Visual Exploratory Data Analysis (EDA)

#### 5.1 Categorical Features
- Bar chart visualization
- Distribution patterns
- Category frequency analysis

#### 5.2 Continuous Features
- **Histogram Analysis:** Distribution shapes, skewness
- **Box Plot Analysis:** Outlier detection, quartiles
- Interpretation of patterns

### 6. Outlier Treatment

#### 6.1 Outlier Analysis
- Identification of extreme values
- Impact assessment on model performance

#### 6.2 Feature Engineering: pdays
- **Problem:** pdays = -1 represents "never contacted" (999 encoding issue)
- **Solution:** Split into two features:
  - `contact_flag`: Binary indicator (0 = never contacted, 1 = previously contacted)
  - `pdays_actual`: Actual days since last contact (for contacted clients only)

#### 6.3 Outlier Treatment Methods
- IQR-based outlier capping/removal
- Visual validation after treatment

### 7. Missing Value Treatment

#### 7.1 Continuous Features
- Strategy: Remove rows with missing values (minimal impact)
- Validation of completeness

#### 7.2 Categorical Features
- **Option 1:** Row removal
- **Option 2:** Mode-based imputation (selected)
- Post-treatment visualization

#### 7.3 Train-Test Split
- 70% training, 30% testing
- Separate splits for outlier-treated and non-treated data
- Distribution validation

### 8. Feature Selection (Statistical & Visual Analysis)

#### 8.1 For Outlier-Treated Data

**8.1.1 Categorical vs Categorical Relationships**
- **Visual:** Grouped bar plots
- **Statistical:** Chi-square test, Mutual Information
- **Selected Features:** Job, marital, education, contact, month, poutcome, housing, loan, default

**8.1.2 Categorical vs Continuous Relationships**
- **Visual:** Box plots by category
- **Statistical:** ANOVA, Pearson correlation, Mutual Information
- **Selected Features:** Age, balance, day, campaign, previous, duration

#### 8.2 For Non-Outlier-Treated Data
- Same methodology applied
- Feature selection validation
- Comparison with outlier-treated results

### 9. Final Feature Selection Summary
- Comprehensive list of selected features for each data regime
- Rationale for inclusion/exclusion

### 10. Data Preprocessing

#### 10.1 For Outlier-Treated Data

**10.1.1 Categorical Encoding**
- Binary variables: 1/0 mapping (default, housing, loan, contact_flag)
- Nominal variables: One-hot encoding (job, marital, education, contact, month, poutcome)

**10.1.2 Data Transformation**
- **Skewness Correction:** Log/square root transformations for right-skewed features
- **Normalization:** StandardScaler for feature scaling
- Validation of transformed distributions

#### 10.2 For Non-Outlier-Treated Data
- Same preprocessing steps applied
- Separate transformation pipelines

### 11. DataFrames Summary
- Overview of all preprocessing pipelines
- Four final datasets ready for modeling:
  1. OT_T (Outlier-Treated + Transformed)
  2. OT_NT (Outlier-Treated + Non-Transformed)
  3. NOT_T (Non-Outlier-Treated + Transformed)
  4. NOT_NT (Non-Outlier-Treated + Non-Transformed)

### 12. Model Construction

#### 12.1 Logistic Regression
- Implementation across all 4 data configurations
- 5-fold cross-validation
- ROC-AUC evaluation

#### 12.2 K-Nearest Neighbors (KNN)
- Distance-based classification
- Sensitivity to feature scaling analyzed
- Optimal k selection

#### 12.3 Decision Tree
- Tree-based rule extraction
- Comparison across data regimes
- Feature importance analysis

#### 12.4 Random Forest
- Ensemble approach
- Variance reduction through bagging
- Feature importance ranking

#### 12.5 ROC Curve Plotting
- Visual comparison of all 16 configurations
- AUC scores for each model-data combination

#### 12.6 Hyperparameter & Threshold Tuning
- Grid search for optimal parameters
- Threshold optimization based on business objectives
- Trade-off analysis between precision and recall

### 13. Comprehensive Model Analysis

#### 13.1 Performance Metrics Definition
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC interpretation
- Confusion matrix elements

#### 13.2 Problem Context & Metric Selection
- **Business Context:** Marketing campaign optimization
- **Cost Consideration:** False positives (wasted contact) vs False negatives (missed opportunity)
- **Metric Priority:** Precision-focused for resource optimization

#### 13.3 Critical Questions Answered

**Q1: Overfitting Assessment**
- Training vs testing performance comparison
- Model generalization evaluation
- Variance-bias trade-off analysis

**Q2: Impact of Preprocessing Settings**
- Does outlier treatment improve performance?
- Does feature transformation affect results?
- Interaction effects between treatments

**Q3: Model Performance & Precision Comparison**
- Which model achieves highest precision?
- Performance across different metrics
- Model ranking by configuration

**Q4: Real-World Applicability**
- Can models be deployed in production?
- Reliability assessment
- Practical limitations

**Q5: Feature Importance & Sufficiency**
- Which features drive predictions?
- Minimum feature set for high accuracy
- Proposed streamlined feature subset

**Q6: Error Analysis & Model Limitations**
- What data characteristics reduce effectiveness?
- Class imbalance impact
- High precision cost (reduced recall)
- Concrete examples from optimal threshold analysis

## 📈 Key Visualizations

The notebook includes extensive visualizations:
- Distribution plots for all features (histograms & box plots)
- Grouped bar charts for categorical relationships
- Correlation heatmaps
- ROC curves for all 16 model-data configurations
- Confusion matrices for optimal models
- Feature importance plots (Random Forest & Decision Tree)
- Threshold tuning curves
- Comparative performance charts

## 🎯 Key Results & Insights

### Model Performance Highlights
- **Best Overall Model:** [Based on ROC-AUC scores across configurations]
- **Highest Precision:** KNN with OT_T configuration (0.537)
- **Preprocessing Impact:** Feature scaling significantly benefits distance-based models (KNN, LogReg)
- **Outlier Treatment:** Shows measurable improvement in model stability

### Feature Importance
- **Duration:** Strongest predictor (but potential data leakage concern)
- **Balance, Age, Campaign:** Important demographic/financial indicators
- **Previous Campaign Outcome:** Strong historical indicator
- **Contact Method:** Communication channel affects success rates

### Practical Recommendations
- **Optimal Configuration:** Model-specific preprocessing requirements identified
- **Feature Engineering:** pdays splitting improves interpretability
- **Threshold Selection:** Business context determines optimal operating point
- **Deployment Considerations:** Precision-recall trade-offs for campaign efficiency

## 📁 Project Structure

```
├── Project_ML_6611_6635_6676.ipynb    # Main analysis notebook
├── README.md                          # This file
```

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn ucimlrepo
```

### Running the Notebook
1. Clone or download the repository
2. Open `Project_ML_6611_6635_6676.ipynb` in Jupyter Notebook
3. Run all cells sequentially
4. Review the comprehensive analysis across all 16 model configurations

## 🔍 Key Takeaways

1. **Systematic Experimentation:** 4×4 factorial design provides robust comparative analysis
2. **Preprocessing Matters:** Feature scaling critically impacts distance-based algorithms
3. **Algorithm Selection:** Tree-based models more robust to outliers than linear models
4. **Feature Engineering:** Domain-specific transformations (pdays splitting) enhance interpretability
5. **Business Context:** Metric selection should align with campaign optimization goals
6. **Model Complexity:** Random Forest provides best balance of performance and robustness
7. **Real-World Deployment:** Precision-recall trade-off requires careful threshold calibration

## 📊 Statistical Methods Used

- **Chi-square Test:** Categorical independence
- **Mutual Information:** Non-linear feature-target relationships
- **ANOVA:** Group mean comparisons
- **Pearson Correlation:** Linear relationships
- **ROC-AUC Analysis:** Model discrimination ability
- **Cross-Validation:** Robust performance estimation

## 🎓 Academic Rigor

This project demonstrates:
- Controlled experimental design (factorial approach)
- Comprehensive EDA with statistical validation
- Proper train-test separation to prevent data leakage
- Multiple evaluation metrics for thorough assessment
- Reproducible analysis with clear documentation
- Critical thinking about real-world applicability
- Business-oriented interpretation of technical results

## ⚠️ Limitations & Future Work

- **Class Imbalance:** Highly imbalanced dataset (88% no vs 12% yes)
- **Duration Feature:** Potential data leakage as it's known only after call
- **Temporal Aspects:** Time-series nature not fully exploited
- **Feature Interactions:** Limited exploration of interaction effects
- **Advanced Methods:** Could explore ensemble stacking, boosting
- **Calibration:** Probability calibration not addressed
- **Cost-Sensitive Learning:** Could incorporate asymmetric misclassification costs

### Recommendations for Enhancement
1. Implement SMOTE or other resampling techniques for class imbalance
2. Explore deep learning approaches (neural networks)
3. Conduct temporal validation (time-based split)
4. Investigate feature interactions systematically
5. Implement cost-sensitive learning frameworks
6. Deploy A/B testing framework for production validation


---

**Course:** Machine Learning  
**Assignment:** Final Project - Classification Model Comparison
