# Machine Learning HW1: SUPPORT2 Dataset Analysis

A comprehensive machine learning project analyzing the SUPPORT2 dataset using multiple regression models and ensemble techniques.

## 📋 Project Overview

This project implements and compares various machine learning algorithms to predict patient charges from the SUPPORT2 dataset. The analysis includes data preprocessing, feature engineering, model training, hyperparameter tuning, and extensive performance evaluation.

## 👥 Group Members

- **Chantouch Orungrote** (ID: 66340500011)
- **Sasish Kaewsing** (ID: 66340500076)

## 📊 Dataset

**Source:** [SUPPORT2 – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/880/support2)

The SUPPORT2 dataset contains medical data about seriously ill hospitalized patients. The target variable is `charges`, representing the total hospital charges.

## 🔧 Technologies & Libraries

- **Python 3.13**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Data Fetching:** ucimlrepo

## 🚀 Features

### Data Preprocessing
- Handling missing values
- Outlier detection and treatment using IQR method
- Feature scaling and normalization
- Categorical variable encoding (one-hot encoding)
- Train-test split (80-20)

### Models Implemented

1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **K-Nearest Neighbors (KNN)**
5. **Decision Tree Regressor**
6. **Random Forest Regressor**
7. **Gradient Boosting Regressor**
8. **Support Vector Regression (SVR)**
9. **Ensemble Methods:**
   - Voting Regressor
   - Stacking Regressor

### Hyperparameter Tuning

All models undergo rigorous hyperparameter optimization using:
- **GridSearchCV** with 5-fold cross-validation
- Custom parameter grids for each algorithm
- Performance-based model selection

### Evaluation Metrics

- **R² Score** (Coefficient of Determination)
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- Residual analysis with visualization

## 📈 Analysis Pipeline

1. **Data Loading & Exploration**
   - Dataset structure analysis
   - Statistical summaries
   - Missing value assessment

2. **Data Cleaning**
   - Outlier detection using IQR (Interquartile Range)
   - Missing value imputation
   - Data type conversions

3. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Feature scaling using StandardScaler
   - Feature selection

4. **Model Training**
   - Individual model training
   - Hyperparameter optimization
   - Ensemble model construction

5. **Model Evaluation**
   - Performance comparison across all models
   - Residual plot analysis
   - Prediction vs actual value visualization

## 📊 Key Results

The notebook includes comprehensive visualizations:
- Residual plots for training and testing data
- Actual vs Predicted scatter plots
- Model performance comparison tables
- Feature importance analysis (for tree-based models)

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

### Running the Notebook
1. Clone or download the repository
2. Open `Homework1.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells sequentially
4. Review the output, visualizations, and model performance metrics

## 📁 Project Structure

```
├── Homework1.ipynb          # Main analysis notebook
├── README.md                # This file
```

## 🎯 Model Performance

Each model is evaluated on both training and testing datasets with the following metrics:
- **Training Performance:** Assesses model fit
- **Testing Performance:** Evaluates generalization capability
- **Residual Analysis:** Identifies prediction patterns and biases

## 🔍 Key Insights

- Comprehensive comparison of linear, tree-based, and ensemble methods
- Ensemble methods (Voting and Stacking) often provide robust predictions
- Hyperparameter tuning significantly impacts model performance
- Residual analysis helps identify model limitations and areas for improvement

## 📝 Notes

- The notebook includes Thai language comments for accessibility
- All visualizations are optimized for clear interpretation
- The code follows best practices for reproducibility

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the group members.

## 📄 License

This project is part of academic coursework. Please respect academic integrity policies when referencing this work.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the SUPPORT2 dataset
- Course instructors and teaching assistants for guidance
- scikit-learn community for excellent documentation

---

**Last Updated:** February 2026
