# 💼 Employee-Salary-Prediction-FCIS
This project predicts whether a person earns more than 50K per year using the [Adult Census dataset](https://www.kaggle.com/datasets/uciml/adult-census-income/data). it includes data preprocessing, encoding categorical features, and training classification models to build an accurate prediction system.

## 📊 Dataset
 
The project uses the **Adult Income Dataset** (also known as the Census Income dataset), a classic benchmark for income prediction tasks.
 
| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `workclass` | Employment type (private, government, etc.) |
| `education` | Highest level of education attained |
| `education-num` | Education encoded as a number |
| `marital-status` | Marital status |
| `occupation` | Job occupation |
| `relationship` | Relationship status |
| `race` | Race |
| `sex` | Gender |
| `capital-gain` | Capital gains recorded |
| `capital-loss` | Capital losses recorded |
| `hours-per-week` | Average working hours per week |
| `native-country` | Country of origin |
| `salary` | **Target** — `>50K` or `<=50K` |
 
---

## 🤖 Models Implemented
 
| # | Model | Description |
|---|---|---|
| 1 | **XGBoost** | Gradient boosted trees using XGBoost library; handles imbalanced data and missing values efficiently |
| 2 | **CatBoost** | Gradient boosting with native categorical feature support; minimal preprocessing required |
| 3 | **Random Forest** | Ensemble of decision trees using bagging; robust to overfitting |
| 4 | **SVM** | Support Vector Machine with kernel trick; effective for high-dimensional spaces |
| 5 | **Logistic Regression** | Baseline probabilistic linear classifier; interpretable and fast |
| 6 | **Decision Tree** | Tree-based model; highly interpretable with clear decision boundaries |
| 7 | **KNN** | K-Nearest Neighbors; instance-based, non-parametric classifier |
 
---

## ⚙️ Installation
 
### Prerequisites
 
- Python 3.8+
- Jupyter Notebook or JupyterLab
### Setup
 
```bash
# 1. Clone the repository
git clone https://github.com/oMaR-M-M/Employee-Salary-Prediction-FCIS.git
cd Employee-Salary-Prediction-FCIS
 
# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
### Dependencies
 
```
pandas
numpy
scikit-learn
xgboost
catboost
matplotlib
seaborn
jupyter
```
 
---

## 👥 Contributors
 
| Contributor | GitHub | Contribution |
|---|---|---|
| **Omar Mohamed** | [@oMaR-M-M](https://github.com/oMaR-M-M) | Data Preprocessing · XGBoost Model · CatBoost Model |
| **Fares Zahran** | [@StrivingForGood](https://github.com/StrivingForGood) | Random Forest Model |
| **Omar Karam** | [@8-Omoshikiii-8](https://github.com/8-Omoshikiii-8) | SVM Model |
| **Mahmoud Elshahat** | [@ii7oDaZ](https://github.com/ii7oDaZ) | Logistic Regression Model |
| **Omar shokry** | [@Omar-Mohamed-2006](https://github.com/Omar-Mohamed-2006) | Decision Tree Model |
| **Omar Hassaan** | [@omarhassaan22](https://github.com/omarhassaan22) | KNN Model |
 
---
