# Salary Prediction ML Model

## Problem Statement

Predict employee salaries based on job-related features such as job title, location, company size, required skills, and experience level. This machine learning model helps:
- **Job seekers** understand expected salary ranges for positions
- **HR departments** benchmark competitive salary offers
- **Companies** make data-driven compensation decisions

## Dataset

The dataset (`salary_data.csv`) contains job postings with features including:
- **Job characteristics**: Title, seniority level, description length
- **Company details**: Size, sector, type of ownership, revenue
- **Location**: Job state
- **Skills**: Python, Spark, AWS, Excel, etc.
- **Compensation**: Minimum, maximum, and average salary (target variable)

**Dataset size**: ~700 job postings

---

## Project Structure

```
midterm_project/
├── train.ipynb              # Model training notebook
├── prediction.ipynb         # Prediction testing notebook
├── salary_data.csv          # Dataset
├── salary_predictions.csv   # Prediction results
├── salary_model.pkl         # Trained model (pickled)
├── requirements.txt         # Python dependencies
├── server/                  # Docker deployment
│   ├── Dockerfile          # Container configuration
│   ├── prediction.py       # Flask prediction server
│   ├── raise_request.py    # Sample request script
│   ├── requirements.txt    # Server dependencies
│   ├── sample_customer.json # Sample request payload
│   └── README_DOCKER.md    # Docker-specific docs
└── README.md               # This file
```

---

## Model Training Process

### 1. Data Cleaning & Preprocessing

**Normalization:**
- Column names: lowercase with underscores
- String values: lowercase with underscores
- Missing values: filled with 'missing' (categorical) or 0 (numerical)

**Code snippet from `train.ipynb`:**
```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
for c in categorical_features:
    df[c] = df[c].str.lower().str.replace(' ', '_')
```

### 2. Feature Engineering

**Numerical Features:**
- Analyzed correlation with target variable (avg_salary)
- Applied correlation threshold: `|correlation| > 0.05`
- Selected strong features: Features with meaningful correlation to salary

**Categorical Features:**
- Analyzed unique values, missing data percentage, and correlation
- Selection criteria:
  - Unique values < 50 (manageable one-hot encoding)
  - Missing data < 30%
  - |Correlation| > 0.02

**Feature Encoding:**
- Used `DictVectorizer` for combined numerical + categorical encoding
- Automatically handles one-hot encoding for categorical features
- Preserves numerical features in their original form

### 3. Correlation Analysis

Visualized feature correlations with salary to identify:
- **Strong positive correlators**: Features that increase with salary
- **Strong negative correlators**: Features inversely related to salary
- **Weak correlators**: Removed to reduce noise

**Result:** Selected features with meaningful predictive power while reducing dimensionality.

### 4. Model Selection

**Models Evaluated:**
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Random Forest Regressor ⭐
6. Gradient Boosting Regressor
7. AdaBoost Regressor
8. Support Vector Regression (SVR)
9. K-Nearest Neighbors (KNN)

**Evaluation Metrics:**
- **R² Score** (Test): Model's ability to explain variance
- **RMSE** (Root Mean Squared Error): Average prediction error
- **Cross-Validation R²**: Consistency across data splits
- **Overfitting Gap**: Train R² - Test R²

**Best Model: Random Forest Regressor**
- Test R²: ~0.45-0.50 (explains 45-50% of salary variance)
- Test RMSE: ~$33-35k
- Low overfitting (gap < 0.10)
- Handles non-linear relationships well
- Robust to outliers

**Why Random Forest?**
- Captures complex interactions (e.g., experience + skills)
- Works well with mixed data types
- Naturally handles feature importance
- Resistant to salary outliers

### 5. Model Persistence

Saved model package includes:
```python
{
    'model': best_model_final,           # Trained Random Forest
    'dict_vectorizer': vectorizer        # Feature encoder
}
```

**File:** `salary_model.pkl` (pickled with Python's `pickle` module)

---

## Running the Project

### Prerequisites

```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pickle

### 1. Train the Model

Open and run `train.ipynb`:

```bash
jupyter notebook train.ipynb
```

**Key cells:**
1. Load and explore data
2. Data cleaning and preprocessing
3. Correlation analysis and feature selection
4. Train multiple models and compare
5. Select best model and save

**Output:** `salary_model.pkl`

### 2. Make Predictions

Open and run `prediction.ipynb`:

```bash
jupyter notebook prediction.ipynb
```

**What it does:**
1. Loads trained model from `salary_model.pkl`
2. Preprocesses new data (same pipeline as training)
3. Makes predictions on full dataset
4. Saves results to `salary_predictions.csv`
5. Tests single-sample predictions

**For custom predictions:**
```python
customer_data = {
    'python_yn': 1,
    'spark': 0,
    'aws': 1,
    'num_comp': 3,
    'desc_len': 3747,
    'employer_provided': 0,
    'excel': 1,
    'hourly': 0,
    'seniority': 'jr',
    'job_state': 'tx',
    'type_of_ownership': 'company_-_public',
    'sector': 'real_estate',
    'job_simp': 'data_scientist',
    'revenue': '$1_to_$2_billion_(usd)',
    'size': '201_to_500_employees'
}

customer_x = dv.transform([customer_data])
prediction = model.predict(customer_x)
```

---

## Docker Deployment

### Build Docker Image

Navigate to the `server/` directory:

```bash
cd server
docker build -t salary-prediction-server .
```

**What the Dockerfile does:**
1. Uses Python 3.12 slim base image
2. Installs dependencies from `requirements.txt`
3. Copies model file (`salary_model.pkl`)
4. Copies Flask server (`prediction.py`)
5. Exposes port 9696
6. Runs Flask server on startup

### Run Docker Container

```bash
docker run -p 9696:9696 salary-prediction-server
```

**Options:**
- `-p 9696:9696`: Maps container port to host port
- `-d`: Run in detached mode (background)
- `--name salary-server`: Give container a name

**Verify it's running:**
```bash
docker ps
```

### Make Prediction Request

**Using the provided script:**
```bash
cd server
python raise_request.py
```

**Manual cURL request:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d @sample_customer.json
```

**Sample request payload (`sample_customer.json`):**
```json
{
  "python_yn": 1,
  "spark": 0,
  "aws": 1,
  "num_comp": 3,
  "desc_len": 3747,
  "employer_provided": 0,
  "excel": 1,
  "hourly": 0,
  "seniority": "jr",
  "job_state": "tx",
  "type_of_ownership": "company_-_public",
  "sector": "real_estate",
  "job_simp": "data_scientist",
  "revenue": "$1_to_$2_billion_(usd)",
  "size": "201_to_500_employees"
}
```

**Expected response:**
```json
{
  "predicted_salary": 85.23,
  "salary_range": "$80k - $90k"
}
```

### Stop Container

```bash
docker stop salary-server
docker rm salary-server
```

---

## API Endpoints

### POST `/predict`

**Description:** Predict salary for a job profile

**Request:**
- **Method:** POST
- **Content-Type:** application/json
- **Body:** JSON object with job features

**Response:**
```json
{
  "predicted_salary": float,  // Predicted salary in thousands (k)
  "salary_range": string      // Human-readable range
}
```

**Example using Python:**
```python
import requests

data = {
    "python_yn": 1,
    "aws": 1,
    "seniority": "jr",
    "job_state": "ca",
    # ... other features
}

response = requests.post(
    "http://localhost:9696/predict",
    json=data
)

print(response.json())
```

---

## Model Performance

**Best Model:** Random Forest Regressor

| Metric | Value |
|--------|-------|
| Test R² | 0.45-0.50 |
| Test RMSE | $33-35k |
| Cross-Validation R² | 0.43 (±0.05) |
| Overfitting Gap | < 0.10 |

**Interpretation:**
- Model explains ~45-50% of salary variance
- Average prediction error: ±$33k
- Good generalization (low overfitting)
- Acceptable for salary range estimation

---

## Key Insights

1. **Strongest predictors of salary:**
   - Seniority level
   - Company size and revenue
   - Job title/type
   - Technical skills (Python, AWS)

2. **Model limitations:**
   - R² ~0.45 means ~55% of variance unexplained
   - Likely due to missing features (negotiation, industry trends, individual performance)
   - Suitable for salary range estimation, not exact predictions

3. **Feature engineering impact:**
   - Correlation-based filtering improved model performance
   - DictVectorizer simplified preprocessing pipeline
   - Handling missing values crucial for robustness

---

## Future Improvements

- [ ] Collect more data (current: ~700 samples)
- [ ] Add features: education, certifications, years of experience
- [ ] Implement feature engineering: interaction terms
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Ensemble multiple models
- [ ] Add confidence intervals to predictions
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Create web UI for predictions

---

## Repository

**GitHub:** https://github.com/Ganeshkumbara/machine-learning-zoomcamp-projects

**Project:** `midterm_project/`

---

## Author

**Course:** Machine Learning Zoomcamp  
**Project:** Midterm Project - Salary Prediction Model

---

## License

This project is for educational purposes as part of the ML Zoomcamp course.
