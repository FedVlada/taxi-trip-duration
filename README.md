# NYC Taxi Trip Duration Prediction

A machine learning project to predict taxi trip duration in New York City using various regression models.

## Project Overview

This project addresses a real-world machine learning problem: **predicting the total duration of taxi rides in NYC**. The ability to accurately forecast trip duration enables taxi services to:
- Estimate trip costs in real-time
- Optimize pricing strategies
- Improve customer experience with accurate time estimates

## Business & Technical Context

**Business Goal:** Automate the process of predicting taxi trip duration based on various characteristics like pickup/dropoff location, time of day, weather conditions, etc.

**Technical Goal:** Build a regression model that predicts trip duration in seconds using machine learning algorithms.

## Key Project Objectives

1. ✅ Aggregate and prepare data from multiple sources
2. ✅ Conduct exploratory data analysis (EDA) and feature engineering
3. ✅ Identify patterns and relationships in the data
4. ✅ Build and compare multiple regression models
5. ✅ Select the best-performing model based on RMSLE metric
6. ✅ Deploy predictions on test data

## Technology Stack

- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: scikit-learn, XGBoost
- **Geospatial**: Haversine distance calculation
- **Jupyter**: Interactive notebooks for development

---
## Metric: RMSLE

We use **Root Mean Squared Logarithmic Error (RMSLE)** as the primary metric:

$$RMSLE = \sqrt{\frac{1}{n}\sum_{i=1}^n(\log(y_i+1)-\log(\hat{y_i}+1))^2}$$

Where:
- $y_i$ = actual trip duration
- $\hat{y_i}$ = predicted trip duration

This metric is scale-invariant and penalizes underestimates more than overestimates, which is appropriate for this business case.

## Model Performance

| Model | Train RMSLE | Valid RMSLE |
|-------|------------|------------|
| Linear Regression | 0.536 | 0.539 |
| Decision Tree (optimal depth) | 0.406 | 0.43 |
| Random Forest | 0.40 | 0.414 |
| Gradient Boosting | **0.371** | **0.393** |
| XGBoost | 0.384 | 0.394 |

**Best Model:** Gradient Boosting Regressor
- Achieved the lowest RMSLE on the validation set
- Good balance between training and validation performance (minimal overfitting)


## Project Structure

```
├── nyc_taxi_trip_duration.ipynb    # Jupyter notebook with analysis
├── README.md                       # Project documentation
├── LICENSE                         # MIT license 
├── gitignore
└── requirements.txt
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/FedVlada/taxi-trip-duration.git
cd taxi-trip-duration
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the data**
Place the data files in the `data/` directory:
- `train.csv`              https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data
- `holiday_data.csv`       https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/#url=Historical-Data
- `weather_data.csv`       (request the data from me as others)
- `osrm_data_train.csv`    https://github.com/Project-OSRM
- `test_data.csv`          (for test predictions, the same link as "train")
- `osrm_data_test.csv`     (for test predictions, the same link as "osmr_data_train")

4. **Run the notebook**
```bash
jupyter notebook nyc_taxi_trip_duration.ipynb
```

## Key Findings from EDA

1. **Temporal Patterns:**
   - Taxi demand peaks during morning (8-10 AM) and evening (5-7 PM) rush hours
   - Weekend trips tend to be slightly longer
   - Minimal demand between 2-5 AM

2. **Geographic Insights:**
   - Geographic clustering reveals 10 distinct zones in NYC
   - Trip duration correlates with distance traveled

3. **Feature Importance (from Gradient Boosting):**
   - Top 3 features: total_distance, total_travel_time, haversine_distance
   - Weather conditions have moderate impact on trip duration
   - Holiday and special event flags provide additional predictive power

4. **Data Quality:**
   - ~1.5M trips in training set
   - Removed ~1000 outliers (trips >24h or avg speed >300 km/h)
   - No missing values in core features after data cleaning

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Author

[FedVlada](https://github.com/FedVlada)

For questions or suggestions, please open an issue on GitHub.