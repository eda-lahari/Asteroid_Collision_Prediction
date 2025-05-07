# 🌌 Asteroid Collision and Impact Prediction using Machine Learning

> A smart ML-powered system that predicts the threat level of asteroids and estimates potential damage, combining space science with artificial intelligence for planetary protection.

---

## 🚀 Project Overview

Asteroids can pose severe threats to Earth — from local disasters to planetary-scale extinction events. In this project, we leverage machine learning to assess the danger levels of near-Earth objects (NEOs) and predict their size and potential impact severity.

Our system includes:
- **Binary classification** to determine if an asteroid is hazardous  
- **Regression analysis** to predict the asteroid's diameter  

This predictive framework is designed to aid early warning systems, evacuation planning, and space mission strategy.

---

## 🎯 Objectives

- Predict asteroid size and estimate potential impact severity  
- Classify whether an asteroid is potentially hazardous  
- Support real-time decision-making using ML predictions  
- Contribute towards global disaster preparedness and mitigation

---

## 🧠 Machine Learning Techniques Used

### 1. Hazardous Asteroid Classification

- **Algorithms Used**:  
  - Logistic Regression  
  - Decision Tree Classifier  
  - Support Vector Machine (SVM)  
  - **XGBoost Classifier** → *Best Accuracy: 99.43%*

- **Metrics for Evaluation**:  
  Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

### 2. Asteroid Diameter Prediction (Regression Task)

- **Algorithms Used**:  
  - Ridge Regression → *R² Score: 0.9837*  
  - K-Nearest Neighbors (KNN)  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  

---

## 📊 Dataset Details

We utilized two main datasets:

### 🛰️ NASA NEO Dataset
- Entries: 4,687  
- Features: diameter, estimated mass, speed, trajectory, orbit class, close approach data  

### 🌍 Asteroid Open Dataset (Kaggle)
- Entries: 827,646  
- Features: perihelion distance, orbital parameters, estimated diameter, eccentricity  

**Data Sources**:  
- [NASA Center for Near-Earth Object Studies (CNEOS)](https://neo.jpl.nasa.gov/)  
- [Kaggle - Asteroid Hazard Prediction Dataset](https://www.kaggle.com/datasets/brsdincer/asteroid-classification-for-hazardous-prediction)

---

## 🛠️ Tech Stack & Tools

- **Programming Language**: Python  
- **Libraries**: scikit-learn, NumPy, pandas, seaborn, matplotlib, XGBoost  
- **Environment**: Google Colab, Jupyter Notebooks  

---

## 📈 Model Performance Summary

| Task                        | Model                 | Metric       | Score       |
|-----------------------------|-----------------------|--------------|-------------|
| Hazard Classification       | XGBoost Classifier    | Accuracy     | **99.43%**  |
| Diameter Prediction         | Ridge Regression      | R² Score     | **0.9837**  |

---

## 🔁 Project Workflow

1. **Data Cleaning & Preprocessing**  
   - Missing value handling, label encoding, scaling, outlier removal

2. **Exploratory Data Analysis (EDA)**  
   - Heatmaps, distribution plots, scatter matrices, PCA (if needed)

3. **Model Training & Hyperparameter Tuning**  
   - GridSearchCV, RandomizedSearchCV for best parameter selection

4. **Model Evaluation & Visualization**  
   - Confusion matrix, classification report, regression error plots

5. **(Planned)** Deployment  
   - Real-time web-based alert system (via Flask or Streamlit)

---

## 🌍 Future Enhancements

- Live integration with NASA’s NEO API for real-time predictions  
- Implementation of Deep Learning models (LSTMs for time-series orbital predictions)  
- Creation of a public-facing dashboard for asteroid alerts  
- Estimation of geological and climate impact using geospatial ML  
- Incorporation of data from space missions like NASA DART & ESA Hera

---


## 📚 References

- [Kaggle Dataset on Hazardous Asteroid Prediction](https://www.kaggle.com/datasets/brsdincer/asteroid-classification-for-hazardous-prediction)  
- [IEEE Research Article on Asteroid Prediction](https://ieeexplore.ieee.org/document/10481589)  
- [IOSR Journal on Hazardous Object Detection](https://www.iosrjournals.org/iosr-jce/papers/Vol26-issue6/Ser-1/F2606013744.pdf)  

