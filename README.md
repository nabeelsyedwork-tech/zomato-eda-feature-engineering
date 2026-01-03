# 🍽️ Zomato Restaurant Data Analysis & Feature Engineering

A comprehensive **exploratory data analysis (EDA) and feature engineering pipeline** built on the Zomato restaurant dataset, designed to transform raw, messy data into **model-ready features**.

---

## Overview

This project focuses on preparing a real-world restaurant dataset for downstream **machine learning tasks** such as rating prediction, recommendation systems, or price classification.

Instead of stopping at visualization, the project emphasizes:
- Robust data cleaning
- Feature creation driven by domain intuition
- Handling categorical explosion and sparsity
- Producing a **clean, encoded, and scaled dataset**

---

## Key Objectives

- Understand restaurant ratings, pricing, and customer behavior
- Handle missing values and inconsistent formats
- Engineer meaningful numerical and categorical features
- Prepare the dataset for supervised learning models

---

## Dataset

- **Source:** Zomato restaurant dataset
- **Key features:**
  - Ratings
  - Location
  - Cuisine types
  - Cost for two
  - Online ordering & table booking
  - Votes and dish popularity

---

## Data Cleaning

- Removed duplicate records
- Converted rating strings (`"NEW"`, `"-"`) into numerical values
- Filled missing values using:
  - Mean (ratings)
  - Mode (location)
  - Domain defaults (`"Unknown"` for cuisines)
- Cleaned and converted cost values into numerical format

---

## Feature Engineering

### 🔹 Binary Features
- `Has_Online` — online ordering availability
- `Has_Table` — table booking availability
- `Is_Expensive` — based on median cost

### 🔹 Aggregated Features
- Average rating per **location**
- Average rating per **cuisine**

### 🔹 Text-Based Features
- Cuisine count per restaurant
- Grouped rare restaurant types and locations
- Extracted and grouped **top 10 most liked dishes**

---

## Outlier Handling

Applied **IQR-based clipping** to:
- Ratings
- Cost for two
- Number of votes

This prevents extreme values from skewing model behavior.

---

## Encoding & Scaling

- Standardized numerical features using **StandardScaler**
- One-hot encoded:
  - Locations
  - Restaurant types
  - Popular dishes
- Converted boolean fields to integers
- Removed non-predictive columns (URLs, addresses, reviews, menus)

---

## Final Output

The final dataset is:
- Fully numerical
- Scaled
- Encoded
- Free of duplicates and major outliers

✅ Ready for machine learning models such as:
- Regression
- Classification
- Recommendation systems

---

## Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## Use Cases

- Restaurant rating prediction
- Cost classification (budget vs premium)
- Cuisine popularity analysis
- Recommendation systems

---

## Project Status

Complete — designed as a **feature engineering baseline** for ML projects using restaurant or retail datasets.
