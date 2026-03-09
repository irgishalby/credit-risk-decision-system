# Credit Risk Decision System

---

## Executive Summary

* **Goal:** Develop an **automated machine learning system** to evaluate and decide on **credit card applications**.
* **Inputs:** Integrated **demographic profiles** (income, employment, family status) and **historical financial behavior**.
* **Outcome:** Successfully identified **100% of high-risk applicants** in testing, prioritizing **financial security**.

## Problem Being Solved

* **Financial Risk:** Financial institutions experience significant **revenue loss** when customers **default** on their credit obligations.
* **Prediction Challenge:** Identifying applicants likely to be **overdue by more than 60 days** before a line of credit is issued.

## Dataset

* **Source:** "**Credit Card Approval Prediction**" dataset retrieved from **Kaggle**.
* **Content:** Merged **demographic information** with **historical credit records** to create a complete profile of customer behavior.
* **Purpose:** Facilitate **comprehensive analysis** of applicant risk factors.

## Goal of the Project

* **Classification:** Build a **binary classification model** to accurately **predict "bad customers"**.
* **Business Priority:** **Minimize institutional risk** rather than maximizing application volume.
* **Focus:** Prioritize the **detection of potential defaults** to ensure **long-term profitability**.

## Approach

* **Data Cleaning:** Handled **missing occupation data** and performed **categorical encoding** for model readiness.
* **Target Engineering:** Defined a "**bad customer**" specifically as anyone with **overdue payments of 90 days or more**.
* **Validation Strategy:** Conducted **time-based validation** by training on the oldest 70% of records and testing on the newest 30% to ensure **production stability**.

## Models Used

* **Logistic Regression:** Selected for high **interpretability** and **computational efficiency**.
* **Random Forest Classifier:** Utilized to capture **non-linear patterns** and enhance **prediction accuracy**.
* **Standardization:** Applied **feature scaling** to ensure uniform weight distribution across demographic and financial variables.

## Key Results

* **Financial Optimization:** Implementation of a **cost-aware threshold (0.04)** minimized total **business loss to zero** in test scenarios.
* **Economic Impact:** Realized **IDR 19.5 million** in **reclaimed profit** compared to standard classification techniques.
* **Predictive Power:** Maintained a **1.00 F1-score** during **time-based validation**, confirming high reliability for future accounts.
* **Interpretability:** Leveraged **SHAP (SHapley Additive exPlanations)** values to identify **key risk drivers** in applicant profiles.

## How This Would Be Used in Practice

* **Automated Screening:** Integration into the **application pipeline** for **instantaneous approval or rejection** based on risk scores.
* **Dynamic Risk Management:** Ability to adjust **decision thresholds** in **real-time** according to changing financial constants and **risk appetite**.

## Limitations and Next Steps

* **Deployment:** Transition the validated model into a **production environment** for **live application processing**.
* **Monitoring:** Establish **continuous monitoring protocols** to detect **temporal drift** in applicant behavior.
