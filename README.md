# Credit Risk Decision System

---

## Executive Summary
* **Goal:** Build a machine learning system to approve or reject credit card applications.
* **Inputs:** Uses applicant demographics and financial history.
* **Outcome:** Prioritizes financial safety, successfully catching 100% of high-risk applicants in testing.

---

## Problem Being Solved
* Financial institutions lose significant revenue when customers default.
* The specific challenge is identifying applicants likely to be overdue by more than 60 days *before* issuing a card.

---

## Dataset
* **Source:** "Credit Card Approval Prediction" dataset from Kaggle.
* **Content:** Merges demographic info (income, family status, employment) with historical credit records.
* **Purpose:** Creates a complete profile of customer behavior for analysis.

---

## Goal of the Project
* Build a binary classification model to predict "bad customers."
* **Business Priority:** Minimize risk.
* **Focus:** Prioritize catching defaults rather than maximizing the volume of approved cards.

---

## Approach
* Cleaned the data and engineered a target variable.
* **Definition:** Defined a "bad customer" as anyone with overdue payments of 90 days or more.
* **Preprocessing:** Handled missing occupation data and encoded categorical variables.

---

## Models Used
* **Model:** Logistic Regression.
* **Reasoning:** Selected for interpretability and speed.
* **Configuration:** Applied "balanced" class weights because actual defaulters are rare compared to good customers.

---

## Key Results
* **Recall:** 1.00 (100%) for bad customers.
    * *Meaning:* The model correctly identified every single high-risk applicant in the test set.
* **Precision:** 0.09.
    * *Meaning:* The model is very conservative and flags a significant number of safe customers as risky to ensure no bad debt slips through.

---

## How This Would Be Used in Practice
This model serves as a **pre-screening filter**:
* **Auto-Approve:** Applicants predicted as "safe" are approved immediately.
* **Manual Review:** Applicants flagged as "risky" are sent to a human underwriter.
* **Benefit:** Prevents the bank from auto-approving risky customers, while allowing borderline cases to be reviewed by actual people.

---

## Limitations and Next Steps
* **Limitation:** The current model generates high false alarms (False Positives), increasing manual workload.
* **Next Step:** Test tree-based models (like Random Forest) to improve precision without sacrificing the perfect recall rate.

---

## Repository Structure
* `credit_risk_decision_system.ipynb`: Contains data cleaning, EDA, and modeling code.
* `README.md`: Project summary and business context.
