# Big W Sales Forecasting & Market Penetration Analysis

## 📌 Project Overview
**Context:** Woolworths Group (Big W) vs. Competitors (Kmart, Target, Amazon)

This project analyzes sales drivers for **Big W**, a major Australian discount department store operating nearly 180 stores nationwide. While Big W leverages the massive **Everyday Rewards (EDR)** loyalty program (14 million members), it faces intensifying competition from traditional rivals like Kmart and Target, as well as the rapid growth of e-commerce players like Amazon.

The primary objective is to understand market penetration drivers and develop a robust predictive model for sales forecasting. This enables Big W to optimize:
* **Store Locations:** Identifying the ideal distance from competitors to maximize foot traffic without cannibalization.
* **Media Investment:** Allocating the ~$50M annual advertising budget effectively across TV, Online, and Radio channels.
* **Inventory Strategy:** Tailoring stock levels for high-value customer segments in specific regions.

## 📂 Dataset Description
The analysis utilizes a comprehensive set of internal and external data sources (~1.8 million rows in total):

* **Sales by Customer Location:** Transactional data linking customer "price-lifestage" segments (e.g., *Budget-Retirees*) to specific store performance.
* **Store Metadata:** Geospatial data for 425+ stores, including a critical derived metric: **Distance to nearest Kmart/Target**.
* **Media Investment:** Weekly advertising spend data across various channels.
* **External Demographics:** ABS 2021 Census data (Median Personal Income by State) to proxy purchasing power.

## 🛠️ Methodology & Detailed Analysis

### 1. Data Preprocessing & Imputation
* **Missing Value Strategy**:
    * *Geographical Data*: Missing state/postcode information was imputed using the **mode** of neighboring stores to preserve spatial coherence.
    * *Customer Segments*: 16,210 missing entries in `price_lifestage_segment` were imputed by grouping stores with similar sales distributions (e.g., mapping unknown segments in low-income areas to the 'Budget' category).
* **Outlier Treatment**: A **Log Transformation** was applied to the target variable (`total_sale_value`) to address the heavy right-skew typical of retail sales data, ensuring better model convergence.

### 2. Feature Engineering
We enriched the dataset by extracting granular features from composite variables:
* **Segment Decomposition**: The `price_lifestage_segment` string was parsed into three distinct features: `Family_Status` (e.g., Family, Couple), `Economy_Status` (e.g., Budget, Premium), and `Age_Group`.
* **Spatial Interaction**: Created a `Competitor_Proximity_Flag` to distinguish stores that are co-located with Kmart/Target versus those with a strategic buffer distance.
* **Media Spend Normalization**: Applied **Box-Cox transformation** to `Media_Amount_Spend` to stabilize variance and linearize relationships.

## 💡 Key Findings
Our exploratory data analysis (EDA) and model feature importance analysis yielded three critical business insights:

### 1. The "Buffer Zone" Effect
Contrary to the assumption that being closest to competitors is best, stores located **1-3km** from a Kmart or Target consistently outperformed those strictly co-located (0km) or isolated (>5km).
* *Insight*: A 1-3km distance captures "spillover" traffic from competitor hubs while reducing direct price comparison and cannibalization.

### 2. High-Value vs. High-Volume Segments
* **High Value**: "Mainstream Families" and "Older Singles/Couples" demonstrated the largest basket sizes.
* **Underperformance**: "Young Singles" segments showed significantly lower transaction values despite high foot traffic potential.
* *Strategy*: Inventory in suburban stores should prioritize family-oriented bulk goods, while city stores should focus on lower-price, high-turnover items for young singles.

### 3. Regional Disparities
* **Northern Territory (NT)**: Exhibited the highest median transaction value, likely due to lower market saturation and fewer competitor alternatives.
* **Victoria (VIC)**: Showed high transaction volume but lower average value, indicating a highly competitive, price-sensitive market requiring aggressive discounting strategies.

## 🤖 Machine Learning Models
Six models were trained to predict log-sales, ranging from interpretable baselines to complex ensembles:

1.  **Linear Regression (Lasso)**: Used for feature selection and establishing a baseline.
2.  **K-Nearest Neighbors (KNN)**: Captures local spatial similarities between stores.
3.  **Random Forest Regressor**: Handles non-linearities; tuned using **Optuna** for `min_samples_leaf` and `n_estimators`.
4.  **XGBoost**: Gradient boosting machine chosen for its speed and performance on tabular data; tuned for learning rate and tree depth.
5.  **Neural Network (PyTorch)**: A feed-forward deep learning model with:
    * 3 Hidden Layers
    * **Leaky ReLU** activation function
    * **SGD with Momentum** optimizer to escape local minima.
6.  **Stacking Ensemble (Meta-Learner)**: Combines predictions from RF, XGBoost, and KNN using a secondary linear model to correct individual biases.

## 🏆 Model Performance & Results
The **Stacking Ensemble** proved to be the superior model, balancing the bias-variance trade-off better than any single model.

| Model | RMSE (Root Mean Square Error) | R-Squared (R²) |
| :--- | :--- | :--- |
| **Model Stack** | **0.860** | **0.214** |
| XGBoost | 0.868 | 0.200 |
| Neural Network | 0.880 | 0.177 |
| Random Forest | 0.893 | 0.153 |
| Linear Regression | 0.944 | 0.052 |

**Conclusion**: While the RMSE is low (indicating good predictive accuracy for general trends), the relatively low R² suggests that sales are also heavily influenced by uncaptured factors such as store-specific inventory availability, local weather, or manager performance.

## 📢 Strategic Recommendations
Based on the modeling results, we recommend Big W:
1.  **Optimize Store Placement**: Avoid direct co-location with Kmart/Target. Target sites with a **1-3km buffer** to capture spillover traffic without direct price cannibalization.
2.  **Refine Targeting**: Shift marketing focus from generic "Young Singles" to **"Mainstream Families"** and **"Retirees"** in suburban areas, as they show higher basket sizes.
3.  **Digital Integration**: Leverage the EDR program to gather more granular individual-level data, which could improve the R² of future predictive models by capturing personal preferences.

---

### 💻 Code & Analysis Reference
The complete data processing pipeline, model training scripts, and hyperparameter tuning logs are available in the associated Jupyter Notebook.

> **File**: `Big W Written Report Group 28 Coding Part.ipynb`  
> **Repository**: *[Insert Link to Your Repository Here]* > **Authors**: Group 28, USYD QBUS6600 (Semester 2, 2023)  
> **Tech Stack**: Python (Pandas, Scikit-Learn, XGBoost, PyTorch, Optuna)
