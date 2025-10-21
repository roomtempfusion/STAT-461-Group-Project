# Variant-Specific Death Prediction with Spatial ARIMA Models

## ✓ FILE SAVED SUCCESSFULLY

### Problem Identified and Fixed:
The adjacency matrices loaded from CSV had **mismatched data types**:
- **Index**: int64 (17001, 17003, ...)
- **Columns**: string ('17001', '17003', ...)

This caused matrix multiplication failures when calculating spatial lags.

### Solution Applied:
In cell #2, added explicit type conversion:
```python
border_adj.columns = border_adj.columns.astype(int)
airport_adj.columns = airport_adj.columns.astype(int)
highway_adj.columns = highway_adj.columns.astype(int)
```

Now both index and columns are int64, allowing proper matrix operations.

---

## Notebook Overview

**File**: `variant_deaths_arima_spatial.ipynb`

### What It Does:

1. **Variant-Specific Deaths**: Calculates deaths attributed to each variant using:
   ```
   Variant_Deaths = Daily_New_Deaths × Variant_Prevalence
   ```

2. **Time Windows**: Uses different periods for each variant based on when they were actively spreading:
   - **Alpha**: Jan 17, 2021 → Aug 18, 2021 (214 days)
   - **Delta**: Jan 10, 2021 → Feb 18, 2022 (404 days)
   - **Epsilon**: Oct 27, 2020 → May 25, 2021 (210 days)
   - **Iota**: Feb 10, 2021 → Jul 10, 2021 (150 days)

3. **Spatial Features**: Creates spatial lag variables using three adjacency types:
   - **Border adjacency** - Counties sharing physical borders
   - **Airport adjacency** - Counties with airports
   - **Highway adjacency** - Counties connected by interstates

4. **Two Model Types**:
   - **ARIMA**: Pure time series models
   - **ARIMAX**: ARIMA + spatial features (neighbor county effects)

5. **Evaluation**: Compares models using RMSE, MAE, and R² metrics

---

## How to Run

1. **Open Jupyter**:
   ```bash
   jupyter notebook sree_models/variant_deaths_arima_spatial.ipynb
   ```

2. **Run all cells** sequentially (Cell → Run All)

3. **Expected outputs**:
   - Model performance comparisons
   - Prediction visualizations
   - Residual diagnostics
   - CSV files with predictions

---

## Key Features

### ✓ Spatial Connectivity
Accounts for disease spread between neighboring counties using weighted adjacency matrices.

### ✓ Variant-Specific Modeling
Separate models for each variant during their active periods.

### ✓ ARIMA vs ARIMAX Comparison
Shows whether spatial features improve predictions.

### ✓ Comprehensive Diagnostics
- Stationarity tests (ADF)
- ACF/PACF plots
- Residual analysis
- Performance metrics

---

## Output Files

When you run the notebook, it creates:

1. **arima_arimax_comparison.csv** - Performance comparison table
2. **Alpha_predictions.csv** - Alpha variant forecasts
3. **Delta_predictions.csv** - Delta variant forecasts
4. **Epsilon_predictions.csv** - Epsilon variant forecasts
5. **Iota_predictions.csv** - Iota variant forecasts

Each prediction file contains:
- Date
- Actual deaths
- ARIMA predictions
- ARIMAX predictions (if available)

---

## Requirements

Required Python packages:
```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

---

## Troubleshooting

### If you get "matrices are not aligned" error:
Make sure you're running the **corrected version** of the notebook with the adjacency matrix column type conversion in cell #2.

### If you get "insufficient data" warnings:
This is normal for variants with short active periods. The notebook will skip those and continue with others.

### If plots don't show:
Make sure you have `%matplotlib inline` in the first cell and are running in Jupyter (not plain Python).

---

## Contact

For questions or issues, check the main project documentation or review the data processing notebook at `data processing/data_processor.ipynb`.
