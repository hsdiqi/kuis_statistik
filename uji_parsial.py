import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Membaca dataset
df = pd.read_csv('cars_raw.csv')

# Analisis dengan 10 variabel numerik menggunakan Python
X_full = df[['ConsumerReviews', 'SellerRating', 'SellerReviews', 'ComfortRating', 'InteriorDesignRating', 'PerformanceRating', 'ValueForMoneyRating', 'ExteriorStylingRating', 'ReliabilityRating']]
y_full = df['ConsumerRating']

# Menambahkan konstanta untuk termasuk intercept
X_full = sm.add_constant(X_full)

# Membuat model regresi
model_full = sm.OLS(y_full, X_full).fit()

# Menampilkan hasil regresi
print("Hasil Regresi untuk 10 Variabel Numerik:")
print(model_full.summary())

# Uji parsial regresi (t-test) untuk masing-masing koefisien regresi
print("\nUji Parsial Regresi:")
for i, coef_name in enumerate(X_full.columns):
    coef_value = model_full.params.iloc[i]
    coef_std_error = model_full.bse.iloc[i]
    t_statistic = model_full.tvalues.iloc[i]
    p_value = model_full.pvalues.iloc[i]

    print(f"\nVariabel: {coef_name}")
    print(f"Koefisien: {coef_value:.4f}")
    print(f"Standard Error: {coef_std_error:.4f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")

    # Uji hipotesis
    if p_value < 0.05:
        print("Koefisien signifikan secara statistik pada tingkat signifikansi 0.05")
    else:
        print("Koefisien tidak signifikan secara statistik pada tingkat signifikansi 0.05")