import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Membaca dataset
df = pd.read_csv('cars_raw.csv')

# 1. Analisis dengan 5 variabel numerik menggunakan Python
X_sample = df[['ConsumerReviews', 'SellerRating', 'SellerReviews', 'ComfortRating', 'InteriorDesignRating', 'PerformanceRating', 'ValueForMoneyRating', 'ExteriorStylingRating', 'ReliabilityRating']]
y_sample = df['ConsumerRating']

# Menambahkan konstanta untuk termasuk intercept
X_sample = sm.add_constant(X_sample)

# Membuat model regresi
model_sample = sm.OLS(y_sample, X_sample).fit()

# Menampilkan hasil regresi
print("Hasil Regresi untuk 5 Variabel Numerik:")
print(model_sample.summary())

# 2. Analisis dengan 10 variabel numerik menggunakan Python
X_full = df[['ConsumerReviews', 'SellerRating', 'SellerReviews', 'ComfortRating', 'InteriorDesignRating', 'PerformanceRating', 'ValueForMoneyRating', 'ExteriorStylingRating', 'ReliabilityRating']]
y_full = df['ConsumerRating']

# Menambahkan konstanta untuk termasuk intercept
X_full = sm.add_constant(X_full)

# Membuat model regresi
model_full = sm.OLS(y_full, X_full).fit()

# Menampilkan hasil regresi
print("\nHasil Regresi untuk 10 Variabel Numerik:")
print(model_full.summary())

# 3. Uji Parsial (t-test) untuk semua variabel pada data 100 sample
t_test = model_full.t_test(np.eye(len(model_full.params)))
t_stat = t_test.tvalue
p_values = t_test.pvalue

print("\nUji Parsial (t-test) untuk semua variabel pada data 100 sample:")
for i, var in enumerate(model_full.params.index):
    print(f"{var}: T-Stat: {t_stat[i]}, P-Value: {p_values[i]}")
    if p_values[i] < 0.05:
        print(f"Variabel '{var}' signifikan secara parsial.")
    else:
        print(f"Variabel '{var}' tidak signifikan secara parsial.")


# Menentukan hipotesis nol
hypothesis = [
    "ConsumerReviews = 0",
    "SellerRating = 0",
    "SellerReviews = 0",
    "ComfortRating = 0",
    "InteriorDesignRating = 0",
    "PerformanceRating = 0",
    "ValueForMoneyRating = 0",
    "ExteriorStylingRating = 0",
    "ReliabilityRating = 0",
    "const = 0"
]

# Melakukan uji F
f_test = model_full.f_test(hypothesis)

# Menampilkan hasil uji F
print(f"\nHasil Uji F untuk Hipotesis Nol:")
print(f"F-Statistic: {f_test.fvalue:.4f}")  # Removed subscripting
print(f"P-Value: {f_test.pvalue:.4f}")


# 5. Uji Kebaikan Model menggunakan R-squared
r_squared = model_full.rsquared
print(f"\nR-Squared (Koefisien Determinasi) untuk model dengan 10 variabel numerik:")
print(f"R-Squared: {r_squared}")
