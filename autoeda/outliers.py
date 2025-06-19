import pandas as pd
import numpy as np
from scipy.stats import zscore
import json
import os

# 1. Load and inspect the scaled dataset
input_path = "notebooks/output-files/autoEDA_scaled_output.csv"
df = pd.read_csv(input_path)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

detection_methods = {}
outliers_detected = {}
rows_with_outliers = set()
outlier_flags = pd.DataFrame(index=df.index)

# 2. Detect Skewness and apply outlier detection
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
    skew_val = df[col].skew()

    if abs(skew_val) < 1.0:
        method = "Z-score"
        if df[col].std() == 0:
            z_scores = np.zeros_like(df[col])
        else:
            z_scores = zscore(df[col])
        outliers = np.abs(z_scores) > 3
        lower_bound = df[col].mean() - 3 * df[col].std()
        upper_bound = df[col].mean() + 3 * df[col].std()
    else:
        method = "IQR"
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

    detection_methods[col] = method
    outliers_detected[col] = int(outliers.sum())
    rows_with_outliers.update(df[outliers].index)
    outlier_flags[f"{col}_is_outlier"] = outliers.astype(int)

# 5. Save flagged dataset
flagged_df = pd.concat([df, outlier_flags], axis=1)
flagged_df.to_csv("autoEDA_outliers_flagged.csv", index=False)

# 6. Outlier Capping (Winsorization)
capped_df = df.copy()
for col in numeric_cols:
    method = detection_methods[col]
    if method == "Z-score":
        lower_bound = df[col].mean() - 3 * df[col].std()
        upper_bound = df[col].mean() + 3 * df[col].std()
    else:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

    capped_df[col] = np.clip(df[col], lower_bound, upper_bound)

capped_df.to_csv("autoEDA_outliers_capped.csv", index=False)

# 7. Outlier Removal
removed_df = df.drop(index=list(rows_with_outliers))
removed_df.to_csv("autoEDA_outliers_removed.csv", index=False)

# 8. Generate JSON report
report = {
    "detection_methods": detection_methods,
    "outliers_detected": outliers_detected,
    "rows_removed": len(rows_with_outliers),
    "output_files": {
        "flagged": "autoEDA_outliers_flagged.csv",
        "capped": "autoEDA_outliers_capped.csv",
        "removed": "autoEDA_outliers_removed.csv"
    }
}

with open("autoEDA_outlier_report.json", "w") as f:
    json.dump(report, f, indent=2)

# 9. Save tabular summary report
summary_df = pd.DataFrame({
    "Column": list(detection_methods.keys()),
    "Detection_Method": list(detection_methods.values()),
    "Outliers_Detected": list(outliers_detected.values())
})
summary_df.to_csv("autoEDA_outlier_summary.csv", index=False)

# 10. Final message
print("✅ Outlier detection, flagging, capping, and removal complete.")
print("📄 Reports saved:")
print("- autoEDA_outlier_report.json")
print("- autoEDA_outlier_summary.csv")
