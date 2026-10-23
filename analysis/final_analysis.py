import pandas as pd
import numpy as np

# 1. Load your master dataset
df = pd.read_csv("spectral_final_ai_ready.csv", index_col=0)

# 2. Extract Wavelengths and Target PPM
# We skip 'Target_PPM' to only look at spectral data
target = df['Target_PPM']
spectra = df.drop(columns=['Target_PPM'])
wavelengths = spectra.columns.astype(float)

peak_results = []

print(f"{'Sample ID':<10} | {'PPM':<6} | {'Peak Wavelength':<15} | {'Peak Intensity':<15} | {'Status'}")
print("-" * 70)

for index, row in spectra.iterrows():
    # Find the maximum intensity and its corresponding wavelength
    # NOTE: If looking for Absorbance peaks, use idxmax(). 
    # If looking for Transmission peaks, you might want idxmin().
    max_val = row.max()
    peak_wave = row.idxmax()
    
    # Check for potential Saturation
    # If intensity is extremely high (e.g., 65535 for 16-bit) it might be clipped
    # If we are looking at UV Nitrate, the "peak" we care about is actually the 
    # point of highest absorption (lowest light).
    status = "OK"
    if max_val < 10: # Threshold depends on your sensor's scale
        status = "POTENTIAL SATURATION (TOO DARK)"
    
    peak_results.append({
        "Sample": index,
        "PPM": target[index],
        "Peak_WL": float(peak_wave),
        "Peak_Int": max_val,
        "Status": status
    })
    
    print(f"{index:<10} | {target[index]:<6} | {float(peak_wave):<15.2f} | {max_val:<15.2f} | {status}")

# 3. Save these results to a summary CSV
summary_df = pd.DataFrame(peak_results)
summary_df.to_csv("spectral_peak_summary.csv", index=False)
print(f"\nSummary saved to: spectral_peak_summary.csv")