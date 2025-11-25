import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ========================================================
# Kreiranje jedinstvenog foldera za grafove
# ========================================================
timestamp = datetime.now().strftime("viz_%Y-%m-%d_%H-%M-%S")
os.makedirs(timestamp, exist_ok=True)

print(f"📁 Generišem grafove u folder: {timestamp}/")

# ========================================================
# Učitavanje CSV
# ========================================================
methods   = []
times     = []
speedups  = []
results   = []

with open("results.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        methods.append(row["Method"])
        times.append(float(row["TimeSeconds"]))
        speedups.append(float(row["SpeedupVsClassic"]))
        results.append(int(row["Result"]))

times     = np.array(times)
speedups  = np.array(speedups)

colors = ["#2E86C1", "#28B463", "#CA6F1E", "#8E44AD", "#C0392B"]


# ========================================================
# 1) Apsolutna vremena
# ========================================================
plt.figure(figsize=(10,5))
plt.title("Vrijeme izvršavanja knapsack metoda", fontsize=14)
plt.bar(methods, times, color=colors[:len(methods)])
plt.ylabel("Vrijeme (s)")
plt.xlabel("Metoda")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{timestamp}/01_times.png", dpi=220)


# ========================================================
# 2) Log skala
# ========================================================
plt.figure(figsize=(10,5))
plt.title("Vrijeme izvršavanja (log skala)", fontsize=14)
plt.bar(methods, times, color=colors[:len(methods)])
plt.yscale("log")
plt.ylabel("Vrijeme (log s)")
plt.xlabel("Metoda")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{timestamp}/02_times_log.png", dpi=220)


# ========================================================
# 3) Ubrzanja vs Classic 2D
# ========================================================
plt.figure(figsize=(10,5))
plt.title("Ubrzanje u odnosu na Classic 2D", fontsize=14)
plt.bar(methods, speedups, color=colors[:len(methods)])
plt.ylabel("Ubrzanje (x)")
plt.xlabel("Metoda")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{timestamp}/03_speedups_vs_classic.png", dpi=220)


# ========================================================
# 4) SIMD vs 1D baseline i fast
# ========================================================
idx_simd = methods.index("SIMD_stream")
idx_base = methods.index("1D_baseline")
idx_fast = methods.index("1D_fast")

simd_speed_vs_base = times[idx_base] / times[idx_simd]
simd_speed_vs_fast = times[idx_fast] / times[idx_simd]

labels = ["SIMD vs 1D_baseline", "SIMD vs 1D_fast"]
values = [simd_speed_vs_base, simd_speed_vs_fast]

plt.figure(figsize=(10,5))
plt.title("Ubrzanje SIMD-a u odnosu na 1D metode", fontsize=14)
plt.bar(labels, values, color=["#1F618D", "#AF7AC5"])
plt.ylabel("Ubrzanje (x)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{timestamp}/04_simd_vs_1d.png", dpi=220)


# ========================================================
# 5) SIMD vs SIMD+OpenMP
# ========================================================
idx_simd_omp = methods.index("SIMD_stream_OpenMP")

labels = ["SIMD stream", "SIMD stream + OpenMP"]
values = [times[idx_simd], times[idx_simd_omp]]

plt.figure(figsize=(10,5))
plt.title("Uticaj OpenMP paralelizacije", fontsize=14)
plt.bar(labels, values, color=["#2E86C1", "#C0392B"])
plt.ylabel("Vrijeme (s)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{timestamp}/05_openmp_vs_simd.png", dpi=220)


# ========================================================
# 6) Line chart – trend vremena
# ========================================================
plt.figure(figsize=(12,6))
plt.plot(methods, times, marker="o", linewidth=3, color="#2E86C1")
plt.title("Trend vremena svih metoda", fontsize=14)
plt.xlabel("Metoda")
plt.ylabel("Vrijeme (s)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{timestamp}/06_time_trend.png", dpi=220)


print("\n✅ Svi grafovi su uspješno generisani!")
print(f"📁 Pogledaj folder: {timestamp}/")
