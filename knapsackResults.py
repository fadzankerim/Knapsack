import csv
import matplotlib.pyplot as plt

methods   = []
times     = []
speedups  = []

# Učitaj CSV
with open("results.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        methods.append(row["Method"])
        times.append(float(row["TimeSeconds"]))
        speedups.append(float(row["SpeedupVsClassic"]))

# 1) Graf vremena 
plt.figure(figsize=(8, 4))
plt.title("Vrijeme izvrsavanja knapsack metoda")
plt.bar(methods, times)
plt.ylabel("Vrijeme (s)")
plt.xlabel("Metoda")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("knapsack_times.png", dpi=200)
# plt.show()

# 2) Graf ubrzanja 
plt.figure(figsize=(8, 4))
plt.title("Ubrzanje u odnosu na Classic 2D")
plt.bar(methods, speedups)
plt.ylabel("Ubrzanje (x)")
plt.xlabel("Metoda")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("knapsack_speedups.png", dpi=200)
# plt.show()
