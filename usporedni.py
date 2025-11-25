import csv
import matplotlib.pyplot as plt
import numpy as np

methods = []
times = []
speedups = []

with open("results.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        methods.append(row["Method"])
        times.append(float(row["TimeSeconds"]))
        speedups.append(float(row["SpeedupVsClassic"]))

x = np.arange(len(methods))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

bars1 = ax1.bar(x - width/2, times, width, label='Time (s)')
bars2 = ax2.bar(x + width/2, speedups, width, label='Speedup (x)', color='orange')

ax1.set_ylabel("Time (Seconds)")
ax2.set_ylabel("Speedup (x)")
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_title("Knapsack Performance – time vs speedup")

plt.tight_layout()
plt.savefig("combined_chart.png", dpi=200)
