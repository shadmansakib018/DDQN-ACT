import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

df = pd.read_excel("24hour_all.csv", engine="openpyxl")

x = np.linspace(0, 24, len(df))
x_smooth = np.linspace(0, 24, 300)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, ax = plt.subplots(figsize=(12, 6))

for col, color in zip(df.columns, colors):
    y = df[col].values
    spline = make_interp_spline(x, y, k=3)
    ax.plot(x_smooth, spline(x_smooth), label=col, color=color, linewidth=2)

ax.set_title("24-Hour Simulation Results", fontsize=14)
ax.set_xlabel("Hour")
ax.set_ylabel("Avg Response Time (s)")
ax.set_xticks(range(0, 25))
ax.legend()
ax.grid(linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("line_plot_24hr.png", dpi=150)
plt.show()
