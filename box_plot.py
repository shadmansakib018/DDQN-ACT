import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("box_plot_data.csv", engine="openpyxl")

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    [df[col].dropna() for col in df.columns],
    labels=df.columns,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="navy"),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="navy"),
    capprops=dict(color="navy"),
    flierprops=dict(marker="o", color="gray", alpha=0.5),
)

# Clip y-axis so outliers don't shrink the boxes; annotate clipped values
q3  = df.quantile(0.75)
iqr = df.quantile(0.75) - df.quantile(0.25)
y_max = (q3 + 3 * iqr).max()
ax.set_ylim(top=y_max)

# Annotate any column whose max exceeds the clip limit
for i, col in enumerate(df.columns, start=1):
    col_max = df[col].max()
    if col_max > y_max:
        ax.annotate(
            f"max: {col_max:.1f}",
            xy=(i, y_max),
            xytext=(i, y_max * 0.97),
            ha="center", va="top", fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

ax.set_title("Box and Whisker Plot", fontsize=14)
ax.set_ylabel("Value")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("box_plot_2.png", dpi=150)
plt.show()
