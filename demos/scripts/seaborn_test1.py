import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# tips = pd.read_csv("../../Data/seaborn_data/tips.csv")
# sns.set(style="ticks")                                     #设置主题
# sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")   #palette 调色板
# plt.show()


sns.set()
# Load the example flights dataset and conver to long-form
flights_long = pd.read_csv("../../Data/seaborn_data/flights.csv")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.show()