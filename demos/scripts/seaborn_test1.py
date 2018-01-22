import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

tips = pd.read_csv("../../Data/seaborn_data/tips.csv")
sns.set(style="ticks")                                     #设置主题
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")   #palette 调色板
plt.show()