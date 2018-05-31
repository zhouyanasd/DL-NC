import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

X = pd.Series(['first','second', 'third', 'fourth']*6)
Y = pd.Series([1.0, 0.508, 0.548, 0.662,
               1.0, 0.75, 0.57, 0.68,
               0.504, 0.522, 0.57, 0.89,
               0.498, 0.552, 0.594, 0.99,
               0.73, 0.52, 0.62, 0.95,
               0.644, 0.563, 0.617, 0.943])
parameter_change = pd.Series(['initial parameter','initial parameter','initial parameter','initial parameter',
                              'neuron decay t = 200ms', 'neuron decay t = 200ms', 'neuron decay t = 200ms', 'neuron decay t = 200ms',
                              'S_EE = S_EI = S_IE = S_II = 0', 'S_EE = S_EI = S_IE = S_II = 0', 'S_EE = S_EI = S_IE = S_II = 0', 'S_EE = S_EI = S_IE = S_II = 0',
                              'R = 1', 'R = 1', 'R = 1', 'R = 1',
                              'R = 1.2', 'R = 1.2', 'R = 1.2', 'R = 1.2',
                              'C_EE = 0.2 C_EI = 0.4, C_IE = 0.6, C_II = 0.1', 'C_EE = 0.2 C_EI = 0.4, C_IE = 0.6, C_II = 0.1', 'C_EE = 0.2 C_EI = 0.4, C_IE = 0.6, C_II = 0.1', 'C_EE = 0.2 C_EI = 0.4, C_IE = 0.6, C_II = 0.1'])

results = pd.DataFrame({'X':X, 'Y':Y, 'parameter_change': parameter_change})

g = sns.factorplot(x="X", y="Y", hue="parameter_change", data=results,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("accuracy")
g.set_xlabels("segment")
plt.show()