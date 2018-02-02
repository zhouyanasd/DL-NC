# import numpy as np
# from sklearn.metrics import roc_auc_score
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# print(roc_auc_score(y_true, y_scores))

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
def get_optimal_threshold(fpr,tpr,thresholds):
    r = list(tpr - fpr)
    return thresholds[r.index(max(r))]

y = np.array([1, 1, 2, 2,1,1,2,2,1,2,1])
scores = np.array([0.1, 0.4, 0.35, 0.8,0.1,0.3,0.4,0.7,0.3,0.5,0.5])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)
print(fpr,tpr,thresholds)
print(get_optimal_threshold(fpr,tpr,thresholds))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print(get_optimal_threshold(fpr,tpr,thresholds))