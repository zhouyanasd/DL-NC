'''
The type of data is N groups in a python list,
and each group is a numpy array which form is [feature, number].
 Example: python-list[np.array[feature, number],...]
 [array([[0, 0, 1, ..., 0, 0, 0],
       [2, 2, 0, ..., 1, 0, 4],
       [1, 0, 2, ..., 3, 0, 1],
       [1, 1, 0, ..., 1, 0, 1],
       [1, 1, 3, ..., 0, 5, 0]]), array([[3, 2, 1, ..., 2, 3, 1],
       [1, 3, 0, ..., 1, 2, 0],
       [2, 1, 1, ..., 3, 2, 2],
       [3, 0, 0, ..., 1, 0, 1],
       [3, 1, 3, ..., 2, 0, 7]])
'''

from .simple import Simple