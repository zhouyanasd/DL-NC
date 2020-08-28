import numpy as np


def mse(y):
    """
    Mean squared error for decision tree (ie., mean) predictions
    """
    return np.mean((y - np.mean(y)) ** 2)


def entropy(y):
    """
    Entropy of a label sequence
    """
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y):
    """
    Gini impurity (local entropy) of a label sequence
    """
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum([(i / N) ** 2 for i in hist])



def bootstrap_sample(X, Y):
    N, M = X.shape
    idxs = np.random.choice(N, N, replace=True)
    return X[idxs], Y[idxs]


class Node:
    def __init__(self, left, right, rule):
        self.left = left
        self.right = right
        self.feature = rule[0]
        self.threshold = rule[1]


class Leaf:
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region
        """
        self.value = value


class DecisionTree:
    def __init__(
        self,
        classifier=True,
        max_depth=None,
        n_feats=None,
        criterion="entropy",
        seed=None,
    ):
        """
        A decision tree model for regression and classification problems.

        Parameters
        ----------
        classifier : bool
            Whether to treat target values as categorical (classifier =
            True) or continuous (classifier = False). Default is True.
        max_depth: int or None
            The depth at which to stop growing the tree. If None, grow the tree
            until all leaves are pure. Default is None.
        n_feats : int
            Specifies the number of features to sample on each split. If None,
            use all features on each split. Default is None.
        criterion : {'mse', 'entropy', 'gini'}
            The error criterion to use when calculating splits. When
            `classifier` is False, valid entries are {'mse'}. When `classifier`
            is True, valid entries are {'entropy', 'gini'}. Default is
            'entropy'.
        seed : int or None
            Seed for the random number generator. Default is None.
        """
        if seed:
            np.random.seed(seed)

        self.depth = 0
        self.root = None

        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth if max_depth else np.inf

        if not classifier and criterion in ["gini", "entropy"]:
            raise ValueError(
                "{} is a valid criterion only when classifier = True.".format(criterion)
            )
        if classifier and criterion == "mse":
            raise ValueError("`mse` is a valid criterion only when classifier = False.")

    def fit(self, X, Y):
        """
        Fit a binary decision tree to a dataset.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            An array of integer class labels for each example in `X` if
            self.classifier = True, otherwise the set of target values for
            each example in `X`.
        """
        self.n_classes = max(Y) + 1 if self.classifier else None
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow(X, Y)

    def predict(self, X):
        """
        Use the trained decision tree to classify or predict the examples in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features

        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The integer class labels predicted for each example in `X` if
            self.classifier = True, otherwise the predicted target values.
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        """
        Use the trained decision tree to return the class probabilities for the
        examples in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features

        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N, n_classes)`
            The class probabilities predicted for each example in `X`.
        """
        assert self.classifier, "`predict_class_probs` undefined for classifier = False"
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def _grow(self, X, Y, cur_depth=0):
        # if all labels are the same, return a leaf
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
            return Leaf(prob) if self.classifier else Leaf(Y[0])

        # if we have reached max_depth, return a leaf
        if cur_depth >= self.max_depth:
            v = np.mean(Y, axis=0)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v)

        cur_depth += 1
        self.depth = max(self.depth, cur_depth)

        N, M = X.shape
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)

        # greedily select the best split according to `criterion`
        feat, thresh = self._segment(X, Y, feat_idxs)
        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()

        # grow the children that result from the split
        left = self._grow(X[l, :], Y[l], cur_depth)
        right = self._grow(X[r, :], Y[r], cur_depth)
        return Node(left, right, (feat, thresh))

    def _segment(self, X, Y, feat_idxs):
        """
        Find the optimal split rule (feature index and split threshold) for the
        data according to `self.criterion`.
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            thresholds = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels
            gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresholds[gains.argmax()]

        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        """
        Compute the impurity gain associated with a given split.

        IG(split) = loss(parent) - weighted_avg[loss(left_child), loss(right_child)]
        """
        if self.criterion == "entropy":
            loss = entropy
        elif self.criterion == "gini":
            loss = gini
        elif self.criterion == "mse":
            loss = mse

        parent_loss = loss(Y)

        # generate split
        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # impurity gain is difference in loss before vs. after split
        ig = parent_loss - child_loss
        return ig

    def _traverse(self, X, node, prob=False):
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            return node.value
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        return self._traverse(X, node.right, prob)


class RandomForest:
    """
    A random forest of decision trees.
    """

    def __init__(
        self, n_trees, max_depth, n_feats, classifier=True, criterion="entropy"
    ):
        """
        An ensemble (forest) of decision trees where each split is calculated
        using a random subset of the features in the input.

        Parameters
        ----------
        n_trees : int
            The number of individual decision trees to use within the ensemble.
        max_depth: int or None
            The depth at which to stop growing each decision tree. If `None`,
            grow each tree until the leaf nodes are pure.
        n_feats : int
            The number of features to sample on each split.
        criterion : str (default: 'entropy')
            The error criterion to use when calculating splits. Valid entries
            are {'entropy', 'gini'}.
        """
        self.trees = []
        self.n_trees = n_trees
        self.n_feats = n_feats
        self.max_depth = max_depth
        self.criterion = criterion
        self.classifier = classifier

    def fit(self, X, Y):
        """
        Create `n_trees`-worth of bootstrapped samples from the training data
        and use each to fit a separate decision tree.
        """
        self.trees = []
        for _ in range(self.n_trees):
            X_samp, Y_samp = bootstrap_sample(X, Y)
            tree = DecisionTree(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                criterion=self.criterion,
                classifier=self.classifier,
            )
            tree.fit(X_samp, Y_samp)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target value for each entry in X.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features

        Returns
        -------
        y_pred : np.array of shape (N,)
            Model predictions for each entry in X.
        """
        tree_preds = np.array([[t._traverse(x, t.root) for x in X] for t in self.trees])
        return self._vote(tree_preds)

    def _vote(self, predictions):
        """
        Return the aggregated prediction across all trees in the RF for each problem.

        Parameters
        ----------
        predictions : np.array of shape (n_trees, N)
            The array of predictions from each decision tree in the RF for each
            of the N problems in X

        Returns
        -------
        y_pred : np.array of shape (N,)
            If classifier is True, the class label predicted by the majority of
            the decision trees for each problem in X. If classifier is False,
            the average prediction across decision trees on each problem.
        """
        if self.classifier:
            out = [np.bincount(x).argmax() for x in predictions.T]
        else:
            out = [np.mean(x) for x in predictions.T]
        return np.array(out)
