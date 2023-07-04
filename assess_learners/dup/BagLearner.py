import numpy as np


class BagLearner:
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.learners = []

    def author(self):
        return 'dpanirwala3'

    def add_evidence(self, X, Y):
        for _ in range(self.bags):
            index = np.random.choice(
                range(X.shape[0]), size=X.shape[0], replace=True)
            X_bag = X[index]
            Y_bag = Y[index]

            learner = self.learner(**self.kwargs)
            learner.add_evidence(X_bag, Y_bag)

            self.learners.append(learner)

    def query(self, X):
        results = []
        for learner in self.learners:
            result = learner.query(X)
            results.append(result)

        return np.mean(results, axis=0)
