import numpy as np


class RTLearner:
    def __init__(self, leaf_size=5, random_state=903262441, verbose=False):
        self.leaf_size = leaf_size
        self.random_state = random_state
        self.tree = {}

    def author(self):
        return "dpanirwala3"

    def add_evidence(self, X, y):
        self.tree = self.constructTree(X, y)

    def query(self, points):
        preds = []
        for point in points:
            preds.append(self.predict(point, self.tree))
        return np.array(preds)

    def getEntropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probArray = counts / len(y)
        entropy = -np.sum(probArray * np.log2(probArray))
        return entropy

    def getInfoGain(self, X, y, feature, threshold):
        L_index = X[:, feature] <= threshold
        R_index = X[:, feature] > threshold
        L_ent = self.getEntropy(y[L_index])
        R_ent = self.getEntropy(y[R_index])
        L_weight = len(y[L_index]) / len(y)
        R_weight = len(y[R_index]) / len(y)
        information_gain = self.getEntropy(
            y) - (L_weight * L_ent + R_weight * R_ent)
        return information_gain

    def constructTree(self, X, y):
        if len(np.unique(y)) == 1 or len(X) <= self.leaf_size:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            return {"class": unique_classes[np.argmax(class_counts)]}

        np.random.seed(self.random_state)
        n_features = X.shape[1]
        random_feature = np.random.choice(n_features)
        unique_values = np.unique(X[:, random_feature])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        best_info_gain = -np.inf
        best_threshold = None

        for threshold in thresholds:
            info_gain = self.getInfoGain(
                X, y, random_feature, threshold)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold

        if random_feature is None or best_threshold is None:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            return {"class": unique_classes[np.argmax(class_counts)]}

        L_index = X[:, random_feature] <= best_threshold
        R_index = X[:, random_feature] > best_threshold
        X_L = X[L_index]
        y_L = y[L_index]
        X_R = X[R_index]
        y_R = y[R_index]
        L_subtree = self.constructTree(X_L, y_L)
        R_subtree = self.constructTree(X_R, y_R)

        return {"feature": random_feature, "threshold": best_threshold, "L": L_subtree, "R": R_subtree}

    def predict(self, x, node):
        if "class" in node:
            return node["class"]

        if x[node["feature"]] <= node["threshold"]:
            return self.predict(x, node["L"])
        else:
            return self.predict(x, node["R"])


def author():
    return "dpanirwala3"
