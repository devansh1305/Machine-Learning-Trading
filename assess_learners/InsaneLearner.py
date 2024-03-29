import BagLearner as bl
import LinRegLearner as lrl


class InsaneLearner():
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={
        }, bags=20, boost=False, verbose=False)] * 20

    def author(self):
        return 'dpanirwala3'

    def add_evidence(self, dataX, dataY):
        for learner in self.learners:
            learner.add_evidence(dataX, dataY)

    def query(self, points):
        out = []
        for learner in self.learners:
            out.append(learner.query(points))
            return sum(out) / len(out)
