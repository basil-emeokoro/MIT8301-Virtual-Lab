import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from mit8301_credit_risk.logistic_regression_scratch import LogisticRegressionScratch


def test_sigmoid_and_probability_bounds():
    values = LogisticRegressionScratch._sigmoid(np.array([-1000, 0, 1000]))
    assert np.all((values >= 0) & (values <= 1))
    assert values[1] == 0.5


def test_loss_decreases_and_predictions_agree_reasonably():
    X, y = make_classification(n_samples=500, n_features=8, random_state=42)
    X = StandardScaler().fit_transform(X)
    model = LogisticRegressionScratch(max_iter=3000).fit(X, y)
    benchmark = LogisticRegression(max_iter=2000).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert set(model.predict(X)) <= {0, 1}
    assert np.all((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1))
    assert abs(accuracy_score(y, model.predict(X)) - accuracy_score(y, benchmark.predict(X))) < 0.08
