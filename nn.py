from sklearn.neural_network import MLPClassifier as nn

import sklearn.datasets as dt

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score as score

model = nn()

X, y = dt.make_moons(n_samples=30000)

xtr, xts, ytr, yts = tts(X, y)

model.fit(xtr, ytr)

print(score(yts, model.predict(xts)))
