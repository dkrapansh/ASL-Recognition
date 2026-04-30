import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
labels = np.asarray([letters[int(l)] for l in labels])
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

cv_scores = cross_val_score(model, data, labels, cv=5)
print("Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)".format(
    cv_scores.mean() * 100, cv_scores.std() * 100))

print("\nPer-Class Performance:")
print(classification_report(y_test, y_predict))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved.")