from typing import Any, Callable, List, Union, Dict, Tuple
import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def stats(y_test: List, y_test_predict: List) -> None:
    start_time = datetime.datetime.now()
    fp = 0
    tn = 0
    fn = 0
    tp = 0
    f = 0
    t = 0
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_test_predict[i] == 1:
            fp += 1
            f += 1
        if y_test[i] == 0 and y_test_predict[i] == 0:
            fn += 1
            f += 1
        else:
            if y_test[i] == 1 and y_test_predict[i] == 0:
                tn += 1
                t += 1
            if y_test[i] == 1 and y_test_predict[i] == 1:
                tp += 1
                t += 1

    print("False positive error = ", float(fp) / len(y_test))
    print("True negative error = ", float(tn) / len(y_test))
    print("False negative = ", float(fn) / f)
    print("True positive = ", float(tp) / t)
    print("TIME = ", datetime.datetime.now() - start_time)


def calc(
        segment: List[Any],
        model_creator: Callable[
            [int],
            Union[
                KNeighborsClassifier,
                MultinomialNB,
                LogisticRegression,
                DecisionTreeClassifier,
                SVC
            ]
        ],
        X_train: List,
        y_train: List,
        config: Dict
) -> Tuple[int, int, List]:
    accuracy = []
    best_k = segment[0]
    best_score = 0
    for k in segment:
        kf = KFold(
            len(y_train),
            random_state=config['models_random_state'],
            shuffle=True
        )
        scores = cross_val_score(model_creator(k), X_train, y_train, cv=kf)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_k = k
        accuracy.append(scores.mean())
    return best_k, best_score, accuracy
