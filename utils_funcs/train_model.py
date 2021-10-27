import time
import datetime
import yaml

from sklearn.model_selection import KFold, cross_val_score

import matplotlib.pyplot as plt


with open('config.yaml') as f:
    config = yaml.load(f, yaml.FullLoader)

def get_best_param(X_train, y_train, create_model, model_name):
    param = config[model_name]['param']
    values = eval(config[model_name]['values'])

    accuracy = []
    best_param = 0
    best_score = 0
    for k in values:
        model = create_model(**{param: k})
        kf = KFold(config['k_folds'], random_state=config['seed'], shuffle=True)
        scores = cross_val_score(model, X_train, y_train, cv=kf)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_param = k
        accuracy.append(scores.mean())

    plt.plot(values, accuracy)
    plt.show()
    print (f"Best {param} = ", best_param)
    print ("Best score = ", best_score)
    return best_param

def get_score(X_train, y_train, X_test, y_test, best_param, create_model, model_name):
    param = config[model_name]['param']
    
    start_time = datetime.datetime.now()
    model = create_model(**{param: best_param})
    model.fit(X_train, y_train)  
    y_test_predict = model.predict(X_test)
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

    print ("False positive error = ", float(fp) / len(y_test))
    print ("True negative error = ", float(tn) / len(y_test))
    print ("False negative = ", float(fn) / f)
    print ("True positive = ", float(tp) / t)
    print ("TIME = ", datetime.datetime.now() - start_time)