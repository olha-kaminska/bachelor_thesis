import numpy as np
import yaml
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


with open('config.yaml') as f:
    config = yaml.load(f, yaml.FullLoader)
    
def read_data():
    with open(config['data']) as f:
        lines = f.readlines()
    X = []
    y = []
    for line in lines:
        if line[0] == 'h':
            y.append(0)
            string = line[4 : len(line) - 4]
            ss = ''
            for s in string:
                if s == ' ' or (ord(s) >= ord('a') and ord(s) <= ord('z')) or (ord(s) >= ord('A') and ord(s) <= ord('Z')):
                    ss += s
            X.append(ss)
        else:
            y.append(1)
            string = line[5 : len(line) - 4]
            ss = ''
            for s in string:
                if s == ' ' or (ord(s) >= ord('a') and ord(s) <= ord('z')) or (ord(s) >= ord('A') and ord(s) <= ord('Z')):
                    ss += s
            X.append(ss)
    y = np.array(y)
    return X, y

def preprocess_data(data):
    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(data)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    
    return X_train_tfidf

def train_test_split(data, y):
    return model_selection.train_test_split(data, y, test_size = config['test_size'], random_state = config['seed'])