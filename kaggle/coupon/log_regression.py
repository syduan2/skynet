"""Logistical Regression Script 1.0

Script for Coupon Logistical regression. Uses a simple logistical regression
over the set of features using only 1-Dimensional features without any
combinations.
"""

from collections import defaultdict
import datetime, time, multiprocessing, argparse
from scipy.special import expit
from scipy.optimize import fmin_bfgs
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn import linear_model


integer_indices = [2, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
date_hour_indices =  [5, 6]
date_indices = [8, 9]

def get_arguments():
  parser = argparse.ArgumentParser(description='Coupon Logistical regression.')
  parser.add_argument()


  return arguments



def parse_time(time_string, includes_time):
  """Function for parsing the date/time string

  Args:
    time_string: the string form of the date/time combination
    includes_time: True/False value if the string includes time as well as date
  Returns:
    float representing the ctime of the date.
  """

  if time_string == 'NA':
    return -1.0
  elif includes_time:
    time_instance = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
  else:
    time_instance = datetime.datetime.strptime(time_string, '%Y-%m-%d')

  return time.mktime(time_instance.timetuple())


def parse_coupon_line(line_data):
  """Clean up and organize the data in the feature data.

  Takes in the the string of one line of the input feature data and parses it
  into a tuple so that it can be easily stored into a dictionary.

  Args:
    data: one line of the feature data csv file
  Returns:
    Tuple with first element being the coupon hash and the second value a list
    of the remaining feature data.
  """
  coupon_hash = line_data[-1]
  features = line_data[0 : len(line_data) - 1]

  feature_list = []
  for i in range(len(features)):
    if features[i] == 'NA':
      feature_list.append(False)
    elif i in integer_indices:
      feature_list.append(int(features[i]))
    elif i in date_hour_indices:
      feature_list.append(parse_time(features[i], True))
    elif i in date_indices:
      feature_list.append(parse_time(features[i], False))
    else:
      feature_list.append(features[i])

  return (coupon_hash, feature_list)

def logistic_cost(y, feature_list, theta):


class Classifiers(object):
  def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, **kwargs):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.cv = cv

  def set(self, X_train, y_train, X_test, y_test):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test

  def set_cv(self, cv):
    self.cv = cv    

  def set_clf(self, clf):
    self.clf = clf

  def learning_algorithm(self, cv, algorithm, **kwargs):
    self.cv = cv
    if algorithm == 'SVM':
      self.SVM(**kwargs)
    elif algorithm == 'logistic_regression':
      self.logistic_regression(**kwargs)
    elif algorithm == 'LinearSVC'
      self.LinearSVC(**kwargs)
    else:
      raise Exception('Currently we can only handle SVM, logistic regression and LinearSVC')

  def LinearSVC(self, **kwargs):
    C = 1.0 if 'C' not in kwargs else kwargs['C']
    parameters = {'classifier__C': C}
    random_state = None if 'random_state' not in kwargs else kwargs['random_state']
    classifier = svm.LinearSVC(random_state=random_state)
    parallel_processing(classifier, parameters, **kwargs)

  def logistic_regression(self, **kwargs):
    self.clf = linear_model.LogisticRegression()
    self.clf.fit(self.X_train, self.y_train)
  
  def parallel_processing(classifier, parameters, **kwargs):
    scoring = 'accuracy' if 'scoring' not in kwargs else kwargs['scoring']
    # other scoring options: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    n_jobs = multiprocessing.cpu_count() if 'n_jobs' not in kwargs else kwargs['n_jobs']
    scaler = preprocessing.StandardScaler() if 'scaling' not in kwargs else kwargs['scaling']
    clf = Pipeline(steps = [('normalize', scaler), ('classifier', classifier)])
    self.clf = GridSearchCV(clf, param_grid=parameters, scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
    self.clf.fit(self.X_train, self.y_train)

    return clf_fold.grid_scores_, clf_fold.best_params_, results

  def predict(self, **kwargs):
    results = {}
    if 'decision' in kwargs and kwargs['decision'] == True:
      results.update({'decision': clf.decision_function(self.X_test)})
    if 'predict' in kwargs and kwargs['predict'] == True:
      results.update({'predict': clf.predict(self.X_test)})
    if 'predict_prob' in kwargs and kwargs['predict_prob'] == True:
      results.update({'predict_prob': clf.predict_proba(self.X_test)})
    if 'score' in kwargs and kwargs['score'] == True:
      results.update({'score': clf.score(self.X_test, self.y_test)})
    if 'get_params' in kwargs and kwargs['get_params'] == True:
      results.update({'params': clf.get_params()})

    return results
 

  def SVM(self, **kwargs): 
    kernel = 'rbf' if 'kernel' not in kwargs else kwargs['kernel']
    random_state = None if 'random_state' not in kwargs else kwargs['random_state']
    prob = True if 'prob' not in kwargs else kwarg['prob']
    C = 1.0 if 'C' not in kwargs else kwargs['C']
    gamma = 0.0 if 'gamma' not in kwargs else kwargs['gamma']
    parameters = {'classifier__C': C, 'classifier__gamma': gamma}
    classifier = svm.SVC(kernel=kernel, probability=prob, random_state=random_state)
    parallel_processing(classifier, parameters, **kwargs)


def main():
  feature_file = open('data/coupon_list_train.csv', 'r')

  coupon_data = {}  # Access feature data using coupon hash
  purchase_data = defaultdict(int)  # Dictionary of purchases with number of purchases as val
  with open('data/coupon_list_train.csv', 'rb') as csvfile:
    features = csv.reader(csvfile, delimiter=',')
    feature_list = features.next()
    for feature in features:
      coupon_hash, feature_attr = parse_coupon_line(feature)
      coupon_data[coupon_hash] = feature_attr

  with open('data/coupon_detail_train.csv', 'rb') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for purchase_line in csv_reader:
      purchase_key = (purchase_line['USER_ID_hash'],
                      purchase_line['COUPON_ID_hash'])
      purchase_data[purchase_key] += 1

  cv = sklearn.cross_validation.KFold(n= , n_folds=5, shuffle=True, random_state=None) 

  clf = Classifiers(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  clf.learning_algorithm(cv, 'logistic_regression')
  results = clf.predict(decision=True, predict=True, predict_prob=True, score=True, get_params=True)

if __name__ == '__main__':
  main()
