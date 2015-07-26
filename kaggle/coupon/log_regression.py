"""Logistical Regression Script 1.0

Script for Coupon Logistical regression. Uses a simple logistical regression
over the set of features using only 1-Dimensional features without any
combinations.
"""

import csv
import datetime, time


integer_indices = [2, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
date_hour_indices =  [5, 6]
date_indices = [8, 9]

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

def main():
  feature_file = open('data/coupon_list_train.csv', 'r')

  coupon_data = {}  # Access feature data using coupon hash
  with open('data/coupon_list_train.csv', 'rb') as csvfile:
    features = csv.reader(csvfile, delimiter=',')
    feature_list = features.next()
    for feature in features:
      coupon_hash, feature_attr = parse_coupon_line(feature)
      coupon_data[coupon_hash] = feature_attr
    print feature_list

if __name__ == '__main__':
  main()
