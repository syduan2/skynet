import numpy as np
import math
import random
import itertools
import scipy.misc
from collections import defaultdict
import matplotlib.pyplot as plt

input_size = 10
hidden_size = 15
output_size = 15

heb_learning_rate = 0.1
log_learning_rate = 0.05
data_len = 7
random.seed()
def sigmoid(x):
    return 1/(1+math.e ** (- 2.05 *x))

def heb_time_step(x, h, wxh, whh, val):
    x_new = np.ones_like(x) * 0.1

    # Calculate values
    x_new[0][val] = 1
    h_new = sigmoid(np.dot(x, wxh) + np.dot(h, whh))
    wxh = 2*sigmoid(wxh + heb_learning_rate * (np.dot(np.transpose(x), h_new) - np.dot(np.transpose(x_new), h))) - 1
    whh = 2*sigmoid(whh + heb_learning_rate * (np.dot(np.transpose(h), h_new) - np.dot(np.transpose(h_new), h))) - 1
    #for idx in range(len(whh)):
    #    whh[idx][idx] = 0
    return x_new, h_new, wxh, whh

def log_time_step(h, y, why):
    m = h.shape[0]
    y_pred = sigmoid(np.dot(h, np.transpose(why)))
    y_real = np.zeros_like(y_pred)
    if y != -1:
        y_real[0][y] = 1
    #error = np.sum((y_real - y_pred) ** 2)
        error = np.argmax(y_pred[0], 0) == y
    else:
        error = 1
    b = y_pred - y_real
    a = log_learning_rate * np.dot(np.transpose(y_pred - y_real), h)
    why -= 1 / m * np.dot(np.transpose(y_pred - y_real), h)

    y_pred = sigmoid(np.dot(h, np.transpose(why)))
    return why, y_pred, error
def data_gen():
    data = [1,1,1,1,1,1,7]
    i = 0
    while True:
        if i%data_len < len(data):
            yield data[i% len(data)]
        else:
            yield -1
        i += 1

def main():
    gen = data_gen()
    data = [gen.next() for i in range(20000)]
    value_x = np.random.rand(1, input_size)
    value_h = np.random.rand(1, hidden_size)

    weight_xh = (np.random.rand(input_size, hidden_size) - 0.5)
    weight_hh = (np.random.rand(hidden_size, hidden_size) - 0.5)
    weight_hy = (np.random.rand(hidden_size, output_size) - 0.5)

    graph  = []
    # Train the Auto-Encoder
    i=0
    j=0
    err_tot = 0
    for val in data:
        [value_x, value_h, weight_xh, weight_hh] = \
            heb_time_step(value_x, value_h, weight_xh, weight_hh, val)
        weight_hy, _, error = log_time_step(value_h, val, weight_hy)
        if i % data_len < 7:
            err_tot += error
            j+=1
            if j%data_len == 0:
                print float(err_tot) / j
                j = 0
                err_tot = 0
        i += 1
        graph.append(value_h[0])
    # Log Regression

    pic = np.vstack(graph)
    scipy.misc.imsave('outfile.jpg', graph)
    """
    vals = defaultdict(list)
    valslist = []
    for i in range(len(data)):
        val = data[i]
        _, value_h, _, _ = \
            heb_time_step(value_x, value_h, weight_xh, weight_hh, val)

        vals[i%data_len].append(value_h[0])
        valslist.append(value_h[0])
        print error
    pic = np.vstack(valslist)
    scipy.misc.imsave('outfile.jpg', graph)
    scipy.misc.imsave('outfile0.jpg', vals[0])
    scipy.misc.imsave('outfile1.jpg', vals[1])
    scipy.misc.imsave('outfile2.jpg', vals[2])"""
    #plt.plot(pic)
    #plt.show()



if __name__ == '__main__':
    main()
