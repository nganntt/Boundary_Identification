import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
import os
from sys_out import warning
import pandas as pd

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    '''
    Draw learning curve (When building machine learning models, we want to keep error as low as possible)
    A common way of evaluating the performance of active learning algorithm is to plot the learning curve.
    Where the X-axis is the number samples of queried, and the Y-axis is the corresponding error rate.
    there are two error scores to monitor: one for the validation set, and one for the training sets.
    These are called learning curves, a learning curve shows how error changes as the training set size increases
    :param estimator:
    :param title: title of figure
    :param X:
    :param y:
    :param ylim: limit value of y
    :param cv: cross validation
    :param n_jobs:
    :param train_sizes: number of samples
    :return: None
    '''
    plt.figure(figsize=[7,5])
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print("\n train size", train_sizes)
    print("\n train score", train_scores)
    print("\n test score", test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def compare_learning_curve_model( data_random, data_uncertainty):
    '''
     Compare learning curve of the models which use random technique and uncertainty technique
    :param data_random: samples which are collect from random method
    :param data_uncertainty: amples which are collect from uncertainty method
    :return: None
    '''
    X_random =  data_random[:,0:4]
    y_random = data_random[:,4]
    X_uncertainty = data_uncertainty[:, 0:4]
    y_uncertainty = data_uncertainty[:, 4]

    kernel = 1.0 * RBF(1.0)
    title = "Learning Curves (Random Method)"
    estimator = GaussianProcessClassifier(kernel=kernel, random_state=0)
    plot_learning_curve(estimator, title, X_random, y_random, ylim=(0, 2), cv=10)

    title = "Learning Curves (Uncertainty Method)"
    # Cross validation with 100 iterations to get smoother mean test and train
    plot_learning_curve(estimator, title, X_uncertainty, y_uncertainty, ylim=(0, 2), cv=10)
    plt.show()

def compare_two_values_selected_strateger(num_query, value1, value2, title, compare_type, ylim=None):
    '''
    This function compares the difference in history performance of active learning model.
    The comparison can compare error, inerror, accuracy from 2 selected strategies
    :param num_query: number of sample which are collected in active learning process
    :param value1: error classification, incorrect classification, accuracy of classification
    :param value2: error classification, incorrect classification, accuracy of classification
    :param title: title of figure
    :param ylim: limit of y
    :return: None
    '''
    X = np.arange(1, num_query + 1, 5)
    row = int(num_query/5)
    value1 = np.average(value1.reshape(row,5), axis=1)
    value2 = np.average(value2.reshape(row, 5),axis=1)
    plt.figure(figsize=[7, 5])
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(compare_type)
    # y_pef_rand_mean = np.mean(value1, axis=0)
    # y_pef_rand_std = np.std(value1, axis=0)
    # y_pef_uncer_mean = np.mean(value2, axis=0)
    # y_pef_uncer_std = np.std(value2, axis=0)
    plt.grid()
    # plt.fill_between(X, value1 - y_pef_rand_std,
    #                   value1 + y_pef_rand_std, alpha=0.1, color="r")
    #
    # plt.fill_between(X, value2 - y_pef_uncer_std,
    #                  value2 + y_pef_uncer_std, alpha=0.1, color="g")
    plt.plot(X, value1, 'o-', color="r", label="Random Values")
    plt.plot(X, value2, 'o-', color="g", label="Uncertainty Values")
    plt.legend(loc="best")

    return plt

#format of performance from active learning
def fake_data_compare_perf(query):
    '''
    use this to test the graph from compare ration between 2 grap
    :param query:
    :return:
    '''
    num_query = query
    acc = np.random.uniform(0.0, 1.0, num_query)
    error = np.random.randint(1,num_query,num_query)
    non_error = 100 - error
    perf = np.empty([num_query,3],dtype=float)
    perf = np.asarray(list(zip(non_error,error, acc)))
    print("perf", perf)
    return perf

def read_performance_sampling(file):
    '''
       Read data of the performance after the sampling process
       Data is saved in cvs file
       '''
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    store_dir = os.path.join(curr_dir, 'store')
    file_name = os.path.join(store_dir, file)
    df = pd.read_csv(file_name)
    feature_col = ['NonError', 'Error', 'Accuracy']
    perf = df.loc[:, feature_col]
    perf = perf.to_numpy()
    return perf

def read_samples_from_queries(file):
    '''
       Read data of the performance after the sampling process
       Data is saved in cvs file
       '''
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    store_dir = os.path.join(curr_dir, 'store')
    file_name = file_name = os.path.join(store_dir, file)
    df = pd.read_csv(file_name)
    feature_col = ['Road_shape', 'speed', 'light', 'weather', 'result']
    queries = df.loc[:, feature_col]
    return queries.to_numpy()



    curr_dir = os.path.dirname(os.path.abspath(__file__))
    store_dir = os.path.join(curr_dir, 'store')
    file_name = os.path.join(store_dir, file)
    df = pd.read_csv(file_name)
    feature_col = ['NonError', 'Error', 'Accuracy']
    perf = df.loc[:, feature_col]
    perf = perf.to_numpy()
    return perf


def compare_performance( file_perf_random, file_perf_uncer, type_compare):
    '''
    Compare error of two strateger
    :param file_perf_random: file name of the samples which are collected by random mehthod
    :param file_perf_uncer: file name of the samples which are collected by random mehthod
    :param type_compare: type of compare: error, non_error, accuracy
    :return:
    '''
    perf_random = read_performance_sampling(file_perf_random)
    perf_uncer = read_performance_sampling(file_perf_uncer)
    if perf_random.shape[0] != perf_uncer.shape[0]:
        warning("The number of array in the performance should be equal")
        return 0
    else:
        #,NonError,Error,Accuracy are type of the graph which we would like to show
        if type_compare == "NonError":
            no_err_random = perf_random[:,0]
            no_err_uncer = perf_uncer[:,0]
            title = "NON ERROR between RANDOM method and UNCERTAINTY method"
            plt = compare_two_values_selected_strateger(perf_random.shape[0], no_err_random,
                                                  no_err_uncer, title, type_compare)
        elif type_compare == "Error":
            err_random = perf_random[:, 1]
            err_uncer = perf_uncer[:, 1]
            title = "ERROR between RANDOM method and UNCERTAINTY method"
            plt = compare_two_values_selected_strateger(perf_random.shape[0], err_random,
                                                      err_uncer, title, type_compare)
        elif type_compare == "Accuracy":
            acc_random = perf_random[:, 2]
            acc_uncer = perf_uncer[:, 2]
            title = "ACCURACY between RANDOM method and UNCERTAINTY method"
            plt = compare_two_values_selected_strateger(perf_random.shape[0], acc_random,
                                                  acc_uncer, title, type_compare)

        return plt


