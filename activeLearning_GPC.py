import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_uncertainty
from scenario import create_sample_AL, run_TC_on_BeamNG, str_to_beamNG_TcFormat
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from performance import compare_learning_curve_model, compare_performance
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import sys_out
import time

from random import randint
from scenario import gen_testcase
#read_data('normalize_left_curve.cvs')
def read_data(file):
    ''' 
    Read data (testcase) from cvs file name
    '''
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    store_dir = os.path.join(curr_dir, 'store')
    file_name = file_name = os.path.join(store_dir, file)
    df = pd.read_csv(file_name)
    feature_col = ['Road_shape', 'speed','light' ,'weather', 'result']
    X = df.loc[:, ['Road_shape', 'speed','light' ,'weather']]
    y = df.loc[:, ['result']]
    return X ,y

def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]


def uncertainty_sampling_ps(classifier, X_pool):
    '''
    This is a select strategy technique for active sampling
    find the uncertainty of a sample from the pool. Classifier uncertainty, which is 1 - P(prediction is correct).
    :param classifier: model of classifier
    :param X_pool: pool of samples
    :return: index of selected sample, and a sample has high uncertainty
    '''
    uncertainty = classifier_uncertainty(classifier, X_pool)
    print("\n uncertainty of samples: ", uncertainty)
    idx = np.argmax(uncertainty)
    return idx, X_pool[idx]


def check_performance(learner, X_raw, y_raw):
    '''
    This function check the performance of the training model
    :param learner:
    :param X_raw: features of a sample
    :param y_raw: labeled of a sample
    :return: number of correct prediction, number of incorrect prediction, the accuracy of model
    '''
    predictions = learner.predict(X_raw)
    is_correct = (predictions == y_raw.ravel())
    num_correct = np.sum(is_correct)
    num_incorrect = is_correct.shape[0] - num_correct
    model_accuracy = learner.score(X_raw, y_raw)
    return num_correct, num_incorrect, model_accuracy

def update_mode_AL(learner, X_pool, X_pool_bng, X_test, y_test, budget):
    '''
    Update model for active learning process
    :param learner: Activelearning
    :param X_pool: set of testcases are normalize
    :param X_pool_bng: set of testcases are formated to run on BeamNG
    :param X_test: set of testcases for testing model
    :param y_test: set of the result of testcases for testing model
    :param budget: the budget to train model
    :return: number of queries are collected in the updating process, set of labeled testcases,
        and history of the accuracy in iterations
    '''
    sys_out.print_title("Start process the updated model by quesrying the samples from Pool")
    model_accuracy = learner.score(X_test, y_test)
    performance_history = []
    queries = list()
    count = 0
    N_BUDGET = budget
    while model_accuracy < 0.90 and count < N_BUDGET:
        count = count + 1
        query_idx, query_instance = learner.query(X_pool)
        query_idx = int(query_idx)
        #run testcase on BeamNG
        y_beamNG = run_TC_on_BeamNG(str_to_beamNG_TcFormat(X_pool_bng[query_idx])) # todo uncommand this to run real active learning
        #y_beamNG = randint(0, 1)
        #update model with th result from beamNG
        learner.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=np.asarray(y_beamNG).reshape(1, )
        )
        query = np.insert(X_pool[query_idx], 4, y_beamNG)
        print("\n Query instance", query)
        queries.append(query)
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        X_pool_bng = np.delete(X_pool_bng, query_idx, axis=0)
        #model_accuracy = learner.score(X_test, y_test)
        num_correct, num_incorrect, accuracy = check_performance(learner, X_test, y_test)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=count, acc=accuracy))
        # Save our model's performance for plotting.
        performance_history.append([num_correct, num_incorrect, accuracy])
    return count, queries, performance_history


def query_sample_from_updated_Model(learner, num_query_update_model, num_query, X_pool, X_pool_bng, queries):
    '''
    Once model reaches the threshold accuracy, the labled of testcase is predicted by the updated model
    :param learner: Activelearning
    :param num_query_upModel: number of queries which are collected during updating model process
    :param num_query: number of samples which we need to collect for sampling process
    :param X_pool: set of testcases are normalized
    :param X_pool_bng: set of testcases are formated to run on beamNG
    :param queries: set of all testcases are sampled in the updating process
    :return: final set of testcases are sampled for whole process
    '''
    sys_out.print_title("Start process queries lables base on the updated model ")
    while num_query_update_model < num_query:
        num_query_update_model = num_query_update_model + 1
        query_idx, query_instance = learner.query(X_pool)
        query_idx = int(query_idx)
        query_instance = np.asarray(query_instance).reshape(1,-1)
        y_predic = learner.predict(query_instance)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        X_pool_bng = np.delete(X_pool_bng, query_idx, axis=0)
        query = np.insert(X_pool[query_idx], 4, y_predic)
        print("\n Query sample and get label: ", query)
        queries.append(query)
    return queries

def pool_base_sampling(num_pool, num_initial, num_query, road_shape, source_data_test_name,
                       select_strategy= "uncertainty"):
    '''
    Active learning with pool base samling method
    :param num_pool: number of testcases in pool
    :param num_initial: number of initial testcases to train initial model of active learning
    :param num_query: numbers of testcases which need to request label
    :param road_shape: type of road for the testcases which we would like to generate and request label
    :return: A set of testcase which are labled as safe or unsafe
    '''
    num_pool, num_initial, num_query = int (num_pool), int(num_initial), int(num_query)
    X_initial, result_init, X_pool, X_pool_bng = create_sample_AL(num_pool, num_initial, road_shape)
    loop = 0
    while True:
        #this check to make sure the initial result must have different classes
        if (0 in result_init) and (1 in result_init): 
            break
        else:
            sys_out.warning("The classies of testcases should be the different class \
                    \(Not only one class in initial \)")
            X_initial, result_init, X_pool, X_pool_bng = create_sample_AL(num_pool, num_initial, road_shape)
            loop = loop +1
        if loop > 5:
            break

    # initial training data
    X_train = np.asarray(X_initial)
    y_train = np.asarray(result_init)
    print("\n X initial: ", X_train)
    print("\n  y initial: ", y_train)

    # initializing the active learner
    kernel = 1.0 * RBF(1.0)
    learner = ""
    if select_strategy == "random":
        learner = ActiveLearner(
            estimator=GaussianProcessClassifier(kernel=kernel, random_state=0),
            query_strategy=random_sampling,
            X_training=X_train, y_training=y_train
        )
    else:
        learner = ActiveLearner(
            estimator=GaussianProcessClassifier(kernel=kernel, random_state=0),
            query_strategy=uncertainty_sampling_ps,
            X_training=X_train, y_training=y_train
        )
    X_raw,y_raw = read_data(source_data_test_name)
    y_raw = y_raw.to_numpy()
    num_correct, num_incorrect, accuracy = check_performance(learner, X_raw, y_raw)

    sys_out.print_title("Initial model :")
    print("\n Number of correct testing samples", num_correct)
    print("\n Number of incorrect testing samples: ", num_incorrect)
    print("\n Accuracy: ", accuracy)

    #Update our model by pool-based sampling
    sys_out.print_sub_tit("Update model by pool sampling \
        \n Allow our model to query our unlabeled dataset for the most \
        \n informative points according to our query strategy (uncertainty sampling).")

    num_query_upModel, queries, performance_history = update_mode_AL(learner,
                                                       X_pool, X_pool_bng, X_raw, y_raw, 200)
    print("\n\n Number of samples which are collect from updating model process", num_query_upModel )
    sys_out.print_title("Performance History of active learning process")
    print( performance_history)

    #check if the number of required queries < the number in updating model process
    # then select queries from the sample list of updating model,
    # else predict sample with the updated model.

    num_correct_rest, num_incorrect_rest, accuracy_rest = check_performance(learner, X_raw, y_raw)
    if num_query_upModel > num_query:
        queries = queries[:num_query]
        performance_history = performance_history[:num_query]
        sys_out.print_title("Final result of all queries")
        print ( np.asarray(queries))
        return np.asarray(queries), performance_history
    else:
        final_queries = query_sample_from_updated_Model(learner, num_query_upModel,
                                                        num_query, X_pool, X_pool_bng, queries)
        remain_query_sample = num_query - num_query_upModel
        for _ in range(remain_query_sample):
            performance_history.append([num_correct_rest, num_incorrect_rest, accuracy_rest])
        sys_out.print_title("Final result of all queries")
        print (np.asarray(final_queries))
        return np.asarray(final_queries), performance_history

def create_fake_data(num_pool,road_shape ):
    beanNG_tc_list, normalize_tc_list = gen_testcase(num_pool, road_shape)
    result = [randint(0, 1) for i in range(num_pool)]
    X = np.asarray(normalize_tc_list)

    y = np.asarray(result)
    queries = np.zeros((num_pool,5))
    queries[:,:4] = X
    queries[:, 4] = y
    return queries

def main(number_feature):

    start_time = time.time()
    sample = np.empty([1,5])
    if number_feature == '1':
        # run pool_base_sampling with uncertaity method
        sample_uncer, performance_uncer = pool_base_sampling(1000, 3, 500,
                    'left_curve', 'normalize_left_curve.cvs', 'uncertainty')
        sys_out.write_data_queries_file(sample_uncer, "Uncertainty_queries")
        sys_out.write_perfromance(performance_uncer, "performance_uncertainty")

    elif number_feature == '2': # run pool_base_sampling with uncertaity method
        sample_random, performance_random = pool_base_sampling(1000,3,500,
                     'left_curve', 'normalize_left_curve.cvs', 'random')
        sys_out.write_data_queries_file(sample_random, "Random_queries")
        sys_out.write_perfromance(performance_random, "Performance_Random")

    elif number_feature == '3':    # draw the learning curve
        random_sample = sys_out.read_samples_from_queries('Uncertainty_queries_10_17_2019__024059.cvs')
        uncertainty_sample = sys_out.read_samples_from_queries('Random_queries_10_18_2019__024906.cvs')
        compare_learning_curve_model(random_sample, uncertainty_sample)

    elif number_feature == '4':  # compare the number error of two model
        # ,NonError,Error,Accuracy are type of the graph which we would like to show
        random_perf_file = 'Performance_Random_10_18_2019__024906.cvs'
        uncer_perf_file = 'performance_uncertainty_10_17_2019__024059.cvs'
        compare_performance(random_perf_file, uncer_perf_file,"Error")
        compare_performance(random_perf_file, uncer_perf_file, "NonError")
        compare_performance(random_perf_file, uncer_perf_file, "Accuracy")
        plt.show()

    end_time = time.time() -start_time
    sys_out.print_title("Run time:  " + str(end_time) + " Seconds")

if __name__ == "__main__":
    main(sys.argv[1])
