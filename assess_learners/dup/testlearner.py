""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
"""


# Report experiment 1




import sys
import time
import matplotlib.pyplot as plt
import InsaneLearner as il
import BagLearner as bl
import RTLearner as rtl
import DTLearner as dtl
import LinRegLearner as lrl
import numpy as np
import math
def experiment1(train_x, train_y, test_x, test_y):
    max_leaf_size = 25
    in_sample_rsmes = []
    outSample_rsmes = []

    for each_leaf_size in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        in_predY = learner.query(train_x)
        in_RMSE = math.sqrt(((train_y - in_predY) ** 2).sum()/train_y.shape[0])
        in_sample_rsmes.append(in_RMSE)
        out_predY = learner.query(test_x)
        out_RMSE = math.sqrt(((test_y - out_predY) ** 2).sum()/test_y.shape[0])
        outSample_rsmes.append(out_RMSE)

    x_indice = range(1, max_leaf_size + 1)
    plt.plot(x_indice, in_sample_rsmes[:25], label="In Sample")
    plt.plot(x_indice, outSample_rsmes[:25], label="Out Sample")

    plt.title("Figure 1 - RSME vs Leaf Size for Decision Tree")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.grid()
    plt.legend()
    plt.savefig("images/figure1.png")
    plt.clf()


def experiment2(train_x, train_y, test_x, test_y):
    max_leaf_size = 10
    bag_size = 5
    in_sample_rsmes = []
    outSample_rsmes = []

    for each_leaf_size in range(1, max_leaf_size + 1):
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={
                                "leaf_size": each_leaf_size}, bags=bag_size, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)
        in_predY = learner.query(train_x)
        in_RMSE = math.sqrt(((train_y - in_predY) ** 2).sum()/train_y.shape[0])
        in_sample_rsmes.append(in_RMSE)
        out_predY = learner.query(test_x)
        out_RMSE = math.sqrt(((test_y - out_predY) ** 2).sum()/test_y.shape[0])
        outSample_rsmes.append(out_RMSE)

    x_indice = range(1, max_leaf_size + 1)
    plt.plot(x_indice, in_sample_rsmes, label="in sample")
    plt.plot(x_indice, outSample_rsmes, label="out sample")

    plt.title("Figure 2 - RSME vs Leaf Size for BagLearner and DT for 10 bags")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.grid()
    plt.legend()
    plt.savefig("images/figure2.png")
    plt.clf()


def experiment31(train_x, train_y):

    max_trainning_size = train_x.shape[0]
    runTime_dt = []
    runTime_rt = []

    for training_size in range(200, max_trainning_size + 1, 200):
        learner = dtl.DTLearner(leaf_size=1, verbose=False)
        start = time.time()
        learner.add_evidence(train_x[:training_size], train_y[:training_size])
        end = time.time()
        running_time = end - start
        runTime_dt.append(running_time)
        learner = rtl.RTLearner(leaf_size=1, verbose=False)
        start = time.time()
        learner.add_evidence(train_x[:training_size], train_y[:training_size])
        end = time.time()
        running_time = end - start
        runTime_rt.append(running_time)
    x_indice = range(200, max_trainning_size + 1, 200)
    plt.plot(x_indice, runTime_dt, label="Decision Tree")
    plt.plot(x_indice, runTime_rt, label="Random Tree")
    plt.title("Figure 3 - Training Time vs Size for Decision Tree and Random Tree")
    plt.xlabel("Trainning Sizes")
    plt.ylabel("Trainning Time/s")
    plt.grid()
    plt.legend()
    plt.savefig("images/figure3.png")
    plt.clf()


def experiment32(train_x, train_y, test_x, test_y):
    max_leaf_size = 20
    outSample_MAE_dt = []
    outSample_MAE_rt = []

    for each_leaf_size in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        out_predY = learner.query(test_x)
        outSample_MAE = np.mean(
            np.abs((np.asarray(test_y) - np.asarray(out_predY))))
        outSample_MAE_dt.append(outSample_MAE * 100)

        learner = rtl.RTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        out_predY = learner.query(test_x)
        outSample_MAE = np.mean(
            np.abs((np.asarray(test_y) - np.asarray(out_predY))))
        outSample_MAE_rt.append(outSample_MAE * 100)

    x_indice = range(1, max_leaf_size + 1)
    plt.plot(x_indice, outSample_MAE_dt, label="Decision Tree")
    plt.plot(x_indice, outSample_MAE_rt, label="Random Tree")
    plt.title("Figure 4 - MAE vs Leaf Size for Decision Tree and Random Tree")
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error * 100")
    plt.grid()
    plt.legend()
    plt.savefig("images/figure4.png")
    plt.clf()


def experiment33(train_x, train_y, test_x, test_y):
    max_leaf_size = 20
    outSample_MAE_bag = []
    outSample_MAE_dt = []

    for each_leaf_size in range(1, max_leaf_size + 1):
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={
                                "leaf_size": each_leaf_size}, bags=10, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)
        out_predY = learner.query(test_x)
        outSample_MAE = np.mean(
            np.abs((np.asarray(test_y) - np.asarray(out_predY))))
        outSample_MAE_bag.append(outSample_MAE * 100)

        learner = dtl.DTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        out_predY = learner.query(test_x)
        outSample_MAE = np.mean(
            np.abs((np.asarray(test_y) - np.asarray(out_predY))))
        outSample_MAE_dt.append(outSample_MAE * 100)

    x_indice = range(1, max_leaf_size + 1)
    plt.plot(x_indice, outSample_MAE_dt, label="Decision Tree")
    plt.plot(x_indice, outSample_MAE_bag, label="Bagged Random Tree")
    plt.title("Figure 5 - MAE vs Leaf Size for DT and 10 Bags RT")
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error * 100")
    plt.grid()
    plt.legend()
    plt.savefig("images/figure5.png")
    plt.clf()


def experiment34(train_x, train_y):
    max_trainning_size = train_x.shape[0]
    runTime_dt = []
    runTime_bag = []

    for training_size in range(200, max_trainning_size + 1, 200):
        learner = dtl.DTLearner(leaf_size=1, verbose=False)
        start = time.time()
        learner.add_evidence(train_x[:training_size], train_y[:training_size])
        end = time.time()
        running_time = end - start
        runTime_dt.append(running_time)

        learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={
                                "leaf_size": 1}, bags=10, boost=False, verbose=False)
        start = time.time()
        learner.add_evidence(train_x[:training_size], train_y[:training_size])
        end = time.time()
        running_time = end - start
        runTime_bag.append(running_time)

    x_indice = range(200, max_trainning_size + 1, 200)
    plt.plot(x_indice, runTime_dt, label="Decision Tree")
    plt.plot(x_indice, runTime_bag, label="10 Bags Random Tree")
    plt.title("Figure 6 - Training Time vs Size for DT and 10 Bags RT")
    plt.xlabel("Trainning Sizes")
    plt.ylabel("Trainning Time/s")
    plt.grid()
    plt.legend()
    plt.savefig("images/figure6.png")
    plt.clf()


def author():
    return "dpanirwala3"


def gtid():
    return 903262441


if __name__ == "__main__":
    choose = 2
    data_shuffle = True
    np.random.seed(gtid())
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(",")[1:]))
         for s in inf.readlines()[1:]]
    )
    inf.close()
    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:, 1:]
    data = data.astype('float')
    if data_shuffle:
        np.random.shuffle(data)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")

    # Istanbul.csv
    experiment1(train_x, train_y, test_x, test_y)
    experiment2(train_x, train_y, test_x, test_y)
    experiment32(train_x, train_y, test_x, test_y)
    experiment33(train_x, train_y, test_x, test_y)

    inf = open("Data/winequality-white.csv")
    data = np.array(
        [list(map(str, s.strip().split(",")[1:]))
         for s in inf.readlines()[1:]]
    )
    inf.close()
    data = data.astype('float')
    if data_shuffle:
        np.random.shuffle(data)
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]

    # winequality-white.csv
    experiment31(train_x, train_y)
    experiment34(train_x, train_y)
