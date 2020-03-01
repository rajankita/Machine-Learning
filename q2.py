import pandas as pd
import numpy as np
import time
from cvxopt import solvers, matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn.metrics import pairwise_distances, euclidean_distances
from svmutil import *
from itertools import combinations
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix

solvers.options['show_progress'] = False


def load_data(file):
    data = pd.read_csv(file, header=None).values
    X = data[:, 0:-1]
    Y = data[:, -1]

    # scaling - map from [0,255] to [0,1]
    X = X / X.max()

    return X, Y


def filter_data(X, Y, l0, l1):

    # Filter data for binary classification: Classes l0 and l1
    ind = np.where((Y == l0) | (Y == l1))[0]
    X = X[ind, :]
    Y = Y[ind]

    # Assign classes: l0 becomes -1, l1 becomes +1
    Y[Y == l0] = -1
    Y[Y == l1] = 1

    return X, Y


class my_svm_linear:
    """
    Class for SVM
    """
    def __init__(self, x, y, C):
        self.C = C

        # Train SVM
        self.w, self.b, self.x_sup = self.train(x, y)

    def train(self, x, y):
        M = x.shape[0]
        # compute kernel
        K = np.dot(x, x.T)

        # input parameters to solver
        P = matrix(np.dot(y, y.T) * K)
        q = matrix(-np.ones((M, 1)))
        A = matrix(y.T, tc='d')
        b = matrix(0.0)
        G = matrix(np.vstack([np.eye(M), (-1) * np.eye(M)]))
        h = matrix(np.vstack([self.C * np.ones((M, 1)), np.zeros((M, 1))]))

        # Optimization
        solver = solvers.qp(P, q, G, h, A, b)

        # Output parameters: alphas
        alphas = np.array(solver['x'])

        # Support vectors (alpha > 0)
        sv_ind = np.where(alphas > 1e-5)[0]
        sv_x = x[sv_ind, :]

        # Compute w
        w = np.sum(alphas * x * y, axis=0)

        sv_margin = np.where((alphas > 1e-5) & (alphas < 1 - 1e-5))[0]
        wb_margin = y[sv_margin, 0] - x[sv_margin, :] @ w.T
        wb = np.mean(wb_margin)

        return w, wb, sv_x

    def predict(self, x_test):
        # Prediction
        Y_pred = x_test @ self.w.T + self.b
        Y_pred[Y_pred > 0] = 1
        Y_pred[Y_pred < 0] = -1
        return Y_pred


class my_svm:
    """
    Class for SVM
    """
    def __init__(self, x, y, C, gamma):
        self.C = C
        self.gamma = gamma

        # Train SVM
        self.b, self.x_sup, self.y_sup, self.alpha_sup = self.train(x, y)

    def train(self, x, y):
        # compute kernel
        M = x.shape[0]
        x_dist = pairwise_distances(x, x, metric='l2') ** 2
        K = np.exp(-self.gamma * x_dist)

        # input parameters to solver
        P = matrix(np.dot(y, y.T) * K)
        q = matrix(-np.ones((M, 1)))
        A = matrix(y.T, tc='d')
        b = matrix(0.0)
        G = matrix(np.vstack([np.eye(M), (-1) * np.eye(M)]))
        h = matrix(np.vstack([self.C * np.ones((M, 1)), np.zeros((M, 1))]))

        # Optimization
        solver = solvers.qp(P, q, G, h, A, b)

        # Output parameters: alphas
        alphas = np.array(solver['x'])

        # Support vectors (alpha > 0)
        sv_ind = np.where(alphas > 1e-5)[0]

        sv_margin = np.where((alphas > 1e-5) & (alphas < 1 - 1e-5))[0]
        wb_margin = []
        for i in sv_margin:
            b = y[i]
            for j in sv_ind:
                k = np.exp(-self.gamma * (x[i] - x[j]).T @ (x[i] - x[j]))
                b = b - alphas[j] * y[j] * k
            wb_margin.append(b)
        wb = np.mean(wb_margin)

        return wb, x[sv_ind], y[sv_ind], alphas[sv_ind]

    def predict(self, x_test):
        # Prediction
        y_pred = np.empty(x_test.shape[0])
        for n, xx in enumerate(x_test):
            dist = np.sqrt(np.sum((self.x_sup - xx) ** 2, axis=1)) ** 2
            k = np.exp(-gamma * dist)
            yy = self.b + np.sum(k[:, None] * self.alpha_sup * self.y_sup)
            # yy = self.b
            # # dist = np.linalg.norm(x[sv_ind] - xx, axis=1)**2
            # # k = np.exp(-gamma*dist)
            # # yy1 = wb + np.sum(k[:, None] * alphas[sv_ind] * y[sv_ind])
            # for i in range(self.x_sup.shape[0]):
            #     kk = np.exp(-gamma * (self.x_sup[i] - xx).T @ (self.x_sup[i] - xx))
            #     yy += self.alpha_sup[i] * self.y_sup[i] * kk
            y_pred[n] = yy

        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = -1

        return y_pred


def binary_classification(part, label0, label1):

    # Filter training and test data
    X_train1, Y_train1 = filter_data(X_train, Y_train, label0, label1)
    X_test1, Y_test1 = filter_data(X_test, Y_test, label0, label1)

    if part == 'a':
        # Train Linear SVM using QP solver
        print('----------- Linear SVM (using cvxopt) --------------')
        start = time.time()
        m = my_svm_linear(X_train1, Y_train1[:, None], C)
        print('Training time = {}'.format(time.time() - start))

        # Prediction
        # Y_pred_train = m.predict(X_train)
        # acc = accuracy_score(y_true=Y_train, y_pred=Y_pred_train)
        # print('Training accuracy = {}'.format(acc))
        Y_pred_test = m.predict(X_test1)
        acc = accuracy_score(y_true=Y_test1, y_pred=Y_pred_test)
        print('Test set accuracy = {}'.format(acc))

    elif part == 'b':
        # Train Gaussian SVM using cvxopt
        print('----------- Gaussian SVM (using cvxopt) --------------')
        start = time.time()
        m = my_svm(X_train1, Y_train1[:, None], C, gamma)
        print('Training time = {}'.format(time.time() - start))
        Y_pred = m.predict(X_test1)
        acc = accuracy_score(y_true=Y_test1, y_pred=Y_pred)
        print('Test accuracy = {}'.format(acc))

    elif part == 'c':
        # Train Linear SVM using libsvm
        print('----------- Linear SVM (using libsvm) --------------')
        start = time.time()
        m = svm_train(Y_train1, X_train1, '-t 0 -c {}'.format(C))
        p_label, p_acc, p_val = svm_predict(Y_test1, X_test1, m)
        print('Training time = {}'.format(time.time() - start))

        # Train Gaussian SVM using libsvm
        print('----------- Linear SVM (using libsvm) --------------')
        start = time.time()
        m = svm_train(Y_train1, X_train1, '-t 2 -c {} -g {}'.format(C, gamma))
        p_label, p_acc, p_val = svm_predict(Y_test1, X_test1, m)
        print('Training time = {}'.format(time.time() - start))


def multiclass_classification(part):

    if part == 'a':
        # Train multi-class SVM using cvxopt one-vs-one
        print('--------------- Training -------------------')
        start = time.time()
        all_combinations = list(combinations(range(K), 2))
        # Train a binary classifier for each combination
        classifiers = {}
        for combn in all_combinations:
            print(combn)
            # Extract training data for these classes
            label0, label1 = combn
            X_train1, Y_train1 = filter_data(X_train, Y_train, label0, label1)

            # Train binary SVM
            m = my_svm(X_train1, Y_train1[:, None], C, gamma)
            classifiers[combn] = m
        print('Training time = {}'.format(time.time() - start))

        print('-------------- Prediction on test set ------------------')
        start = time.time()
        # Collect class votes from each binary classifier
        class_votes = np.zeros((M_test, K))
        class_scores = np.zeros((M_test, K))
        for combn in all_combinations:
            print(combn)
            # Obtain the classifier for this combination
            label0, label1 = combn
            svm_k = classifiers[combn]

            # Predict using this classifier
            label_pred0 = svm_k.predict(X_test)

            # Map the prediction to actual class labels
            label_pred0 = np.asarray(label_pred0).astype(int)
            label_pred = label_pred0.copy()
            label_pred[label_pred0 == -1] = label0
            label_pred[label_pred0 == 1] = label1

            # Update votes for classes
            class_votes[list(range(M_test)), label_pred] += 1
        print('Prediction time = {}'.format(time.time() - start))

        # Predict class with maximum number of votes
        Y_test_pred = np.argmax(class_votes, axis=1)
        acc = accuracy_score(y_true=Y_test, y_pred=Y_test_pred)
        # print(classification_report(Y_test, Y_test_pred))
        print(metrics.confusion_matrix(Y_test, Y_test_pred))
        print('Accuracy = {}'.format(acc))

    elif part == 'b':
        # Train Using libsvm
        start = time.time()
        m = svm_train(Y_train, X_train, '-t 2 -c {} -g {}'.format(C, gamma))
        print('Training time = {}'.format(time.time() - start))
        p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)

    elif part == 'c':
        # Confusion matrix
        start = time.time()
        m = svm_train(Y_train, X_train, '-t 2 -c {} -g {}'.format(C, gamma))
        p_label, p_acc, p_val = svm_predict(Y_train, X_train, m)
        print('Training time = {}'.format(time.time() - start))
        print('Confusion matrix:')
        print(confusion_matrix(y_true=Y_test, y_pred=p_label))

    elif part == 'd':
        # Cross-validation
        # Select cross-validation set
        M_val = round(M / 10)
        val_idx = np.random.permutation(M)[:M_val]
        new_train_idx = [i for i in range(M) if i not in val_idx]
        X_val = X_train[val_idx]
        Y_val = Y_train[val_idx]
        X_train_small = X_train[new_train_idx]
        Y_train_small = Y_train[new_train_idx]

        # Train for different values of C
        C_arr = [1e-5, 1e-3, 1, 5, 10]
        val_acc, test_acc = [], []
        for C_temp in C_arr:
            print('C = {}'.format(C_temp))
            m = svm_train(Y_train_small, X_train_small, '-t 2 -c {} -g {} -q'.format(C_temp, gamma))
            _, p_acc0, _ = svm_predict(Y_val, X_val, m)
            val_acc.append(p_acc0[0])
            _, p_acc1, _ = svm_predict(Y_test, X_test, m)
            test_acc.append(p_acc1[0])

        plt.semilogx(C_arr, val_acc, label='Validation accuracy')
        plt.semilogx(C_arr, test_acc, label='Test accuracy')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    # train_file = 'mnist/train.csv'
    # test_file = 'mnist/test.csv'
    C = 1.0
    gamma = 0.05

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    bin_mul = int(sys.argv[3])
    part_num = sys.argv[4]
    label0 = 0
    label1 = 1

    # Load and pre-process data
    X_train, Y_train = load_data(train_file)
    X_test, Y_test = load_data(test_file)
    # X_train = X_train[:1000]
    # Y_train = Y_train[:1000]
    # X_test = X_test[:1000]
    # Y_test = Y_test[:1000]

    K = len(np.unique(Y_train))  # no. of classes
    M = len(Y_train)  # no. of training examples
    M_test = len(Y_test)  # no. of test data points

    if bin_mul == 0:
        binary_classification(part_num, label0, label1)
    elif bin_mul == 1:
        multiclass_classification(part_num)
    else:
        raise ValueError




