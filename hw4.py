import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None # weights

        # iterations history
        self.Js = []
        self.thetas = []

        self.hypothesis = []
    
    def calc_hypothesis(self, X, theta):
      """
      Calculates the hypothesis function h(x) = theta^T * x.

      Input:
      - X: Input data (m instances over n features).
      - theta: The parameters (weights) of the model being learned.
      """
      denominator = 1 + np.exp(-X.dot(theta))
      self.hypothesis = 1 / denominator  
  
    def compute_cost(self, m, y):
      """
      Calculates the cost function J(theta) = -1/m * sum(y * log(h_theta) + (1 - y) * log(1 - h_theta)).
      Input:
      - m: n_examples.
      - y: True labels (m instances).
      - h_theta: The hypothesis function.
      Returns:
      - j_theta: The cost function for the given input.
      """
      j_theta = (-1 / m) * np.sum((np.log(self.hypothesis)  * y) + (np.log(1 - self.hypothesis) * (1 - y)))
      return j_theta
    
    def calc_new_theta(self, X, y, theta):
      """
      Calculates the new theta vector after one iteration of gradient descent.
      Input:
      - X: Input data (m instances over n features).
      - y: True labels (m instances).
      - theta: The parameters (weights) of the model being learned.
      - h_theta: The hypothesis function.
      Returns:
      - new_theta: The new theta vector after one iteration of gradient descent.
      """
      
      
      theta -= (self.eta * (self.hypothesis - y)).dot(X) 
      return theta

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # without preprocessing we're getting better results
        # X, y = preprocess(X, y)
        X = apply_bias_trick(X)
        # set random seed
        np.random.seed(self.random_state)
        # initialize theta
        theta = np.random.random(X.shape[1])
        # gradient descent
        for iteration in range(self.n_iter):
            self.calc_hypothesis(X, theta) # self.hyposis is a vector of shape 1 * n_examples of continuous values, 0 - 1
            self.Js.append(self.compute_cost(X.shape[0], y))
            if iteration > 0 and self.Js[-2] - self.Js[-1] < self.eps: 
                break
            self.thetas.append(theta)
            theta = self.calc_new_theta(X, y, theta)
        
        self.theta = theta

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = apply_bias_trick(X)
        preds = []
        self.calc_hypothesis(X, self.theta)
        for example in self.hypothesis:
            if  example > 0.5:
                preds.append(1)
            else:
                preds.append(0)

        return np.asarray(preds)

def apply_bias_trick(X):
      """
      Applies the bias trick to the input data.

      Input:
      - X: Input data (m instances over n features).

      Returns:
      - X: Input data with an additional column of ones in the
          zeroth position (m instances over n+1 features).
      """
      return np.c_[np.ones(X.shape[0]),X]

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - np.mean(X, axis = 0)) / (np.amax(X, axis = 0) - np.amin(X, axis = 0))
    y = (y - np.mean(y, axis = 0)) / (np.amax(y, axis = 0) - np.amin(y, axis = 0))
    return X, y

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    cv_accuracy = 0
    sum_all_folds_accuracy = 0

    # set random seed
    np.random.seed(random_state)

    # shuffle data
    shuffled_indices = np.random.permutation(X.shape[0])
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # create folds
    X_folds = np.array_split(X, folds)
    y_folds = np.array_split(y, folds)  

    # train on each fold
    for fold_index in range(folds):
        # concataneate the other folds
        training_data = np.concatenate(X_folds[:fold_index] + X_folds[fold_index + 1:])
        training_labels = np.concatenate(y_folds[:fold_index] + y_folds[fold_index + 1:])
        # train on current training data
        algo.fit(training_data, training_labels)
        # test on current fold.
        preds = algo.predict(X_folds[fold_index])
        # calculate accuracy
        sum_all_folds_accuracy += np.sum(preds == y_folds[fold_index]) / y_folds[fold_index].shape[0] ###
    
    # return average accuracy
    cv_accuracy = sum_all_folds_accuracy / folds
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = np.exp(-0.5 * np.power(((data - mu) / sigma), 2))
    p = coefficient * exponent
    
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """
    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = []
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = []
    
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.weights = np.ones((self.k,)) / self.k
        self.mus = (np.random.rand(self.k)) * np.max(data)
        self.sigmas = np.ones((self.k,))  

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        self.responsibilities = np.zeros((len(data), self.k))
        
        for instance in range(len(data)):
          for distribution in range(self.k):
              likelihood = norm_pdf(data[instance], self.mus[distribution], self.sigmas[distribution])
              # Calculate the responsibility as the product of likelihood and weight
              self.responsibilities[instance, distribution] = self.weights[distribution] * likelihood
          # Normalize the responsibilities to sum up to 1
          self.responsibilities[instance] /= np.sum(self.responsibilities[instance])
        return self.responsibilities

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        # Update the weight
        self.weights = self.responsibilities.sum(0) / len(data)
        for distribution in range(self.k):  
            new_weight_N = self.weights[distribution]*len(data)  
            # Update mu
            self.mus[distribution] = sum(data[instance] * self.responsibilities[instance][distribution] for instance in range(len(data))) / new_weight_N
            # Update sigma
            self.sigmas[distribution] = np.sqrt(sum(self.responsibilities[instance][distribution] * (data[instance]-self.mus[distribution])**2 for instance in range(len(data))) / new_weight_N)

    def compute_cost_log_likelihood(self, data):
        """
        Calculate log-likelihood of the EM algorithm
        """
        log_likelihood = 0
        for distribution in range(self.k):
            # Calculate the likelihood 
            likelihood = norm_pdf(data, self.mus[distribution], self.sigmas[distribution])
            # Calculate the log likelihood as the log of the product of likelihoods
            log_likelihood += np.sum(-np.log2(self.weights[distribution] * likelihood))
        # Return the log likelihood
        return log_likelihood

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
  
        for iteration in range(self.n_iter):
            # E-step
            self.expectation(data)
            # M-step
            self.maximization(data)
            # calc cost
            cost = self.compute_cost_log_likelihood(data)
            self.costs.append(cost)
            if iteration > 0 and self.costs[-2] - self.costs[-1] < self.eps:
                break
    
    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = np.zeros_like(data, dtype=np.float64)
    for i in range(len(weights)):
        pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """
    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.likelihoods = None
        self.numberOfClasses = None
        self.class_values = None
        self.EMs = None
        
    def get_prior(self, y):
        self.prior = np.zeros(self.numberOfClasses)
        for class_i in range(self.numberOfClasses):
            self.prior[class_i] = np.sum(y == self.class_values[class_i]) / y.shape[0]

    def get_likelihood(self, instance, feature, class_i):
        likelihood = gmm_pdf(instance, self.EMs[class_i][feature].weights, self.EMs[class_i][feature].mus, self.EMs[class_i][feature].sigmas)

        return likelihood
    
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        # supposed to leave with a gmm for each class
        self.numberOfInstances = X.shape[0]
        self.features = X.shape[1]
        self.class_values = np.unique(y)
        self.numberOfClasses = len(self.class_values)
        self.EMs = np.empty((self.numberOfClasses, self.features), dtype=EM)

        self.get_prior(y)

        # build a gmm for each class and feature
        for class_i in range(self.numberOfClasses):
            for feature in range(self.features):
                data_with_specific_class_and_feature = X[y == self.class_values[class_i]][:, feature]
                gmm_class_feature_obj = EM(self.k)
                gmm_class_feature_obj.fit(data_with_specific_class_and_feature)
                self.EMs[class_i][feature] = gmm_class_feature_obj

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = np.empty(X.shape[0], dtype=np.int64)
        posterior = 0

        # make sure axis are in sync
        self.numberOfInstances = X.shape[0]
        self.features = X.shape[1]

        # classify each instance
        for instance in range(self.numberOfInstances):
            max_posterior = -np.inf
            max_class = None

            # calculate the posterior for each class and choose the max
            for class_i in range(self.numberOfClasses):
                posterior = self.prior[class_i]
                # use the naive base assumption to calculate the likelihood
                for feature in range(self.features):
                  posterior *= self.get_likelihood(X[instance, feature], feature, self.class_values[class_i])
                # update max
                if posterior > max_posterior:
                    max_posterior = posterior
                    max_class = self.class_values[class_i]
            # classify the instance and move to the next.
            preds[instance] = max_class
        
        return np.asarray(preds)

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 
    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    lor = LogisticRegressionGD(eta=best_eta,eps=best_eps)
    lor.fit(x_train, y_train)
    lor_train_acc = calc_acc(lor.predict(x_train), y_train)
    lor_test_acc = calc_acc(lor.predict(x_test), y_test)

    bayes = NaiveBayesGaussian(k=k) 
    bayes.fit(x_train, y_train)
    bayes_train_acc = calc_acc(bayes.predict(x_train), y_train)
    bayes_test_acc = calc_acc(bayes.predict(x_test), y_test)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}, lor, bayes

def calc_acc(pred, y):
    return np.sum(pred == y)/y.shape[0]

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    
    # Dataset A: Independent features suitable for Naive Bayes
    
    # Generate means for each class
    mean_1 = [8, 8, 8]
    mean_2 = [-8, -8, -8]
    mean_3 = [16, 16, 16]
    mean_4 = [-16, -16, -16]
    # Generate samples from independent Gaussian distributions
    dataset_a_features = np.vstack([multivariate_normal.rvs(mean_1, np.diag([1, 1, 1]), size=500),
                                    multivariate_normal.rvs(mean_2, np.diag([0.5, 0.5, 0.5]), size=500),
                                    multivariate_normal.rvs(mean_3, np.diag([1, 1, 1]), size=500),
                                    multivariate_normal.rvs(mean_4, np.diag([0.5, 0.5, 0.5]), size=500)])
    dataset_a_labels = np.hstack([np.zeros(1000), np.ones(1000)])
    
    # Dataset B: Dependent features suitable for Logistic Regression
    
    # Generate means for each class
    mean_1 = [0, 0, 0]
    mean_2 = [0, 5, 10]
    
    # Generate covariance matrix with dependency
    cov = np.array([[1, 0.5, 0.3],
                    [0.5, 1, 0.2],
                    [0.3, 0.2, 1]])
    
    # Generate samples from multivariate Gaussian distributions with dependency
    dataset_b_features = np.vstack([multivariate_normal.rvs(mean_1, cov, size=1000),
                                    multivariate_normal.rvs(mean_2, 0.5 * cov, size=1000)])
    dataset_b_labels = np.hstack([np.zeros(1000), np.ones(1000)])
    
    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }, dataset_a_features, dataset_a_labels, dataset_b_features, dataset_b_labels