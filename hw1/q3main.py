import pandas
import numpy
import copy
from numpy import inf
numpy.seterr(all="ignore")

"""
CS464 Introduction to Machine Learning
Homework 1

Ece Kahraman
21801879
"""


def multinomial_naive_bayes(spam, normal, dirichlet):
    """
    Multinomial Naive Bayes Classifier by using maximum likelihood estimation.
    Implemented with Dirichlet distribution, but the Dirichlet prior is set to 0
    for the question 2.2.

    :param spam: array of spam mails
    :param normal: array of normal (ham) mails
    :param dirichlet: an integer for Dirichlet distribution
    :return: confusion matrix
    """

    # Prior likelihood of spam mails and normal mails
    prior_spam = numpy.log(len(spam)/len(x_train))
    prior_normal = numpy.log(len(normal)/len(x_train))

    t_spam = spam.sum().to_numpy()
    t_normal = normal.sum().to_numpy()
    theta_spam = numpy.log((t_spam + dirichlet) / (numpy.sum(t_spam) + dirichlet * len(spam)))
    theta_normal = numpy.log((t_normal + dirichlet) / (numpy.sum(t_normal) + dirichlet * len(spam)))

    # Set negative infinity to -10^12 for easier calculations
    theta_spam[theta_spam == -inf] = -(10 ** 12)
    theta_normal[theta_normal == -inf] = -(10 ** 12)

    # First, make sure the array we are processing is a numpy array.
    # Then, calculate the Multinomial NB formula for each label, we have only two labels here.
    # Lastly, argmax() chooses the label that maximizes one of the likelihoods.
    p_spam_doc = numpy.sum(x_test.to_numpy() * theta_spam, axis=1) + prior_spam
    p_normal_doc = numpy.sum(x_test.to_numpy() * theta_normal, axis=1) + prior_normal
    results = numpy.argmax([p_normal_doc, p_spam_doc], axis=0)

    # Create the confusion matrix and fill it accordingly
    # 0, 0 -> true negative
    # 0, 1 -> false negative
    # 1, 0 -> false positive
    # 1, 1 -> true positive
    confused_matrix = numpy.zeros((2, 2))
    for x, y in numpy.ndindex((2, 2)):
        confused_matrix[x, y] = numpy.sum((results == x) & (y_test == y))

    return confused_matrix


def bernoulli_naive_bayes(spam, normal):
    """
    Bernoulli Naive Bayes Classifier by using maximum likelihood estimation.

    :param spam: array of spam mails
    :param normal: array of normal (ham) mails
    :return: confusion matrix
    """
    # Prior likelihood of spam mails and normal mails
    prior_spam = numpy.log(len(spam) / len(x_train_binary))
    prior_normal = numpy.log(len(normal) / len(x_train_binary))

    s_spam = spam.sum().to_numpy()
    s_normal = normal.sum().to_numpy()
    theta_spam = s_spam / len(spam)
    theta_normal = s_normal / len(normal)

    # Set negative infinity to -10^12 for easier calculations
    theta_spam[theta_spam == -inf] = -(10 ** 12)
    theta_normal[theta_normal == -inf] = -(10 ** 12)

    # First, make sure the array we are processing is a numpy array.
    # Then, calculate the Bernoulli NB formula for each label, we have only two labels here.
    # Lastly, argmax() chooses the label that maximizes one of the likelihoods.
    doc = x_test_binary.to_numpy()
    p_spam_doc = numpy.sum(numpy.log(doc * theta_spam + (1 - doc) * (1 - theta_spam)), axis=1) + prior_spam
    p_normal_doc = numpy.sum(numpy.log(doc * theta_normal + (1 - doc) * (1 - theta_normal)), axis=1) + prior_normal
    results = numpy.argmax([p_normal_doc, p_spam_doc], axis=0)

    # Create the confusion matrix and fill it accordingly
    # 0, 0 -> true negative
    # 0, 1 -> false negative
    # 1, 0 -> false positive
    # 1, 1 -> true positive
    confused_matrix = numpy.zeros((2, 2))
    for x, y in numpy.ndindex((2, 2)):
        confused_matrix[x, y] = numpy.sum((results == x) & (y_test == y))

    return confused_matrix


# Open every data files before everything, and specify "Prediction" column for y files.
x_train = pandas.read_csv("x_train.csv")
y_train = pandas.read_csv("y_train.csv")["Prediction"]
x_test = pandas.read_csv("x_test.csv")
y_test = pandas.read_csv("y_test.csv")["Prediction"]

# Divide the x_train file by spam and normal
x_spam = x_train[y_train == 1]
x_normal = x_train[y_train == 0]

print("2.2 Multinomial Naive Bayes Classifier")
confusion_matrix_1 = multinomial_naive_bayes(x_spam, x_normal, 0)
true_negatives = confusion_matrix_1[0, 0]
true_positives = confusion_matrix_1[1, 1]
accuracy_1 = (true_positives + true_negatives) / confusion_matrix_1.sum()

false_negatives = confusion_matrix_1[0, 1]
false_positives = confusion_matrix_1[1, 0]
wrong_predictions_1 = false_positives + false_negatives

print("Confusion Matrix:\n ", pandas.DataFrame(confusion_matrix_1, columns=['Actually False', 'Actually True'], index=['Predicted False', 'Predicted True'], dtype=int))
print("Accuracy: %.3f\nNumber of wrong predictions: %d" % (accuracy_1, wrong_predictions_1))

print("\n2.3 Multinomial Naive Bayes Classifier with alpha = 5")
confusion_matrix_2 = multinomial_naive_bayes(x_spam, x_normal, 5)
true_negatives = confusion_matrix_2[0, 0]
true_positives = confusion_matrix_2[1, 1]
accuracy_2 = (true_positives + true_negatives) / confusion_matrix_2.sum()

false_negatives = confusion_matrix_2[0, 1]
false_positives = confusion_matrix_2[1, 0]
wrong_predictions_2 = false_positives + false_negatives

print("Confusion Matrix:\n ", pandas.DataFrame(confusion_matrix_2, columns=['Actually False', 'Actually True'], index=['Predicted False', 'Predicted True'], dtype=int))
print("Accuracy: %.3f\nNumber of wrong predictions: %d" % (accuracy_2, wrong_predictions_2))

# Create binary representations of x_test and x_train before calling bernoulli_naive_bayes
# and divide binary train set by spam and normal to pass to the function.
x_test_binary = copy.deepcopy(x_test)
x_test_binary[x_test_binary > 0] = 1
x_train_binary = copy.deepcopy(x_train)
x_train_binary[x_train_binary > 0] = 1

x_spam = x_train_binary[y_train == 1]
x_normal = x_train_binary[y_train == 0]

print("\n2.4 Bernoulli Naive Bayes Classifier")
confusion_matrix_3 = bernoulli_naive_bayes(x_spam, x_normal)
true_negatives = confusion_matrix_3[0, 0]
true_positives = confusion_matrix_3[1, 1]
accuracy_3 = (true_positives + true_negatives) / confusion_matrix_3.sum()

false_negatives = confusion_matrix_3[0, 1]
false_positives = confusion_matrix_3[1, 0]
wrong_predictions_3 = false_positives + false_negatives

print("Confusion Matrix:\n ", pandas.DataFrame(confusion_matrix_3, columns=['Actually False', 'Actually True'], index=['Predicted False', 'Predicted True'], dtype=int))
print("Accuracy: %.3f\nNumber of wrong predictions: %d" % (accuracy_3, wrong_predictions_3))
