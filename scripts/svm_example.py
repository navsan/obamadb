"""
An example run of scikit-learn's SVM algorithm on the RCV1 dataset.

Usage:
python svm_example.py path_to_rcv1_train_file path_to_rcv1_test_file [binary] 2> example.info 1> example.result
"""

import sys
import csv
import struct
import timeit
import cPickle
import logging
import itertools as it
import collections as cl

import numpy as np
# import pandas as pd
# from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error


NUM_WORDS = 47236
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def read_binary_file(filename):
    """Read example data points from an RCV1 binary file."""
    # members of structs: int, int, double
    format = 'iid'
    struct_size = struct.calcsize(format)

    with open(filename, 'rb') as rcv1:
        num_examples, = struct.unpack('i', rcv1.read(struct.calcsize('i')))
        # Debugging
        print "num_examples:", num_examples
        for _ in xrange(num_examples):
            yield struct.unpack(format, rcv1.read(struct_size))


def read_text_file(filename, delimiter='\t'):
    """Read data from an RCV1 text file."""
    with open(filename, 'r') as file_in:
        for document, word, frequency in csv.reader(file_in, delimiter=delimiter):
            yield [int(document), int(word), np.float64(frequency)]


def pickle_input(input, filename):
    """Save input to a pickled file."""
    with open(filename, 'w') as file_out:
        cPickle.dump(input, file_out, cPickle.HIGHEST_PROTOCOL)


def load_input(filename):
    """Load input from a pickled file."""
    with open(filename, 'r') as file_in:
        return cPickle.load(file_in)


def process_input(rows, chunk_size=1000):
    """Assemble the data from an RCV1 file into feature vectors and labels."""

    labels = []
    Row = cl.namedtuple('Row', ['document', 'word', 'frequency'])
    row = Row(*next(rows))

    # To address memory issues
    row_indices, col_indices = [], []
    values = []

    this_document = row.document
    label = row.frequency

    # Debugging
    logging.debug("Entering iteration")
    for i, row in enumerate(it.imap(Row._make, rows)):

        # Get features of sparse vector
        try:
            while row.document == this_document:
                row_indices.append(i)
                col_indices.append(row.word)
                values.append(row.frequency)
                row = Row(*next(rows))

        except StopIteration:
            pass

        labels.append(label)

        # Debugging
        if i % chunk_size == 0:
            logging.debug(i)

        # Setup for next ingestion of a frequency vector
        this_document = row.document
        label = row.frequency  # Misnomer, this is the class label for this row

    # Debugging
    logging.debug("Greatest row index: %s", row_indices[-1])
    logging.debug("len(values): %s", len(values))
    logging.debug("len(row_indices): %s", len(row_indices))
    logging.debug("len(col_indices): %s", len(col_indices))

    return {
        'row_indices': row_indices,
        'col_indices': col_indices,
        'values': values,
        'labels': labels
    }


def feature_vector_generator(rows):
    """Generate feature vectors and their labels."""
    # For debugging purposes; can check it out in the prompt
    Row = cl.namedtuple('Row', ['document', 'word', 'frequency'])
    row = Row(*next(rows))

    this_document = row.document
    features = np.zeros(NUM_WORDS)
    label = row.frequency

    for row in it.imap(Row._make, rows):

        # Get features of sparse vector
        try:
            # Hm, something wrong here; complaining about an AttributeError on a list 
            # (says row is a list...)
            while row.document == this_document:
                features[row.word] = row.frequency
                row = Row(*next(rows))

        except StopIteration:
            pass

        yield this_document, features, label

        # Setup for next ingestion of a frequency vector
        this_document = row.document
        features = np.zeros(NUM_WORDS)
        label = row.frequency  # Misnomer, this is the class label for this row 


def main(train_filename, test_filename, number=3, epochs=20,
         init_learning_rate=0.1, binary=False):
    """Run scikit-learn's SVM algorithm on the RCV1 dataset; print error."""
    # Convert arguments
    number = int(number)
    epochs = int(epochs)
    init_learning_rate = float(init_learning_rate)

    logging.info("Reading train file")
    read_file = read_binary_file if binary else read_text_file
    train_input = process_input(read_file(train_filename))

    # Unpack the training input
    values = train_input['values']
    row_indices = train_input['row_indices']
    col_indices = train_input['col_indices']
    train_labels = train_input['labels']

    train_feature_vectors = csr_matrix((values, (row_indices, col_indices)))
    logging.info("Number of training examples: %s", train_feature_vectors.shape[0])

    logging.info("Fitting SVM")
    # classifier = svm.LinearSVC()
    kwargs = {
        'loss': 'hinge',
        'penalty': 'l2',
        'n_iter': epochs,
        'eta0': init_learning_rate
    }
    classifier = SGDClassifier(**kwargs)

    # Time model-building process
    def model():
        classifier.fit(train_feature_vectors, train_labels)
    print "Average model runtime:", timeit.timeit(model, number=number)

    # Do it for real in the script
    classifier.fit(train_feature_vectors, train_labels)

    # Wouldn't normally do this, but memory-constrained
    del train_input

    logging.info("Reading test file")
    test_input = process_input(read_file(test_filename))

    # Unpack the test input
    values = test_input['values']
    row_indices = test_input['row_indices']
    col_indices = test_input['col_indices']
    test_labels = test_input['labels']

    test_feature_vectors = csr_matrix((values, (row_indices, col_indices)))
    logging.info("Number of test examples: %s", test_feature_vectors.shape[0])

    predictions = classifier.predict(test_feature_vectors)
    rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    print "RMSE:", rmse


def pickled_main(train_filename, test_filename, number=3):
    """Run scikit-learn's SVM algorithm on the RCV1 dataset; print error."""

    logging.info("Reading train file")
    train_input = load_input(train_filename)

    # Unpack the training input
    values = train_input['values']
    row_indices = train_input['row_indices']
    col_indices = train_input['col_indices']
    train_labels = train_input['labels']

    train_feature_vectors = csr_matrix((values, (row_indices, col_indices)))
    logging.info("Number of training examples: %s", train_feature_vectors.shape[0])

    logging.info("Fitting SVM")
    classifier = svm.LinearSVC()

    # Time model-building process
    def model():
        classifier.fit(train_feature_vectors, train_labels)
    print "Average model runtime:", timeit.timeit(model, number=number)

    # Do it for real in the script
    classifier.fit(train_feature_vectors, train_labels)

    # Wouldn't normally do this, but memory-constrained
    del train_input

    logging.info("Reading test file")
    test_input = load_input(test_filename)

    # Unpack the test input
    values = test_input['values']
    row_indices = test_input['row_indices']
    col_indices = test_input['col_indices']
    test_labels = test_input['labels']

    test_feature_vectors = csr_matrix((values, (row_indices, col_indices)))
    logging.info("Number of test examples: %s", test_feature_vectors.shape[0])

    predictions = classifier.predict(test_feature_vectors)
    rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    print "RMSE:", rmse

    # incorrect = sum(prediction != label for prediction, label in it.izip(predictions, test_labels))
    # error = float(incorrect) / test_feature_vectors.shape[0]
    # print error


if __name__ == '__main__':
    main(*sys.argv[1:])



