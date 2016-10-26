"""
An example run of scikit-learn's SVM algorithm on the RCV1 dataset.

Usage:
python svm_example.py path_to_rcv1_train_file path_to_rcv1_test_file [binary] 2> example.info 1> example.result
"""

import sys
import csv
import struct
import logging
import itertools as it
import collections as cl

import numpy as np
import pandas as pd
from sklearn import svm
from scipy.sparse import csr_matrix


NUM_WORDS = 47236
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def read_binary_file(filename):
    """Read example data points from an RCV1 binary file."""
    # members of structs: int, int, double
    format = 'iid'
    struct_size = struct.calcsize(format)

    with open(filename, 'rb') as rcv1:
        num_examples, = struct.unpack('i', rcv1.read(struct.calcsize('i')))
        for _ in xrange(num_examples):
            yield struct.unpack(format, rcv1.read(struct_size))


def read_text_file(filename, delimiter='\t'):
    """Read data from an RCV1 text file."""
    with open(filename, 'r') as file_in:
        for document, word, frequency in csv.reader(file_in, delimiter=delimiter):
            yield [int(document), int(word), np.float64(frequency)]


def process_input(rows, chunk_size=1000):
    """Assemble the data from an RCV1 file into feature vectors and labels."""

    feature_vectors, labels = [], []
    Row = cl.namedtuple('Row', ['document', 'word', 'frequency'])
    row = Row(*next(rows))

    # To address memory issues
    row_indices, col_indices = [], []
    values = []

    this_document = row.document
    features = np.zeros(NUM_WORDS)
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
        features = np.zeros(NUM_WORDS)
        label = row.frequency  # Misnomer, this is the class label for this row

    # Debugging
    logging.debug("Greatest row index: %s", row_indices[-1])
    logging.debug("len(values): %s", len(values))
    logging.debug("len(row_indices): %s", len(row_indices))
    logging.debug("len(col_indices): %s", len(col_indices))

    feature_vectors = csr_matrix((values, (row_indices, col_indices)))
    return feature_vectors, labels


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


def main(train_filename, test_filename, binary=False):
    """Run scikit-learn's SVM algorithm on the RCV1 dataset; print error."""

    logging.info("Reading train file")
    read_file = read_binary_file if binary else read_text_file
    train_rows = read_file(train_filename)

    logging.info("Processing train input")
    train_feature_vectors, train_labels = process_input(train_rows)
    logging.info("Number of training examples: %s", train_feature_vectors.shape[0])

    logging.info("Fitting SVM")
    classifier = svm.LinearSVC()
    classifier.fit(train_feature_vectors, train_labels)

    # Wouldn't normally do this, but memory-constrained
    del train_feature_vectors
    del train_labels

    logging.info("Reading test file")
    test_rows = read_file(test_filename)

    logging.info("Processing test input")
    test_feature_vectors, test_labels = process_input(test_rows)
    logging.info("Number of test examples: %s", test_feature_vectors.shape[0])

    predictions = classifier.predict(test_feature_vectors)
    incorrect = sum(prediction != label for prediction, label in it.izip(predictions, test_labels))
    error = float(incorrect) / test_feature_vectors.shape[0]
    print error


if __name__ == '__main__':
    main(*sys.argv[1:])



