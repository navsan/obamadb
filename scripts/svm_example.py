"""
An example run of scikit-learn's SVM algorithm on the RCV1 dataset.

Usage:
python svm_example.py path_to_rcv1_train_file path_to_rcv1_test_file [binary]
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
            yield [np.int32(document), np.int32(word), np.float64(frequency)]


def process_input(rows, chunk_size=1000):
    """Assemble the data from an RCV1 file into feature vectors and labels."""

    feature_vectors, labels = [], []
    Row = cl.namedtuple('Row', ['document', 'word', 'frequency'])
    row = Row(*next(rows))

    # To address memory issues, accumulate a list of SparseDataFrames
    feature_vectors = []

    this_document = row.document
    features = np.zeros(NUM_WORDS)
    label = row.frequency

    print "Entering iteration"
    for row in it.imap(Row._make, rows):

        # Get features of sparse vector
        try:
            while row.document == this_document:
                features[row.word] = row.frequency
                row = Row(*next(rows))

        except StopIteration:
            pass

        feature_vectors.append(pd.SparseSeries(features, fill_value=np.float64(0.0)))
        labels.append(label)

        # Debugging
        if len(feature_vectors) % chunk_size == 0:
            print len(feature_vectors)

        # Grow the SparseDataFrame; retains SparseDataFrame type
        """
        if len(feature_vectors) % chunk_size == 0:
            print "Adding to frames"
            frames.append(pd.SparseDataFrame(feature_vectors))
            feature_vectors = []
        """

        # Setup for next ingestion of a frequency vector
        this_document = row.document
        features = np.zeros(NUM_WORDS)
        label = row.frequency  # Misnomer, this is the class label for this row

    # Flush remaining feature vectors to the frame
    # frames.append(pd.DataFrame(feature_vectors).to_sparse(fill_value=np.float64(0.0)))

    print "Creating SparseDataFrame, numpy array"
    return pd.SparseDataFrame(feature_vectors), np.array(labels)


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
    logging.info("Number of training examples: %s", len(train_feature_vectors))

    logging.info("Fitting SVM")
    classifier = svm.LinearSVC()
    classifier.fit(csr_matrix(train_feature_vectors), train_labels)

    # Wouldn't normally do this, but memory-constrained
    del train_feature_vectors
    del train_labels

    logging.info("Reading test file")
    test_rows = read_file(test_filename)

    logging.info("Processing test input")
    test_feature_vectors, test_labels = process_input(test_rows)
    logging.info("Number of test examples: %s", len(test_feature_vectors))

    predictions = classifier.predict(test_feature_vectors)
    incorrect = sum(prediction != label for prediction, label in it.izip(predictions, test_labels))
    error = float(incorrect) / len(test_feature_vectors)
    print error


if __name__ == '__main__':
    main(*sys.argv[1:])



