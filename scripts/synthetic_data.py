"""
Generate synthetic datasets with a variable number of examples, features, and sparsity.
"""

import sys
import csv
import random as r


# Value used instead of a feature ID to signal that the label for the following
# example is being given on this row.
LABEL_SIGNAL = -2


def random_dense_text_file(filename, num_examples, num_features, density,
                           delimiter='\t', lineterminator='\n'):
	"""Generate a CSV file containing examples with dense random features."""
	# For simplicity, all features have values between 0.0 and 1.0.

	num_examples = int(num_examples)
	num_features = int(num_features)
	density = float(density)
	
	with open(filename, 'w') as file_out:
		writer = csv.writer(file_out, delimiter=delimiter, lineterminator=lineterminator)
		for _ in xrange(num_examples):
			row = [r.random() if r.random() < density else 0.0 for _ in xrange(num_features)]
			row += [1.0 if r.random() < 0.5 else -1.0]
			writer.writerow(row)


def random_sparse_text_file(filename, num_examples, num_features, density,
                            delimiter='\t', lineterminator='\n'):
	"""Generate a CSV file containing examples with sparse random features."""
	# For simplicity, all features have values between 0.0 and 1.0.
	# Format follows that of RCV1.train.tsv
	
	num_examples = int(num_examples)
	num_features = int(num_features)
	density = float(density)
	
	with open(filename, 'w') as file_out:
		writer = csv.writer(file_out, delimiter=delim)
		for example_id in xrange(num_examples):
			writer.writerow([example_id, LABEL_SIGNAL, 1.0 if r.random() < 0.5 else -1.0])
			for feature_id in xrange(num_features):
				if r.random() < density:
					writer.writerow([example_id, feature_id, r.random()])


def actual_density_dense_text_file(filename, num_features, delimiter='\t'):
	"""The actual fraction of non-zero feature values in a dense text file."""
	num_values, nonzero = 0, 0
	with open(filename, 'r') as file_in:
		for row in csv.reader(file_in, delimiter=delimiter):
			# The last value in a row is the label; don't count this
			num_values += len(row) - 1
			nonzero += sum(1 for value in row[:-1] if float(value) != 0.0)
	return nonzero / float(num_values)
		

def actual_density_sparse_text_file(filename, num_features, delimiter='\t'):
	"""The actual fraction of non-zero feature values in a sparse text file."""
	num_features = int(num_features)
	
	num_values, nonzero = 0, 0
	with open(filename, 'r') as file_in:
		for row in csv.reader(file_in, delimiter=delimiter):
			if int(row[1]) == LABEL_SIGNAL:
				num_values += num_features
			else:
				nonzero += 1
	return nonzero / float(num_values)


if __name__ == '__main__':
	command = globals()[sys.argv[1]]
	ret = command(*sys.argv[2:])
	if ret != None:
		print ret



