"""
Generate synthetic datasets with a variable number of examples, features, and sparsity.
"""

import sys
import csv
import random as r


# Value used instead of a feature ID to signal that the label for the following
# example is being given on this row.
LABEL_SIGNAL = -2


def random_dense_text_file(filename, num_examples, num_features, density, delim='\t'):
	"""Generate a CSV file containing examples with dense random features."""
	with open(filename, 'w') as file_out:
		writer = csv.writer(file_out, delimiter=delim)
		for _ in xrange(num_examples):
			row = [r.random() if r.random() < density else 0.0 for _ in xrange(num_features)]
			writer.writerow(row)


def random_sparse_text_file(filename, num_examples, num_features, density, delim='\t'):
	"""Generate a CSV file containing examples with sparse random features."""
	# Format follows that of RCV1.train.tsv
	with open(filename, 'w') as file_out:
		writer = csv.writer(file_out, delimiter=delim)
		for example_id in xrange(num_examples):
			writer.writerow([example_id, LABEL_SIGNAL, 1.0 if r.random() < 0.5 else -1.0])
			for feature_id in xrange(num_features):
				if r.random() < density:
					writer.writerow([example_id, feature_id, r.random()])


if __name__ == '__main__':
	command = globals()[sys.argv[1]]
	filename, num_examples, num_features, density = sys.argv[2:]
	num_examples = int(num_examples)
	num_features = int(num_features)
	density = float(density)
	command(filename, num_examples, num_features, density)



