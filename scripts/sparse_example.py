"""
Example usage of pandas' sparse data structures.
"""

import pandas as pd


def small_sparse_dataframe():
	"""An example of a small SparseDataFrame."""
	s1 = pd.SparseSeries([0, 0, 0, 0, 1, 0, 1, 0, 0, 0], fill_value=0)
	s2 = pd.SparseSeries([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], fill_value=0)
	s3 = pd.SparseSeries([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], fill_value=0)
	return pd.SparseDataFrame([s1, s2, s3])
