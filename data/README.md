# Data sets

## SVM

SVM takes a train and a test set, specified by the command line flags. These should be in sparse TSV format.

### Synthetic SVM data

We created a special synthetic dataset specification for experimenting with HogWild! svms. The file must begin with `_synth_svm_` to be interpretted as such by the system. The format for one of these files is
```
number_of_rows  number_of_columns density
```
Where density is the probability that an element will be non zero.

## Matrix Completion

Netflix prize set.
```
user_id movie_id  rating
```

This must be sorted on user_id, then movie_id. 
