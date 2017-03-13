# Unit Tests

Regretfully, these unit tests are far from comprehensive.

A few of these tests work by parsing data (`.dat`) files.
* `heart_scale.dat` is a libsvm format file. It is taken from the [liblinear repo][lib-linear]
* `iris.dat` is in a dense format where each line is an example. It is sourced from UCI's machine learning repo.
* `sparse.dat` is some made-up data in a custom sparse format where each line is an entry and several consecutive lines
form a training example row.

[lib-linear]: https://github.com/cjlin1/liblinear