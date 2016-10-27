% This script uses the output files from main to check
% the reported error of obamadb against octave.

data = csvread('/tmp/RCV1.dense.csv');
t = load('/tmp/theta.dat');

% Actual classifications
b = data(:, columns(data));

% Training examples
A = data(:, 1:columns(data) - 1);

% Prediction
y = sign(A * t');

training_error = sum(y != b)/rows(y)
