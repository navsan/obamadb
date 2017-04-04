#!/usr/bin/env python
from statsmodels.formula.api import ols
import pandas
df = pandas.read_csv('../ObamaDb-osx/temp2.csv')
df['InstPerEpoch'] = df['Instructions']/50
df_dot = df[df['Function'] == 'dot']
df_exec = df[df['Function'] == 'execute']
model = ols("InstPerEpoch ~ NumRows + NumNonZeroes",df_dot).fit()

print("Regression Summary for obamadb::ml::dot")
model = ols("InstPerEpoch ~ NumRows + NumNonZeroes",df_dot).fit()
print(model.summary())
print("Regression Summary for obamadb::SVMTask::execute")
model = ols("InstPerEpoch ~ NumRows + NumNonZeroes",df_exec).fit()
print(model.summary())
