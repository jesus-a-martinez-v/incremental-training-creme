import argparse

from creme import stream
from creme.compose import Pipeline
from creme.linear_model import PAClassifier
from creme.metrics import Accuracy
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-t', '--train', required=True, help='Path to training features CSV file.')
argument_parser.add_argument('-e', '--test', required=True, help='Path to test CSV file.')
argument_parser.add_argument('-n', '--num-cols', type=int, required=True,
                             help='Number of columns in the feature CSV file (excluding label).')
arguments = vars(argument_parser.parse_args())

print('[INFO] Building column names...')
types = {f'feature_{i}': float for i in range(arguments['num_cols'])}  # Data type per feature
types['class'] = int

dataset = stream.iter_csv(arguments['train'], target_name='class', types=types)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('learner', OneVsRestClassifier(binary_classifier=PAClassifier()))
])

metric = Accuracy()

print('[INFO] Training started...')
for index, (X, y) in enumerate(dataset):
    try:
        predictions = model.predict_one(X)
        model = model.fit_one(X, y)
        metric = metric.update(y, predictions)

        if index % 10 == 0:
            print(f'[INFO] Update {index} - {metric}')
    except OverflowError as e:
        print(f'Overflow error. Skipping metric update for {index}')

print(f'[INFO] Final - {metric}')

print('[INFO] Testing model...')
metric = Accuracy()
test_dataset = stream.iter_csv(arguments['test'], target_name='class', types=types)
for index, (X, y) in enumerate(test_dataset):
    predictions = model.predict_one(X)
    metric = metric.update(y, predictions)

    if index % 1000 == 0:
        print(f'[INFO] (TEST) Update {index} - {metric}')

print(f'[INFO] (TEST) Final - {metric}')
