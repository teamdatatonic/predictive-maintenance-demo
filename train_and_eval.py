"""
Script to train and evaluate XGBoost models on datasets in BigQuery. As a
baseline for comparison, we will use simple decision rules based on domain
knowledge:
https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/

We will train simple XGBoost models with no hyperparameter tuning just using
the default settings. For each dataset, we will output several metrics for both
xgboost and baseline models

#TODO: export tables as csvs and then read from cloud storage
(much faster then reading from BQ)
"""
import argparse
import os

import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pickle

from google.cloud import bigquery

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import numpy as np

LABEL_DICT = {'50_50_fail_today': 'failure',
              '50_50_fail_today_or_tomorrow': 'fail_today_or_tomorrow',
              '50_50_fail_this_week': 'fail_this_week',
              '50_50_fail_this_month': 'fail_this_month'}

# Size of windows in BigQuery
WINDOW_SIZE = 10


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        type=str,
                        help='BigQuery dataset')

    parser.add_argument('--dataset',
                        type=str,
                        help='BigQuery dataset')

    parser.add_argument('--bq_source_tables',
                        type=str,
                        default='50_50_fail_today',
                        help='Comma delimited list of tables input')

    parser.add_argument('--results_dir',
                        type=str,
                        default='train_results/',
                        help='Where to output trained models and eval results')

    # TODO: Add more features (whether to hyperparameter tune, how to write
    # out results, drop normalized, which features to look at etc.)

    return parser.parse_args()


def get_table_as_frame(dataset, table, client):
    """
    Get BigQuery table as Pandas DataFrame to do machine learning

    Parameters
    ----------
    dataset : BigQuery dataset

    table : BigQuery table

    client : BigQuery client

    Returns
    -------
    Pandas DataFrame
    """
    query = """
            SELECT
                *
            FROM `{0}.{1}`""".format(dataset, table)

    query_job = client.query(query)
    results = query_job.result().to_dataframe()

    return results


def get_features(groups, cols, history=False):
    """
    Get aggregated features based on grouped dataset.

    Parameters
    ----------
    groups : Pandas GroupBy object
        Data grouped by serial_number and last_day (windows)

    cols : list
        List of columns to keep (all of the 'raw' SMART features)

    history : bool, optional
        Whether to include history (context) to create agg features
    """
    # Aggregations to do
    aggs = ['last']
    if history:
        aggs += [np.mean, np.var, 'min', 'max']

    features = groups[cols].agg(aggs)

    return features


def xgboost_train_and_predict(X_train, y_train, X_test):
    """
    Method to train and evaluate xgboost. At this time no hyperparameter
    tuning is done.

    Parameters
    ----------
    X_train, y_train, X_test : Pandas DataFrames

    Returns
    -------
    preds : Predictions for test set
    """
    bst = xgb.XGBClassifier()

    bst.fit(X_train, y_train)
    preds = bst.predict(X_test)

    return preds, bst


def get_metrics(preds, labels):
    """
    Print relevant metrics given predicted and true labels
    """
    acc = metrics.accuracy_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    precision = metrics.precision_score(labels, preds)
    F1 = metrics.f1_score(labels, preds)
    auc = metrics.roc_auc_score(labels, preds)
    confusion_matrix = metrics.confusion_matrix(labels, preds, labels=(0, 1))

    return [acc, recall, precision, F1, auc] + list(confusion_matrix.ravel())


def evaluate(preds, labels, table, model=None):
    """
    Obtain evaluation metrics for a model

    preds, labels : arrays

    model : str
        Should be table name + model (xgb, xgb_history, baseline)
    """
    result = [table, model]
    result += get_metrics(preds, labels)

    return result


def main():
    args = parse_args()

    def train_eval_and_save(grouped,
                            labels,
                            cols_to_keep,
                            table,
                            history=False):
        """
        Method to train and evaluate XGBoost models and save the trained model
        """
        features = get_features(grouped, cols_to_keep, history=history)
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            labels,
                                                            test_size=0.2,
                                                            random_state=123)

        preds, model = xgboost_train_and_predict(X_train, y_train, X_test)

        if history:
            model_name = "xgb_history"
            file_name = "_history.pickle.dat"
        else:
            model_name = "xgb"
            file_name = ".pickle.dat"

        result = evaluate(preds, y_test, table, model_name)
        # Save the model
        pickle.dump(model, open(args.results_dir + table + file_name, "wb"))

        return result, X_test, y_test

    # Create directory to store trained models and results
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Features we are using to manually create a decision tree
    # TODO: create option to pass these in
    BASELINE_FEATURES = ['smart_5_raw', 'smart_187_raw',
                         'smart_188_raw', 'smart_197_raw', 'smart_198_raw']

    # TODO: authorize bigquery client if not using AI platform
    client = bigquery.Client()

    # Perform analysis for each table.
    # Corresponds to training models at different decision thresholds.
    tables = args.bq_source_tables.split(',')

    results = []
    src_table = '{}.{}'.format(args.project, args.dataset)

    for table in tables:
        dataset = get_table_as_frame(src_table, table, client)
        cols_to_keep = [col for col in dataset.columns if 'smart' in col]

        # Groupby will get 10 day windows
        grouped = dataset.groupby(['serial_number', 'last_day'])
        # Label should always be what the last day in the window is
        labels = grouped[LABEL_DICT[table]].agg('last')

        print("{:=^80}".format(" Training on {} ".format(table)))
        print("{:-^80}".format(" Evaluating XGBoost with History "))
        result_history, _, _ = train_eval_and_save(grouped,
                                                   labels,
                                                   cols_to_keep,
                                                   table,
                                                   history=True)
        results.append(result_history)

        print("{:-^80}".format(" Evaluating XGBoost without History "))
        # Get X_test and y_test so we can evaluate baseline model
        result, X_test, y_test = train_eval_and_save(grouped,
                                                     labels,
                                                     cols_to_keep,
                                                     table,
                                                     history=False)
        results.append(result)

        print("{:-^80}".format(" Evaluating Baseline "))
        preds_manual = np.any(
            X_test[BASELINE_FEATURES].values > 0, axis=1).astype(int)

        results.append(evaluate(preds_manual, y_test, table, "baseline"))

        print("\n \n")

    results_frame = pd.DataFrame(results, columns=['Dataset',
                                                   'model',
                                                   'accuracy',
                                                   'recall',
                                                   'precision',
                                                   'F1',
                                                   'auc',
                                                   'TN',
                                                   'FP',
                                                   'FN',
                                                   'TP'])

    results_frame.to_csv(args.results_dir + 'train_results.csv')


if __name__ == '__main__':
    main()
