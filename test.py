"""
Script to run a test on recent data. The underlying assumption of the testing
procedure is that if an ML model predicts that a drive will fail, then the hard
drive would be taken out of operation. Therefore, at each day predictions are
only made for drives for which no prediction of failure had been made prior to
that day.

The primary metrics we wish to track are the number of unexpected breaks and
the number of false alarms that would have been output by the ML models had
they been used during the first half of 2019.
"""
import numpy as np
import pandas as pd
from google.cloud import storage

import pickle
import argparse

import warnings
with warnings.catch_warnings():
    # filter sklearn\externals\joblib\parallel.py:268:
    # DeprecationWarning: check_pickle is deprecated
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from sklearn import metrics
    import xgboost as xgb


BASELINE_FEATURES = ['smart_5_raw', 'smart_187_raw',
                     'smart_188_raw', 'smart_197_raw', 'smart_198_raw']

MODELS = ['baseline',
          '50_50_fail_today',
          '50_50_fail_today_or_tomorrow',
          '50_50_fail_this_week',
          '50_50_fail_this_month',
          '50_50_fail_today_history',
          '50_50_fail_today_or_tomorrow_history',
          '50_50_fail_this_week_history',
          '50_50_fail_this_month_history']

storage_client = storage.Client()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket',
                        type=str,
                        help='Bucket in GCS with test data')

    parser.add_argument('--prefix',
                        type=str,
                        default='hard-drive-failure/test/test',
                        help='Prefix for test data in bucket')

    parser.add_argument('--results_dir',
                        type=str,
                        default="train_results/",
                        help='Location of saved ML models')

    return parser.parse_args()


def get_test_data(bucket, prefix):
    """
    Function to get all test data

    Parameters
    ----------
    bucket : str
        Name of bucket in gcs

    prefix : str
        Prefix for the test data in bucket

    Returns
    -------
    concatenated_df : DataFrame
        A dataframe containing all of the data from the csvs
    """
    # List the relevant files in the bucket
    files = ['gs://' + ('/').join(blob.id.split('/')[:-1])
             for blob in storage_client.list_blobs(bucket,
                                                   prefix=prefix)]
    # Read the files, concatenate them, and sort by date
    df_for_each_blob = (pd.read_csv(file) for file in files)
    concatenated_df = pd.concat(
        df_for_each_blob, ignore_index=True).sort_values('date')
    # Some drives failed between Jan 1st and Jan 10th, we remove them here
    # (This could also just have been done in BQ ahead of time)
    already_removed_drives = concatenated_df[
        (concatenated_df['failure'] == 1)
        & (concatenated_df['date'] < '2019-01-10')]['serial_number'].values
    concatenated_df = concatenated_df[~concatenated_df['serial_number'].isin(
        already_removed_drives)]

    return concatenated_df


def get_model(model, model_path):
    """
    Load a saved model so we can use it for predictions

    Parameters
    ----------
    model : string
        Name of the model

    model_path : string
        Path to the saved model

    Returns
    -------
    loaded_model : Model
        Trained Sklearn/xgboost model used to make predictions
    """
    model_file = model_path.format(model)
    loaded_model = pickle.load(open(model_file, "rb"))

    return loaded_model


def run_test(model, test_data, model_path):
    """
    Run a test for a particular model

    Parameters
    ----------
    model : string
        Name of the model

    test_data : DataFrame
        Data for which we'd like to run the test

    model_path :

    Returns
    -------
    results : dict
    """
    date_list = test_data['date'].unique()

    # Load model and figure out which aggregations to do
    if model != 'baseline':
        loaded_model = get_model(model, model_path)

    # Same aggregations we used for training
    if 'history' in model:
        aggs = ['last', np.mean, np.var, 'min', 'max']
    else:
        aggs = ['last']

    # Keep track of the drives we have "removed".
    # These represent drives which have failed or that we have
    # done maintenance on.
    removed_drives = set()
    # The two primary metrics we care about
    unexpected_breaks = 0
    false_alarms = 0

    print('{:=^80}'.format(' Evaluating {} '.format(model)))

    # Now basically do 10 day sliding windows, where 10
    # is the length of our historical window.
    for i in range(len(date_list) - 9):
        dates = date_list[i:i+10]

        # Get 10 days of history for drives that have not been 'removed'
        window_df = test_data[(test_data['date'].isin(dates)) & ~(
            test_data['serial_number'].isin(removed_drives))]

        grouped_df = window_df.groupby('serial_number')
        features = grouped_df[[
            col for col in window_df.columns if 'smart' in col]]
        features = features.agg(aggs)

        results = grouped_df['failure'].agg(['last'])

        if model == 'baseline':
            results['prediction'] = np.any(features[BASELINE_FEATURES].values >
                                           0, axis=1).astype(int)
        else:
            results['prediction'] = loaded_model.predict(features)

        predicted_failed_drives = set(
            results[results['prediction'] == 1].index)
        actual_failed_drives = set(results[results['last'] == 1].index)

        # Actually failed but not predicted
        unexpected_breaks += len(actual_failed_drives
                                 - predicted_failed_drives)
        # Predicted but didn't fail
        false_alarms += len(predicted_failed_drives - actual_failed_drives)
        # Both predicted and actually failed are removed
        removed_drives.update(
            predicted_failed_drives.union(actual_failed_drives))

        # There are a lot of false alarms on the first day, keep track of
        # this number
        if i == 0:
            initial_false_alarms = false_alarms

        if i % 20 == 0:
            print("By {} we have {} unexpected breaks and {} false alarms"
                  .format(dates[-1],
                          unexpected_breaks,
                          false_alarms))

    final_results = {'Unexpected Breaks': unexpected_breaks,
                     'False Alarms': false_alarms,
                     'Initial False Alarms': initial_false_alarms}

    print(final_results)
    print('{:-^80}'.format(''))

    return final_results


def main():
    args = parse_args()
    test_data = get_test_data(args.bucket, args.prefix)

    # Location of saved models on disk
    model_path = args.results_dir + '{}.pickle.dat'

    results = {}
    for model in MODELS:
        results[model] = run_test(model, test_data, model_path)

    results_frame = pd.DataFrame(results).T
    results_frame.to_csv('test_results.csv')


if __name__ == '__main__':
    main()
