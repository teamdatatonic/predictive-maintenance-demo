"""
Script to create tables in BigQuery suitable for machine learning.
The script will create four different tables corresponding to four
different classification problems and four different "failure horizons":

    1. Fail today
    2. Fail today or tomorrow
    3. Fail this week
    4. Fail this month
Each table will have a roughly 50-50 class split in order to make learning
easier and faster for a downstream ML model.

Data from the year 2019 is excluded and used for testing of models

TODO: Allow one to pass class split as command line parameter
"""
import argparse

from google.cloud import bigquery

LABELS = ['failure', 'fail_today_or_tomorrow',
          'fail_this_week', 'fail_this_month']
TABLE_NAMES = ['50_50_fail_today', '50_50_fail_today_or_tomorrow',
               '50_50_fail_this_week', '50_50_fail_this_month']
# Conditions to exclude instances depending on the classification problem
CONDITION_DICT = dict(zip(TABLE_NAMES, LABELS))

# Base query that is adapted depending on the decision threshold
BASE_QUERY = """
CREATE OR REPLACE TABLE `{1}.{3}` AS (
WITH data AS (
SELECT
  *
FROM
  `{0}.{1}.{2}`
WHERE
  EXTRACT(year FROM date) < 2019),
failures AS (
SELECT
  serial_number,
  date as last_day
FROM
  data
WHERE {7}=1),

non_failures AS (
SELECT
  serial_number,
  date as last_day
FROM
  data
WHERE {7}=0
AND MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(date,serial_number)))),
(SELECT COUNT(*) FROM data)) <
(SELECT COUNT(*) FROM failures))

SELECT *,
        10 - DATE_DIFF(last_day, date, day) as day_in_window
FROM (
SELECT
  *
  EXCEPT(
    {4},
    {5},
    {6},
    model)
FROM
  data
JOIN
    failures
USING (serial_number)
WHERE
  DATE_DIFF(last_day, date, day) < 10 AND DATE_DIFF(last_day, date, day) >=0
UNION ALL
SELECT
  * EXCEPT(
    {4},
    {5},
    {6},
    model)
FROM
  data
JOIN
    non_failures
USING (serial_number)
WHERE
   DATE_DIFF(last_day, date, day) < 10 AND DATE_DIFF(last_day, date, day) >= 0)
ORDER BY serial_number, last_day, day_in_window)
"""
# Add the order by just to make feature extraction easier later

TEST_DATA_QUERY = """
CREATE OR REPLACE TABLE `{0}.{1}.test_2019` AS (
SELECT
    *
FROM `{0}.{1}.{2}`
WHERE date >= '2019-01-01' AND date <= '2019-06-01'
)
"""


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        type=str,
                        help='Original table to work from')
    parser.add_argument('--dataset',
                        type=str,
                        help='Original table to work from')
    parser.add_argument('--table',
                        type=str,
                        default='cleaned_data',
                        help='Original table to work from')

    return parser.parse_args()


def run_query(query, client, table):
    """
    Actually run the queries
    """
    print('{:-^80}'.format(" Creating Table {} ".format(table)))
    query_job = client.query(query)
    rows = query_job.result()  # Wait for query to finish
    print('{:=^80}'.format(" SUCCESS "))


def create_ml_datasets():
    """Main method to create ML datasets"""
    args = parse_args()
    client = bigquery.Client()
    for i, dest_table in enumerate(TABLE_NAMES):
        # Labels to exclude depending on classification problem
        exclude = [LABELS[idx] for idx, label in enumerate(LABELS) if idx != i]

        query = BASE_QUERY.format(args.project,
                                  args.dataset,
                                  args.table,
                                  dest_table,
                                  *exclude,
                                  CONDITION_DICT[dest_table])
        run_query(query, client, dest_table)

    test_query = TEST_DATA_QUERY.format(args.project,
                                        args.dataset,
                                        args.table)

    run_query(test_query, client, 'Test Data')


if __name__ == '__main__':
    create_ml_datasets()
