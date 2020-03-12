"""
Script to create a table in bigquery for a single harddrive model.

There are three main steps:
    1. Filter model
        Here we filter all of the harddrive data on the model such that only
        the desired model is included in the final table.
    2. Calculate null percentage and filter columns
        Calculate the percentage of nulls in each column of table. If
        the percentage of nulls is greater than some predefined threshold we
        drop the column. Nulls in remaining columns can either be imputed or
        left as is.
    3. Create table
        After we have filtered on the model and dropped any columns, we create
        a "final" table. For each day, we add columns that represent different
        labels corresponding to different decision thresholds:
            - "fail_today_or_tomorrow"
            - "fail_this_week"
            - "fail_this_month"
            - "days_left"

        *In the future, option to add more labels should be added*
    4. Impute Nulls
        Last step is to create a new table with imputed nulls using the mode
        for each feature. Helpful for testing downstream ML models.
"""
import argparse
import pandas as pd

from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.bigquery import job

INPUT_QUERY = """CREATE OR REPLACE TABLE
    `{1}`
    PARTITION BY date
        AS (
          WITH failure_dates_per_serial_number AS (
          SELECT
            serial_number,
            ARRAY_AGG(date ORDER BY date)[ORDINAL(1)] AS first_failure_date
          FROM
            `{0}`
          WHERE
            failure = 1
          GROUP BY
            serial_number),

          no_failure_serial_numbers AS (
          SELECT * EXCEPT(failure) FROM (
            SELECT
              serial_number,
              ARRAY_AGG(failure ORDER BY date) as failure
            FROM
              `{0}`
            GROUP BY serial_number)
          WHERE 1 NOT IN UNNEST(FAILURE)),

          all_data_no_failure AS (
          SELECT *,
            100 AS days_left,
            0 as fail_today_or_tomorrow,
            0 as fail_this_week,
            0 as fail_this_month
          FROM
            `{0}`
          JOIN no_failure_serial_numbers b
          USING(serial_number))

        SELECT
          * EXCEPT(before_first_failure,
                   days_to_failure,
                   first_failure_date),
          CASE WHEN days_to_failure > 100 THEN 100 ELSE days_to_failure END
            AS days_left,
          CASE WHEN days_to_failure < 2 then 1 ELSE 0 END
            AS fail_today_or_tomorrow,
          CASE WHEN days_to_failure < 7 then 1 ELSE 0 END
            AS fail_this_week,
          CASE WHEN days_to_failure < 30 then 1 ELSE 0 END
            AS fail_this_month
        FROM (
          SELECT
            *,
            CASE WHEN date <= first_failure_date THEN 1 ELSE 0 END
                AS before_first_failure,
            ABS(DATE_DIFF(date, first_failure_date, DAY))
                AS days_to_failure
          FROM
            `{0}`
          JOIN
            failure_dates_per_serial_number b
          USING
            (serial_number))
        WHERE
          before_first_failure = 1
        UNION ALL
        SELECT *
        FROM all_data_no_failure
        )"""


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--project',
                        type=str,
                        help='GCP project ID')

    parser.add_argument('--dataset',
                        type=str,
                        help='BigQuery dataset')

    parser.add_argument('--bq_source_table',
                        type=str,
                        default='all_data',
                        help='Original table to work from')

    parser.add_argument('--model',
                        type=str,
                        default='ST4000DM000',
                        help='Harddrive model for which we would like to do \
                                further analysis')

    parser.add_argument('--null_threshold',
                        type=float,
                        default=0.3,
                        help='Feature column dropped if null percentage \
                                greater than threshold')

    parser.add_argument('--bq_dest_table',
                        type=str,
                        default='cleaned_data',
                        help='Bigquery destination table')
    # TODO: Add boolean argument to impute nulls (or --impute, --no_impute)
    # right now we are imputing nulls by default

    return parser.parse_args()


def get_tmp_table(result):
    """
    Find name of table from bigquery job
    """
    paths = result.path.split('/')
    table = ('.').join([paths[2], paths[4], paths[6]])

    return table


def filter_model_job(model, table, client):
    """
    Create query to filter dataset based on the hard drive model.

    Parameters
    ----------
    model : str
        Default is 'ST4000DM000', the most common model

    table : str
        Table containing all backblaze hard
        drive data from 2014-2019

    client : BigQuery client

    Returns
    filter_model_result : RowIterator
        Result of BigQuery job
    """
    query = """
        SELECT
            *
        FROM `{0}`
        WHERE model = '{1}'
        """.format(table, model)

    query_job = client.query(query)

    return query_job.result()


def calc_null_percentage_job(result, client):
    """
    Run query to calculate the percentage of nulls in each column of filtered
    table

    Parameters
    ----------
    result : bigquery.table.RowIterator
        Result of filter_model_job

    client : bigquery client

    Returns
    -------
    null_percentage_result : DataFrame
        Result of BQ job converted to DataFrame
    """
    # Get the field names from the schema
    fields = result.schema
    field_names = [field.name for field in fields]
    # Loop through each field and calculate nulls
    query = """SELECT"""

    for i, field in enumerate(field_names):
        query += '\n \t SUM(IF({0} is NULL,1,0))/count(*) AS {0}'.format(field)
        if not i == len(field_names) - 1:
            query += ','

    tmp_table = get_tmp_table(result)

    query += '\n FROM'
    query += '\n \t `{}`'.format(tmp_table)

    query_job = client.query(query)
    return query_job.result().to_dataframe()


def find_cols_to_keep(result_df, threshold=0.3):
    """
    Find columns to keep based on calculated number of nulls in each field

    Parameters
    ----------
    result_df : DataFrame
        Pandas dataframe containing percentage of nulls for each field

    threshold : Float
        Threshold for allowable percentage of nulls in a column

    Returns
    -------
    cols_to_keep : Index
        Name of columns to keep
    """
    result_df = result_df.transpose().rename_axis('Feature', axis=1)
    result_df.columns = ['null_percentage']

    cols_to_keep = result_df[result_df['null_percentage'] < threshold].index

    return cols_to_keep


def filter_columns_job(cols, result, client):
    """
    Run job to filter columns for which there is a high percentage of nulls

    Parameters
    ----------
    cols : list
        List of strings representing columns we'd like to keep

    result : bigquery.table.RowIterator
        result of filter_model_job

    client : BigQuery client

    Returns
    -------
    Result of BigQuery job
    """
    query = """SELECT"""

    for i, col in enumerate(cols):
        query += '\n \t {}'.format(col)
        if not i == len(cols) - 1:
            query += ','

    tmp_table = get_tmp_table(result)

    query += '\n FROM'
    query += '\n \t `{}`'.format(tmp_table)

    query_job = client.query(query)

    return query_job.result()


def create_table_query(result, dest_table):
    """
    Query to create final table

    Parameters
    ----------
    result :
        Result of filter_columns_job

    dest_table : str
        Table in bigquery to write to

    Returns
    -------
    """
    # TODO: create more options for label creation
    tmp_table = get_tmp_table(result)
    input_query = INPUT_QUERY.format(tmp_table,
                                     dest_table)

    return input_query


def find_mode_per_feature(project, dataset, table, client):
    """
    Create a new table with imputed nulls with the most common value
    per feature.

    Parameters
    ----------
    project : str
        GC project ID

    dataset : str
        BigQuery dataset

    table : str
        BigQuery table created as a result of running create_table_query

    client : BigQuery client
    """
    # Get schema of cleaned table
    dataset_ref = client.dataset(dataset, project=project)

    table_ref = dataset_ref.table(table)
    table_name = client.get_table(table_ref)
    field_names = [field.name for field in table_name.schema]

    source_table = '{}.{}.{}'.format(project, dataset, table)

    query = 'CREATE OR REPLACE TABLE '
    query += '`{}_imputed` PARTITION BY date AS ('.format(source_table)
    query += '\nWITH mode_per_feature AS ('
    query += '\nSELECT'
    query += '\n \t model,'
    # Get most common value per feature
    for i, field in enumerate(field_names):
        if 'smart' in field:
            query += ('APPROX_TOP_COUNT({0}, 1)[ORDINAL(1)].value as mode_{0},'.format(
                field))
    # Remove trailing comma
    query = query[:-1]
    query += '\n FROM'
    query += '\n \t `{}`'.format(source_table)
    query += '\n GROUP BY model)'

    # Now impute nulls
    query += '\n SELECT'
    for i, field in enumerate(field_names):
        if 'smart' in field:
            query += '\n \t IFNULL({0}, mode_{0}) as {0},'.format(field)
        else:
            query += '\n \t {},'.format(field)
    # Remove trailing comma
    query = query[:-1]
    query += '\n FROM'
    query += '\n \t `{}`'.format(source_table)
    query += '\n JOIN mode_per_feature'
    query += '\n USING(model))'

    return query


def create_dataset():
    args = parse_args()

    # TODO: authorize bigquery client if not using AI platform
    client = bigquery.Client()
    print("Filtering input table for model {}...".format(args.model))
    src_table = '{}.{}.{}'.format(args.project,
                                  args.dataset,
                                  args.bq_source_table)

    filter_model_result = filter_model_job(model=args.model,
                                           table=src_table,
                                           client=client)

    print("Calculating percentage of nulls in each column...")
    null_percentage_df = calc_null_percentage_job(result=filter_model_result,
                                                  client=client)

    cols_to_keep = find_cols_to_keep(result_df=null_percentage_df,
                                     threshold=args.null_threshold)

    print("Dropping columns with high percentage of nulls...")
    filter_columns_result = filter_columns_job(cols=cols_to_keep,
                                               result=filter_model_result,
                                               client=client)

    print("Creating table in BigQuery...")
    dest_table = '{}.{}.{}'.format(args.project,
                                   args.dataset,
                                   args.bq_dest_table)

    new_table_query = create_table_query(result=filter_columns_result,
                                         dest_table=dest_table)

    # Create final dataset
    query_job = client.query(new_table_query)
    rows = query_job.result()  # Wait for query to finish

    # Create imputed dataset
    print("Calculating mode per feature and imputing nulls...")
    # if args.impute_nulls:
    impute_query = find_mode_per_feature(project=args.project,
                                         dataset=args.dataset,
                                         table=args.bq_dest_table,
                                         client=client)

    query_job = client.query(impute_query)
    rows = query_job.result()  # Wait for query to finish

    print("SUCCESS")


if __name__ == '__main__':
    create_dataset()
