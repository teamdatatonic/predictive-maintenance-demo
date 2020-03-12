"""Script to create a table with all hard drive data in BigQuery

The following steps are performed:

    1. Get schema by looking at a sample file
    2. Load data into a partioned and clustered table in BQ

Since we have a changing schema we'll have to repeatedly get sample
schemas for the load job.
"""
from google.cloud import bigquery
from google.cloud import storage
from google.cloud.bigquery.table import TimePartitioning
import pandas as pd
import argparse

# Instantiate clients to be used
client = bigquery.Client()
storage_client = storage.Client()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--project',
                        type=str,
                        help='Google cloud project ID')

    parser.add_argument('--bucket',
                        type=str,
                        help='GCS bucket with the data')

    parser.add_argument('--dataset',
                        type=str,
                        help='BigQuery destination dataset')

    parser.add_argument('--table',
                        type=str,
                        default="all_data",
                        help='BigQuery destination table')

    return parser.parse_args()


def get_schema(bucket, folder):
    """
    The schema changes every now and then so we'll have to get a sample
    schema for each folder
    """
    def schema_gen(col):
        """
        Helper function to get desired schema. Needed to ensure that
        'smart' feature columns are loaded as ints and not string in
        case they are all null
        """
        if 'model' in col or 'serial_number' in col:
            return 'STRING'
        elif 'date' in col:
            return 'DATE'
        else:
            return 'INT64'
    # Just use the first file
    sample_file = ['gs://' + ('/').join(blob.id.split('/')[:-1]) for blob
                   in storage_client.list_blobs(bucket,
                                                prefix='hard-drive-failure/{}'.
                                                format(folder),
                                                max_results=1)][0]
    sample_frame = pd.read_csv(sample_file, nrows=1)

    return [bigquery.SchemaField(col, schema_gen(col))
            for col in sample_frame.columns]


def main():
    args = parse_args()
    dest_table = '{}.{}.{}'.format(args.project,
                                   args.dataset,
                                   args.table)

    # Need to do this for each folder separately since we might have different
    # schemas 2013 doesn't have enough data so we skip it for now
    folders = list(set([blob.id.split('/')[2] for blob in
                        storage_client.list_blobs(args.bucket,
                                                  prefix='hard-drive-failure')
                        if '2013' not in blob.id]))

    file_patterns = ["gs://{0}/hard-drive-failure/{1}/*.csv".format(
        args.bucket,
        folder)
        for folder in folders]

    for file_pattern, folder in zip(file_patterns, folders):
        print("{:=^80}".format(" Loading Data for {} ".format(folder)))
        # Update the schema for every folder
        schema = get_schema(args.bucket, folder)

        job_config = bigquery.LoadJobConfig(skip_leading_rows=1,
                                            schema_update_options=[
                                                'ALLOW_FIELD_ADDITION',
                                                'ALLOW_FIELD_RELAXATION'],
                                            schema=schema,
                                            clustering_fields=[
                                                'model',
                                                'serial_number'])
        # Only works when I defer setting it for some reason?
        job_config.time_partitioning = TimePartitioning(field='date')

        load_job = client.load_table_from_uri(
            file_pattern, dest_table, job_config=job_config)
        load_job.result()  # Wait for job to finish

        print("{:-^80}".format(" Data Successfully Loaded "))

    print("SUCCESS")


if __name__ == '__main__':
    main()
