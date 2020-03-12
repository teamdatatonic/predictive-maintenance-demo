# Predictive Maintenance - BackBlaze Hard Drive Dataset
---

This folder contains code for a predictive maintenance use case concerning a [dataset](https://www.backblaze.com/b2/hard-drive-test-data.html) from BackBlaze. The goal is to predict hard drive failure.  

## Table of Contents
---

 - About the Project
 - Getting Started
 - Data
 - Run
 - Next Steps

## About the Project
---

In this project we explore a predictive maintenance use case for hard drives in which there are labelled instances of failure. Using these labels, the predictive maintenance use case can be framed as a supervised learning problem in which the goal is to predict if a particular hard drive will fail some number of days `x` in the future. The training and evaluation procedure is loosely based on the [multiple classifier approach](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/17844756/machine.pdf) of Susto et al.

### Built With
This project was primarily build with the following tools:
 - AI Platform Notebooks
 - BigQuery
 - Python (sklearn, pandas)
 - XGBoost    


## Getting Started
---

The easiest and fastest way to get started is to create a new python AI platform notebook instance. Once the instance is created and jupyterlab is launched, open a new terminal in jupyterlab and perform the following steps:

 1. Clone the repo:
```bash
git clone git@github.com:teamdatatonic/predictive-maintenance-demo.git
```
 2. Install XGBoost and SHAP (only packages to install if using AI platform notebooks)
```bash
pip install --user xgboost==0.90
pip install --user shap
```



## Data
---

Since 2013, BackBlaze has published statistics and insights based on the hard drives in their data center. The data consists of daily snapshots of each operational hard drive. This snapshot includes basic drive information along with the [S.M.A.R.T](https://en.wikipedia.org/wiki/S.M.A.R.T.) statistics reported by that drive. The daily snapshot of one drive is one row of data. All of the drive snapshots for a given day are collected into a csv file in the format YYYY-MM-DD.csv (e.g. 2013-04-10.csv) consisting of a row for each active hard drive.

The first row of each file contains the column names:
 - Date
 - Serial Number : Manufacturer assigned serial number of the drive
 - Model : Manufacturer model of the drive
 - Capacity : Drive capacity in bytes
 - Failure : Binary label with 1 indicating that this was the last day the drive was operational before failing.
 - SMART stats
    - 2013-2014 : Raw and Normalized values for 40 different SMART stats.
    - 2015-2017 : Raw and Normalized values for 45 different SMART stats.
    - 2018 (Q1) : Raw and Normalized values for 50 different SMART stats.
    - 2018 (Q2) : Raw and Normalized values for 52 different SMART stats.
    - 2018 (Q4) : Raw and Normalized values for 62 different SMART stats.


## Run
---

Here we present steps to rerun the pipeline. The main steps include the following:

 1. Loading raw data into GCS
 2. Ingesting the data into BigQuery
 3. Cleaning the data
 4. Training
 5. Testing
 6. Exploring Results

### 1. Load Data to GCS

The first step is to get the data into GCS. GCS is used as a staging location before data is moved to BigQuery. This is done by downloading the zip files from the BackBlaze website, unzipping them, and uploading the raw csv files to GCS. First create a new bucket (<BUCKET>) in google cloud storage within the same project as your AI Platform Notebook instance. Open a new terminal instance and type the following:

```bash
bash upload_to_gcs.sh <BUCKET>
```

This will download all of the relevant data from the backblaze webiste and upload it to GCS. The result will be a folder in GCS called `hard-drive-failure`. Running this script will take about **10-15 minutes**.

### **2. Ingesting the Data into BigQuery** (MAKE SURE DATASET IS IN SAME REGION AS BUCKET)

Create a dataset (<DATASET>) in BigQuery. **Make sure dataset is in the same region as your bucket and notebook instance**. Then run the `GCStoBigQuery.py` with your project, dataset, and bucket as command line arguments:

```bash
python3 GCStoBigQuery.py --project=<PROJECT> --bucket=<BUCKET> --dataset=<DATASET>
```

Running this will take about **5 minutes** and will create a ~55GB partitioned table called `all_data` in BigQuery.

### **3. Clean the data**

The previous step collects all of the data in a single date-partitioned table in BigQuery. The table contains data for all hard drive models. This introduces complications since different manufacturer models behave differently, report different stats, and have different ways of "normalizing" their SMART stats. In order to simplify the problem, ML models are trained for one particular hard drive model at a time. The file `create_dataset.py` performs the following steps:

 1. Filter all of the data on a particular hard drive model (we use ST4000DM000 since it is about 1/3 of the entire dataset)
 2. Calculate the percentage of nulls in each column and filter columns above a certain threshold (default threshold is 30%)
 3. Create a table with additional labels
 4. Create an additional table with imputed nulls

To run:

```bash
python3 create_dataset.py --project=<PROJECT> --dataset=<DATASET>
```
This will take a couple minutes to run.

#### **3.a Create ML Datasets**

```bash
python3 create_ml_datasets.py --project=<PROJECT> --dataset=<DATASET>
```

Four ML datasets corresponding to four different classification problems are created:
 1. Predict fail today
 2. Predict fail tomorrow
 3. Predict fail this week
 4. Predict fail this month

For now, we perform naive downsampling so as to obtain a 50-50 class split. The fail this week and fail this month datasets are therefore quite a bit larger because they have more "positive" instances of failure. This script will also create a table with data from the first half of 2019 for testing, `test_2019`.

#### **3.b Visualizations**
After cleaning the data we can try to visualize some of the SMART features to get a better understanding of their behavior, particularly for drives that are near failure. These visualizations can be found in the `Exploratory_Visualizations.ipynb` notebook.

###  **4. Train Models**

For each of the four decision thresholds train an XGBoost model and compare to a manual decision rule based on the [findings](https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/) from BackBlaze:
```
python3 train_and_eval.py --project=<PROJECT> --dataset=<DATASET> --bq_source_tables='50_50_fail_today,50_50_fail_today_or_tomorrow,50_50_fail_this_week,50_50_fail_this_month'
```

This will save the trained models and classification metrics in a new folder called `train_results/`. Training all 8 models (XGBoost with and without history on four different failure horizons) will take upwards of 30 minutes.

### **5. Test Models**

Metrics give an indication of how successful the models were on different prediction problems, but they give little indication of whether model would provide real business value. First, export the test data to a new "folder" called `test` in the same <BUCKET> you've been using in the steps above.
```
python3 test.py --bucket=<BUCKET>
```

The testing procedure involves finding the number of unexpected breaks and false alarms for each ML model if it had been running in production during the first half of 2019.



## Results
---

Summary of results can be found in an accompanying blog post. If you'd like to better understand
the models and explore individual predictions, then feel free to play around with the
`Model_Interpretability.ipynb` notebook.


## Next Steps
This was the first phase of a predictive maintenance project and mostly involved exploration of approaches. There are many things to address in a possible phase 2:

 - Modelling
     - **Hyperparameter tuning** : No hyperparameter tuning was performed
     - **Feature engineering** : We did some minimal engineering of time series features. This is a potential avenue for further exploration.
     - **Look at different ML models** : Here we only used XGBoost
 - Data
     - **More sophisticated sampling** : Here we only used a 50-50 class split, simplifying the problem but also leading to us throwing away the majority of the data. We could also just try to use all of the data
     - **Deal with different manufacturer models** : Only looked at one manufacturer model
         - Different (ML) models for each (manufacturer) model?
         - One-hot encode manufacturer model and feed as feature into ML model?
         - Transfer knowledge from one hard drive model to another?
 - Testing
     - Testing procedure is very basic at this point
     - Create more interactive results/embed into an application
 - Create pipeline
     - Automate the procedure end-to-end. Quite a few steps need to be done manually at this point in time.
