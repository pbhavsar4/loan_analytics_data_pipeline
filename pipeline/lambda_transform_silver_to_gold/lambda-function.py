import os
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import psycopg2
import csv
import tempfile
import json

# --- S3 client ---
s3 = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# --- Environment Variables ---
BUCKET_NAME = os.environ.get("BUCKET_NAME")
SILVER_PREFIX = os.environ.get("SILVER_PREFIX", "Silver/")
GOLD_PREFIX = os.environ.get("GOLD_PREFIX", "Gold/")
RDS_SECRET_NAME = os.environ.get("DB_SECRET_NAME")
RDS_Host = os.environ.get("DB_HOST")

# --- Read Parquet folder from S3 ---
def read_parquet_folder_from_s3(bucket, folder_prefix):
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=folder_prefix)
    dfs = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                response = s3.get_object(Bucket=bucket, Key=key)
                buffer = BytesIO(response['Body'].read())
                table = pq.read_table(buffer)
                df = table.to_pandas()
                df.columns = [c.lower() for c in df.columns]
                dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No Parquet files found in s3://{bucket}/{folder_prefix}")
    return pd.concat(dfs, ignore_index=True)

# --- Write Parquet to S3 ---
def write_parquet_to_s3(df, bucket, key):
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

# --- Get RDS credentials ---
def get_rds_credentials(secret_name):
    secret_value = secrets_client.get_secret_value(SecretId=secret_name)
    return json.loads(secret_value['SecretString'])

# --- Load DataFrame to RDS using COPY ---
def load_df_to_rds_copy(df, table_name='customer'):
    creds = get_rds_credentials(RDS_SECRET_NAME)
    conn = psycopg2.connect(
        host=RDS_Host,
        port=5432,
        dbname="appdb",
        user=creds['username'],
        password=creds['password']
    )
    cur = conn.cursor()

    # Create table if not exists
    columns_with_types = []
    for col, dtype in zip(df.columns, df.dtypes):
        if pd.api.types.is_integer_dtype(dtype):
            columns_with_types.append(f"{col} INTEGER")
        elif pd.api.types.is_float_dtype(dtype):
            columns_with_types.append(f"{col} FLOAT")
        elif pd.api.types.is_bool_dtype(dtype):
            columns_with_types.append(f"{col} BOOLEAN")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            columns_with_types.append(f"{col} TIMESTAMP")
        else:
            columns_with_types.append(f"{col} TEXT")

    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns_with_types)})"
    cur.execute(create_table_sql)
    conn.commit()

    # Save DataFrame to a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        df.to_csv(tmpfile.name, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
        tmpfile.flush()
        # Copy CSV to RDS
        with open(tmpfile.name, 'r') as f:
            cur.copy_expert(sql=f"COPY {table_name} ({', '.join(df.columns)}) FROM STDIN WITH CSV", file=f)
        conn.commit()

    cur.close()
    conn.close()
    print(f"✅ Loaded {len(df)} records into RDS table '{table_name}'")

# --- Lambda handler ---
def lambda_handler(event, context):
    # Read Silver layer tables
    fact_df = read_parquet_folder_from_s3(BUCKET_NAME, f"{SILVER_PREFIX}fact_loan_payment/")
    customer_df = read_parquet_folder_from_s3(BUCKET_NAME, f"{SILVER_PREFIX}dim_customer/")

    # Join to include Region info
    merged_df = pd.merge(fact_df, customer_df, on="customer_id", how="left")

    # Aggregate metrics by region
    agg_df = merged_df.groupby("region").agg({
        "loan_amount": "sum",
        "outstanding_amount": "sum",
        "emi_amount": "sum",
        "payment_delay_days": "mean",
        "customer_id": "count"
    }).reset_index()

    # Rename columns for clarity
    agg_df.rename(columns={
        "loan_amount": "total_loan_amount",
        "outstanding_amount": "total_outstanding_amount",
        "emi_amount": "total_emi_amount",
        "payment_delay_days": "avg_payment_delay",
        "customer_id": "customer_count"
    }, inplace=True)

    # Write aggregated data to Gold layer
    write_parquet_to_s3(agg_df, BUCKET_NAME, f"{GOLD_PREFIX}region_summary/data.parquet")

    # Load customer_df into RDS
    load_df_to_rds_copy(agg_df, table_name='customer')

    return {
        "status": "Success",
        "message": f"Aggregated region summary saved to Gold layer and customer data loaded into RDS",
        "record_count": len(agg_df)
    }
