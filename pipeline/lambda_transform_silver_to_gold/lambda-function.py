import os
import boto3
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from io import BytesIO

# --- S3 client ---
s3 = boto3.client('s3')

# --- Environment Variables ---
BUCKET_NAME = os.environ.get("BUCKET_NAME")
SILVER_PREFIX = os.environ.get("SILVER_PREFIX", "Silver/")
GOLD_PREFIX = os.environ.get("GOLD_PREFIX", "Gold/")
ATHENA_DATABASE = os.environ.get("ATHENA_DATABASE", "loan_analytics")


def read_parquet_folder_from_s3(bucket, folder_key):
    """Read all Parquet files in a folder and return a Pandas DataFrame."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=folder_key)

    dfs = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                response = s3.get_object(Bucket=bucket, Key=key)
                buffer = BytesIO(response["Body"].read())
                table = pq.read_table(buffer)
                dfs.append(table.to_pandas())

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"No Parquet files found in s3://{bucket}/{folder_key}")


def write_parquet_to_s3(df, bucket, key):
    """Write a Pandas DataFrame to Parquet in S3."""
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def lambda_handler(event, context):
    # --- Read Silver layer tables ---
    fact_df = read_parquet_folder_from_s3(BUCKET_NAME, f"{SILVER_PREFIX}fact_loan_payment/")
    customer_df = read_parquet_folder_from_s3(BUCKET_NAME, f"{SILVER_PREFIX}dim_customer/")

    # --- Join to include Region info ---
    merged_df = pd.merge(fact_df, customer_df, on="Customer_ID", how="left")

    # --- Aggregate metrics by Region ---
    agg_df = merged_df.groupby("Region").agg({
        "Loan_Amount": "sum",
        "Outstanding_Amount": "sum",
        "EMI_Amount": "sum",
        "Payment_Delay_Days": "mean",
        "Customer_ID": "count"
    }).reset_index()

    # --- Rename columns for clarity ---
    agg_df.rename(columns={
        "Loan_Amount": "Total_Loan_Amount",
        "Outstanding_Amount": "Total_Outstanding_Amount",
        "EMI_Amount": "Total_EMI_Amount",
        "Payment_Delay_Days": "Avg_Payment_Delay",
        "Customer_ID": "Customer_Count"
    }, inplace=True)

    # --- Write aggregated data to Gold layer ---
    write_parquet_to_s3(agg_df, BUCKET_NAME, f"{GOLD_PREFIX}region_summary/data.parquet")

    return {
        "status": "Success",
        "message": f"Aggregated region summary saved to Gold layer in {BUCKET_NAME}",
        "record_count": len(agg_df)
    }
