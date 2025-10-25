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

def read_parquet_from_s3(bucket, key):
    """Read Parquet from S3 and return a Pandas DataFrame."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    buffer = BytesIO(obj["Body"].read())
    table = pq.read_table(buffer)
    return table.to_pandas()

def write_parquet_to_s3(df, bucket, key):
    """Write a Pandas DataFrame to Parquet in S3."""
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

def lambda_handler(event, context):
    # --- Read Silver layer tables ---
    fact_df = read_parquet_from_s3(BUCKET_NAME, f"{SILVER_PREFIX}fact_loan_payment/")
    customer_df = read_parquet_from_s3(BUCKET_NAME, f"{SILVER_PREFIX}dim_customer/")

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
