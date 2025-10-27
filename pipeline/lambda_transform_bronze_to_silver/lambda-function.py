import os
import boto3
import awswrangler as wr
import pandas as pd

s3 = boto3.client("s3")

BUCKET_NAME = os.environ["BUCKET_NAME"]
BRONZE_PREFIX = os.environ["BRONZE_PREFIX"]
SILVER_PREFIX = os.environ["SILVER_PREFIX"]
ATHENA_DATABASE = os.environ.get("ATHENA_DATABASE", "loan_analytics")

def lambda_handler(event, context):
    bronze_path = f"s3://{BUCKET_NAME}/{BRONZE_PREFIX}"
    df = wr.s3.read_csv(bronze_path)

    df["Loan_Amount"] = df["Loan_Amount"].astype(float)
    df["Outstanding_Amount"] = df["Outstanding_Amount"].astype(float)
    df["EMI_Amount"] = df["EMI_Amount"].astype(float)
    df["Payment_Delay_Days"] = df["Payment_Delay_Days"].astype(int)
    df["Customer_Score"] = df["Customer_Score"].astype(int)

    df = df[df["Loan_Amount"] > 0]

    df["Payment_On_Time_Flag"] = df["Payment_Delay_Days"].apply(lambda x: "Yes" if x == 0 else "No")
    df["Payment_Behavior"] = df["Payment_Delay_Days"].apply(
        lambda x: "Good" if x == 0 else ("Average" if x <= 10 else "Delayed")
    )

    dim_customer = df[[
        "Customer_ID","Name","Region","Contact_Number","Email","Customer_Score","Risk_Level"
    ]].drop_duplicates(subset=["Customer_ID"])

    wr.s3.to_parquet(
        df=dim_customer,
        path=f"s3://{BUCKET_NAME}/{SILVER_PREFIX}dim_customer/",
        dataset=True,
        database=ATHENA_DATABASE,
        table="dim_customer"
    )

    dim_account = df[[
        "Account_Number","Account_Type","Loan_Type"
    ]].drop_duplicates(subset=["Account_Number"])

    wr.s3.to_parquet(
        df=dim_account,
        path=f"s3://{BUCKET_NAME}/{SILVER_PREFIX}dim_account/",
        dataset=True,
        database=ATHENA_DATABASE,
        table="dim_account"
    )

    fact_loan_payment = df[[
        "Customer_ID","Account_Number","Loan_Amount","Outstanding_Amount","EMI_Amount",
        "Due_Date","Payment_Status","Last_Payment_Date","Payment_Delay_Days",
        "Payment_On_Time_Flag","Payment_Behavior"
    ]]

    wr.s3.to_parquet(
        df=fact_loan_payment,
        path=f"s3://{BUCKET_NAME}/{SILVER_PREFIX}fact_loan_payment/",
        dataset=True,
        database=ATHENA_DATABASE,
        table="fact_loan_payment"
    )

    return {"status": "Success", "message": "Bronze → Silver ETL completed."}
