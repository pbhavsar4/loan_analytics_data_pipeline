AWS Serverless Medallion Data Pipeline Architecture:

Upload data to Raw S3 bucket bronze folder
Triggers Lambda which reads, transforms data
 Write transformed data to S3 Silver folder (Parquet)
 Triggers Lambda and aggregates 
Write transformed data to S3 Gold folder (Parquet)
Data is Copied to RDS


<img width="1256" height="602" alt="image" src="https://github.com/user-attachments/assets/9a64321c-7852-4a5e-ba56-a573cc8ff185" />

