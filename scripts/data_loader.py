import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def load_all_years():
    blobs = container_client.list_blobs()
    
    dfs = []

    for blob in blobs:
        if blob.name.endswith(".csv"):
            blob_client = container_client.get_blob_client(blob)
            data = blob_client.download_blob().readall()
            
            df = pd.read_csv(pd.io.common.BytesIO(data), encoding="latin1")
            df["source_file"] = blob.name  # track year
            
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    df = load_all_years()
    print(df.shape)
    print(df.columns)