import os
import pandas as pd
import pymysql
import sqlalchemy


root = "RBI_Data"
supported_formats = ["XLS","XLSX"]
metric = "NEFT"
neft_headers = ['Sr. No','Bank Name','No. Of Outward Transactions','Amount(Outward)','No. Of Inward Transactions','Amount(Inward)']

root_path = os.path.join(os.getcwd(),root)
os.chdir(root_path)
folders  = [year for year in os.listdir() if year.isdigit()]

user = "root"
password = "2102005"
port = 3306
host = "localhost"

conn = pymysql.connect(
    user = user,
    password = password,
    port = port,
    host = host,
    charset="utf8mb4",
    autocommit=True
)

try:
    with conn.cursor() as cursor:
        cursor.execute("SHOW DATABASES LIKE 'rbi_metric_%';")
        databases = cursor.fetchall()
        
        for (db_name,) in databases:
            drop_query = f"DROP DATABASE `{db_name}`;"
            cursor.execute(drop_query)
        
        new_db_name = f"rbi_metric_{metric}"
        create_query = f"CREATE DATABASE `{new_db_name}`;"
        cursor.execute(create_query)

    print(f"Database Created: {new_db_name}")
        
except Exception as e:
    print(e)

print(f"mysql+pymysql://{user}:{password}@{host}/{new_db_name}")
engine = sqlalchemy.create_engine(f"mysql+pymysql://{user}:{password}@{host}/{new_db_name}")


try:
    with engine.connect() as connection:
        
        for folder in folders:
            current_path = os.path.join(root_path,folder)
            os.chdir(current_path)
            for year_data in os.listdir():
                year = year_data.split(".")[0]
                extension = year_data.split(".")[-1]
                if extension == "XLS":
                    #df = pd.read_excel(year_data,engine="xlrd",sheet_name="NEFT",header=None)
                    engine = "xlrd"
                elif extension == "XLSX":
                    #df = pd.read_excel(year_data,engine="openpyxl",sheet_name="NEFT",header=None)
                    engine = "openpyxl"
                elif extension not in supported_formats:
                    print("File Format Not Supported")
                    continue      
                df = pd.read_excel(year_data,engine=engine,sheet_name="NEFT",header=None)
                search_area = df.head()
                row_idx = df.where((search_area == "Sr. No.") | (search_area == "Sr. No")| (search_area == "Sr.No")).stack().index[0][0]
                #replace later with .isfind and .idxmax, accoutn for all possible permutatios of the col name we are searching for

                df = df.iloc[row_idx+2:-2,:]
                df = df.dropna(axis=1, how='all')
                df.columns = neft_headers
                df = df.reset_index(drop=True)

                df.to_sql(f"{metric}_{year}_{folder}",con=connection,if_exists="replace",index=False)
except Exception as e:
    print(e)
