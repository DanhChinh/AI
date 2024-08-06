import pandas as pd
import os

def readData(csv_file="data.csv"):
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file, index_col=0)
    df = pd.DataFrame(columns=[f"KQ[{i}]" for i in range(27)])
    df.to_csv(csv_file)
    return df
def readDataL2(csv_file="dataTrain.csv"):
    return pd.read_csv(csv_file, index_col=0).to_numpy()

def saveData(data, csv_file="data.csv"):
    data.to_csv(csv_file)
def sortData(data):
    data.index = pd.to_datetime(data.index, format='%d-%m-%Y')   
    data = data.sort_index()
    data.index = data.index.strftime('%d-%m-%Y')
    return data
def readUniqueData(csv_file="data.csv"):
    return pd.read_csv(csv_file, index_col=0).drop_duplicates()

# def checkUnique():
#     len_data = len(readData())
#     len_uniquedata  = len(readUniqueData())
#     print(f"Check unique: {len_data}-{len_uniquedata}={len_data-len_uniquedata}")
def convertToL2(uniqueData):
    indexs = uniqueData.index
    columns = uniqueData.columns
    for index in indexs:
        for column in columns:
            uniqueData.loc[index, column] = uniqueData.loc[index, column]%100       
    return uniqueData

