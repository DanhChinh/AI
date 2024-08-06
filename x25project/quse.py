
from getTextTimes import getTextTimeNow
from qclass import arrToState
import numpy as np
import pandas as pd
import os, sys, json

def readJson(file_path='qcookie.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["date"], int(data["hour"]), int(data["minutes"]), data["result"]
def saveJson(file_path='qcookie.json'):
    from getKq import getKq
    [strhour, strminutes,_]  = getTextTimeNow("%H:%M:%S").split(":")
    textTime = getTextTimeNow()
    data = {
        "date": textTime,
        "hour": int(strhour),
        "minutes": int(strminutes),
        "result": getKq(textTime)
    }
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Tạo tệp mới: {file_path} với nội dung {data}")



file_path = 'qcookie.json'
[strhour, strminutes,_]  = getTextTimeNow("%H:%M:%S").split(":")
textTime = getTextTimeNow()
qk  = []

if os.path.exists(file_path):
    jsonTextTime, hour, minutes, kq = readJson()
    if jsonTextTime!=textTime or hour>19:
        saveJson()
        jsonTextTime, hour, minutes, kq = readJson()
else:
    saveJson(file_path)
print(kq)
q_table = pd.read_csv('q_table.csv',index_col=0).astype(float)
print(q_table)
print("now:", textTime) #06-08-2024
#kq = getKq(textTime) ['82239', '25739', '93992', '38897', '89"]
kq = np.array(kq).astype(int)#[82239, 25739, 93992, 38897, 89]
for i in range(len(kq)):
    kq[i] = kq[i]%100
#[39, 39, 92, 97, 89]
state = arrToState(kq) #
print(state)
if state not in q_table.index:
    sys.exit("state not in q_table")
columns = q_table.columns
for column in columns:
    print(column, q_table.loc[state, column])
print("______________")
print(f"action: {q_table.loc[state].idxmax()}\tmaxpoint: {q_table.loc[state].max()}")