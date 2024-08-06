from getTextTimes import getTextTimes
from handleFrame import readData, saveData, sortData
from getKq import getKq
data = readData("data.csv")
textTimes = getTextTimes('01-01-2017', '30-07-2024')

for textTime in textTimes:
    if textTime not in data.index:
        try:
            new_data = getKq(textTime)
            if new_data:
                data.loc[textTime] = new_data
                print(textTime, "da duoc them")
            else:
                print(textTime,"khong chua du lieu")
        except:
            saveData(data, "data.csv")
    else:
        print(textTime,"da ton tai")

data = sortData(data)
saveData(data, "data.csv")
