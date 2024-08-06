from handleFrame import convertToL2, readUniqueData, saveData


uniqueFrameL2 = convertToL2(readUniqueData())

dataTest = uniqueFrameL2.tail(30)
dataTrain = uniqueFrameL2.head(-30)

saveData(dataTest, "dataTest.csv")
saveData(dataTrain, "dataTrain.csv")