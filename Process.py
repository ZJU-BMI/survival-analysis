import numpy as np
import pandas as pd
import csv


def get_all_patients_features():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_特征.csv'
    df=pd.read_csv(file,header=0,sep=',',engine='python',usecols=[0])
    patientIdList = []
    for i in df.values:
        if i not in patientIdList:
            for x in i:
                j = str(x).split(',')
                patientIdList += j
    print(len(patientIdList))
    print(patientIdList)

    with open(file,'r') as myFile:
        lines = csv.reader(myFile)
        next(lines, None)
        i = 0
        temp = []
        allPatientFeatures = []  # 3 dims
        onePatientFeatures = [] # 2 dims
        allPatientFeaturesArrays = None
        onePatientFeaturesArrays = None
        for line in lines:
            print(line)
            patientId = line[0]
            visitId = line[1]
            feature = list(map(int,line[2:]))
            featureArray = np.array(feature)
            print(featureArray.shape)
            if patientId != patientIdList[i]:
                    i += 1
                    print(len(allPatientFeatures))
                    allPatientFeatures.append(onePatientFeatures[:-1])
                    onePatientFeaturesArraysTemp = np.array(onePatientFeatures[:-1])
                    onePatientFeaturesArrays = np.pad(onePatientFeaturesArraysTemp,((0,42-onePatientFeaturesArraysTemp.shape[0]),(0,0)),'constant')
                    temp.append(onePatientFeaturesArrays)
                    allPatientFeaturesArrays = np.array(temp)
                    onePatientFeatures = []
                    onePatientFeatures.append(feature)

                    print(onePatientFeaturesArraysTemp.shape)
            else:
                onePatientFeatures.append(feature)
        allPatientFeatures.append(onePatientFeatures)   # get all patients features
        temp.append(onePatientFeaturesArrays)
        allPatientFeaturesArrays = np.array(temp)
        np.save("allPatientFeatures1.npy",allPatientFeaturesArrays)
        # allPatientFeaturesArrays = np.dstack((allPatientFeaturesArrays,onePatientFeaturesArrays))

        print(len(allPatientFeatures))
        print(allPatientFeatures)


def get_all_patients_labels():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_标签.csv'
    df = pd.read_csv(file, header=0, sep=',', engine='python', usecols=[0])
    patientIdList = []
    for i in df.values:
        if i not in patientIdList:
            for x in i:
                j = str(x).split(',')
                patientIdList += j
    print(len(patientIdList))
    print(patientIdList)
    with open(file, 'r') as myFile:
        lines = csv.reader(myFile)
        next(lines, None)
        i = 0
        temp = []
        allPatientLabels = []  # 3 dims
        onePatientLabels = []  # 2 dims
        allPatientLabelsArrays = None
        onePatientLabelsArrays = None
        for line in lines:
            patientId = line[0]
            label = list(map(int, line[2:]))
            labelArray = np.array(label)
            print(labelArray.shape)
            if patientId != patientIdList[i]:
                i += 1
                print(len(allPatientLabels))
                allPatientLabels.append(onePatientLabels[:-1])
                onePatientLabelsArraysTemp = np.array(onePatientLabels[:-1])
                onePatientLabelsArrays = np.pad(onePatientLabelsArraysTemp,
                                                  ((0, 42 - onePatientLabelsArraysTemp.shape[0]), (0, 0)), 'constant')
                temp.append(onePatientLabelsArrays)
                allPatientLabelsArrays = np.array(temp)
                onePatientLabels = []
                onePatientLabels.append(label)

                print(onePatientLabelsArraysTemp.shape)
            else:
                onePatientLabels.append(label)
        allPatientLabels.append(onePatientLabels)  # get all patients features
        temp.append(onePatientLabelsArrays)
        allPatientLabelsArrays = np.array(temp)
        np.save("allPatientLabels1.npy", allPatientLabelsArrays)
        # allPatientFeaturesArrays = np.dstack((allPatientFeaturesArrays,onePatientFeaturesArrays))

        print(len(allPatientLabels))
        print(allPatientLabels)

def read_features():
    features = np.load("allPatientFeatures1.npy")
    print(features.shape)

def read_labels():
    labels = np.load("allPatientLabels1.npy")
    print(labels)

def get_logistic_features():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_特征.csv'
    data = pd.read_csv(file,header=0,sep=',', engine='python')
    val = data.values
    allPatientFeatures = float(val[:,2:])
    print(allPatientFeatures.shape)
    np.save("logistic_features.npy",allPatientFeatures)


def get_logistic_labels():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_标签.csv'
    data = pd.read_csv(file,header=0,sep=',', engine='python')
    val = data.values
    allPatientLabels = int(val[:,-1])
    print(allPatientLabels.shape)
    np.save("logistic_labels.npy",allPatientLabels)
if __name__ == '__main__':
    # get_all_patients_features()
    # get_all_patients_labels()
    # read_features()
    read_labels()
    # get_logistic_features()
    # get_logistic_labels()