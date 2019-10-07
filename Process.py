import numpy as np
import pandas as pd
import csv


def get_all_patients_features():
    file = 'E:\\survival analysis\\resources\\use.csv'
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
        np.save("allPatientFeatures_now.npy",allPatientFeaturesArrays)
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
        np.save("allPatientLabels.npy", allPatientLabelsArrays)
        # allPatientFeaturesArrays = np.dstack((allPatientFeaturesArrays,onePatientFeaturesArrays))

        print(len(allPatientLabels))
        print(allPatientLabels)


def read_features():
    features = np.load("allPatientFeatures1.npy")
    features = features.astype(np.float32)
    time_interval = features[:,:,10].reshape(-1)
    max_value = np.max(time_interval)
    for i in range(len(time_interval)):
        time_interval[i] =time_interval[i]/max_value
    time_interval = time_interval.reshape(-1,42)
    features[:,:,10] = time_interval
    np.save('allPatientFeatures1.npy',features)
    print(features.shape)


def read_labels():
    features = np.load("allPatientFeatures1.npy")
    features = features[0:2100,0:5,:].reshape([-1,200])
    np.savetxt('allPatientFeatures1.csv',features,delimiter=',')
    labels = np.load("allPatientLabels1.npy")
    labels = labels[0:2100,0:5,-1].reshape([-1,1])
    np.savetxt('allPatientLabels1.csv',labels,delimiter=',')
    print(np.sum(labels==1))
    print(labels)


def read_stage():
    stages = np.load("all_patient_stage.npy")
    print(stages)


def read_score():
    score = np.load("all_patient_score.npy")
    print(score)


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


# 找到不同stage对应的病人信息，并根据这些病人信息的feature weight 寻找排名前十的特征
def get_patient_in_stage():
    stages = np.load("all_patient_stage_ave.npy")
    all_patient_weights = np.load("allPatientFeatures1.npy")[0:2100,0:5,:]
    patient_in_stage0 = np.zeros(shape=(0,200),dtype=np.int32)
    patient_in_stage1 = np.zeros(shape=(0,200),dtype=np.int32)
    patient_in_stage2 = np.zeros(shape=(0,200),dtype=np.int32)
    patient_in_stage3 = np.zeros(shape=(0,200),dtype=np.int32)
    patient_in_stage4 = np.zeros(shape=(0,200),dtype=np.int32)

    for patient in range(stages.shape[0]):
        for visit in range(stages.shape[1]):
            stage = stages[patient,visit]
            weight = all_patient_weights[patient, visit,:].reshape(-1,200)
            if stage == 0:
                patient_in_stage0 = np.concatenate((patient_in_stage0,weight))
            if stage == 1:
                patient_in_stage1 = np.concatenate((patient_in_stage1, weight))
            if stage == 2:
                patient_in_stage2 = np.concatenate((patient_in_stage2, weight))
            if stage == 3:
                patient_in_stage3 = np.concatenate((patient_in_stage3, weight))
            if stage == 4:
                patient_in_stage4 = np.concatenate((patient_in_stage4, weight))

    print(patient_in_stage0.shape[0])
    print(patient_in_stage1.shape[0])
    print(patient_in_stage2.shape[0])
    print(patient_in_stage3.shape[0])
    print(patient_in_stage4.shape[0])

    stage_0_mean = np.mean(patient_in_stage0,axis=0).reshape(-1)
    stage_1_mean = np.mean(patient_in_stage1,axis=0).reshape(-1)
    stage_2_mean = np.mean(patient_in_stage2,axis=0)
    stage_3_mean = np.mean(patient_in_stage3,axis=0)
    stage_4_mean = np.mean(patient_in_stage4,axis=0)

    stage_0_max_index = np.argsort(stage_0_mean)
    stage_1_max_index = np.argsort(stage_1_mean)
    stage_2_max_index = np.argsort(stage_2_mean)
    stage_3_max_index = np.argsort(stage_3_mean)
    stage_4_max_index = np.argsort(stage_4_mean)
    print(stage_0_max_index)

if __name__ == '__main__':
    # get_all_patients_features()
    # get_all_patients_labels()
    # read_features()
    # read_labels()
    get_patient_in_stage()
    # get_logistic_features()
    # get_logistic_labels()