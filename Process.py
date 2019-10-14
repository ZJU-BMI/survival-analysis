import numpy as np
import pandas as pd
import csv
import MySQLdb


# 得到数据的feature 并将每个病人数据padding 成 42 次入院记录（2）
def get_all_patients_features():
    file = 'E:\\survival analysis\\resources\\合并特征值之后的特征.csv'
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
                    # allPatientFeatures.append(onePatientFeatures[:-1])
                    allPatientFeatures.append(onePatientFeatures)
                    onePatientFeaturesArraysTemp = np.array(onePatientFeatures)
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
        np.save("allPatientFeatures_merge.npy",allPatientFeaturesArrays)
        # allPatientFeaturesArrays = np.dstack((allPatientFeaturesArrays,onePatientFeaturesArrays))

        print(len(allPatientFeatures))
        print(allPatientFeatures)

# （3）得到数据的label 并将每个patient visit padding into 42  次入院记录
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
                # allPatientLabels.append(onePatientLabels[:-1])
                # onePatientLabelsArraysTemp = np.array(onePatientLabels[:-1])
                allPatientLabels.append(onePatientLabels)
                onePatientLabelsArraysTemp = np.array(onePatientLabels)
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
        np.save("allPatientLabels_merge.npy", allPatientLabelsArrays)
        # allPatientFeaturesArrays = np.dstack((allPatientFeaturesArrays,onePatientFeaturesArrays))

        print(len(allPatientLabels))
        print(allPatientLabels)

# （3）将之前的200个特征 去除时间差 心功能一级 心功能二级 心功能三级 心功能四级  改成195个特征
def get_right_data():
    features = np.load("allPatientFeatures_right.npy")
    time = features[:,:,10]
    np.savetxt('time.csv',time,delimiter=',')
    features1 = features[:,:,0:3]
    features2 = features[:,:,7:10]
    features3 = features[:,:,11:]
    features_concentrate = np.concatenate((features1,np.concatenate((features2,features3),axis=2)),axis=2)
    print(features_concentrate.shape)
    np.save("allPatientFeatures_right_1.npy",features_concentrate)

# TODO：将数据整理成没有二值化的数据（采样需要重新整理）
def get_cox_data():
    features = np.load("pick_5_visit_features.npy")
    features1 = features[:,:,0:3]
    features2 = features[:,:,7:]
    features = np.concatenate((features1,features2),axis=2)
    print(features.shape)
    np.save("allPatientFeatures_right_cox.npy",features)
# 将时间归一化
def read_features():
    features = np.load("allPatientFeatures_right.npy")
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
    all_patient_weights = np.load("average_weight.npy")
    labels =np.load("pick_5_visit_labels.npy")
    patient_in_stage0 = np.zeros(shape=(0,195),dtype=np.int32)
    patient_in_stage1 = np.zeros(shape=(0,195),dtype=np.int32)
    patient_in_stage2 = np.zeros(shape=(0,195),dtype=np.int32)
    patient_in_stage3 = np.zeros(shape=(0,195),dtype=np.int32)
    patient_in_stage4 = np.zeros(shape=(0,195),dtype=np.int32)

    patient_in_stage0_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage1_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage2_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage3_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage4_labels = np.zeros(shape=(0, 1), dtype=np.int32)

    for patient in range(stages.shape[0]):
        for visit in range(stages.shape[1]):
            stage = stages[patient,visit]
            weight = all_patient_weights[patient, visit,:].reshape(-1,195)
            label = labels[patient,visit,:].reshape(-1,1)
            if stage == 0:
                patient_in_stage0 = np.concatenate((patient_in_stage0,weight))
                patient_in_stage0_labels = np.concatenate((patient_in_stage0_labels,label))
            if stage == 1:
                patient_in_stage1 = np.concatenate((patient_in_stage1, weight))
                patient_in_stage1_labels = np.concatenate((patient_in_stage1_labels,label))
            if stage == 2:
                patient_in_stage2 = np.concatenate((patient_in_stage2, weight))
                patient_in_stage2_labels = np.concatenate((patient_in_stage2_labels, label))
            if stage == 3:
                patient_in_stage3 = np.concatenate((patient_in_stage3, weight))
                patient_in_stage3_labels = np.concatenate((patient_in_stage3_labels,label))
            if stage == 4:
                patient_in_stage4 = np.concatenate((patient_in_stage4, weight))
                patient_in_stage4_labels = np.concatenate((patient_in_stage4_labels,label))

    print(patient_in_stage0.shape[0])
    print(patient_in_stage1.shape[0])
    print(patient_in_stage2.shape[0])
    print(patient_in_stage3.shape[0])
    print(patient_in_stage4.shape[0])

    print(patient_in_stage0_labels.shape[0])
    print(patient_in_stage1_labels.shape[0])
    print(patient_in_stage2_labels.shape[0])
    print(patient_in_stage3_labels.shape[0])
    print(patient_in_stage4_labels.shape[0])

    death_rate = {}
    # death_rate['stage0'] = float(len(np.where(patient_in_stage0_labels==1)[0])/patient_in_stage0_labels.shape[0])
    # death_rate['stage1'] = len(np.where(patient_in_stage1_labels==1)[0])/patient_in_stage1_labels.shape[0]
    # death_rate['stage2'] = len(np.where(patient_in_stage2_labels==1)[0])/patient_in_stage2_labels.shape[0]
    # death_rate['stage3'] = len(np.where(patient_in_stage3_labels==1)[0])/patient_in_stage3_labels.shape[0]
    # death_rate['stage4'] = len(np.where(patient_in_stage4_labels==1)[0])/patient_in_stage4_labels.shape[0]


    stage_0_mean = np.mean(patient_in_stage0,axis=0).reshape(-1)
    stage_1_mean = np.mean(patient_in_stage1,axis=0).reshape(-1)
    stage_2_mean = np.mean(patient_in_stage2,axis=0)
    stage_3_mean = np.mean(patient_in_stage3,axis=0)
    stage_4_mean = np.mean(patient_in_stage4,axis=0)
    dataframe = pd.DataFrame({"stage0":stage_0_mean,"stage1":stage_1_mean,"stage2":stage_2_mean,"stage3":stage_3_mean,"stage4":stage_4_mean})
    dataframe.to_csv("stage_weights_1.csv",index=False,sep=",")

    stage_0_max_index = np.argsort(stage_0_mean)
    stage_1_max_index = np.argsort(stage_1_mean)
    stage_2_max_index = np.argsort(stage_2_mean)
    stage_3_max_index = np.argsort(stage_3_mean)
    stage_4_max_index = np.argsort(stage_4_mean)
    dataframe2 = pd.DataFrame({"stage0": stage_0_max_index, "stage1": stage_1_max_index, "stage2": stage_2_max_index,
                               "stage3": stage_3_max_index, "stage4": stage_4_max_index})
    dataframe2.to_csv("前十的weight.csv", index=False, sep=",")

    print(stage_0_max_index)


# （1 ）将最后一次的数据加上作为预测-修改最后一次入院的错误标签
def get_right_label():
    file = 'E:\\survival analysis\\resources\\1-1.csv'
    df = pd.read_csv(file, header=0, sep=',', engine='python', usecols=[0])
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
        right_labels = []
        death_label_last = -1
        for line in lines:
            print(line)
            patientId_now = line[0]
            death_label_now = int(line[2])
            two_year_label_now = int(line[3])
            if(patientId_now == patientIdList[i]):
                right_labels.append(two_year_label_now)
            else:
                i = i+1
                right_labels.pop()
                right_labels.append(death_label_last)
                right_labels.append(two_year_label_now)
            death_label_last = death_label_now
        print(len(right_labels))
        dataframe = pd.DataFrame({'right_label':right_labels})
        dataframe.to_csv('right_label.csv',index=False,sep=',')
        return right_labels

def get_insert_sql():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_特征.csv'
    data = pd.read_csv(file, header=0, sep=',', engine='python')
    val = data.values
    patientInfo = val[:,0:2]
    sql_inserts = ""
    for i in range(patientInfo.shape[0]):
        values = "VALUES"+"("+ "'"+ str(patientInfo[i,0])+"'"+","+"'" +str(patientInfo[i,1])+"'"+")"
        sql = "INSERT into hf_stop.patients "+ values+";"
        print(sql)
        sql_inserts += sql
    with open("1.sql",'w') as file:
        file.write(sql_inserts)
    return sql_inserts

#  将数据的入院记录时长保存
def read_time():
    features = np.load("allPatientFeatures_right.npy")
    mask = np.sign(np.max(np.abs(features),2))
    length = np.sum(mask,1)
    time = []
    for patient in range(features.shape[0]):
        patient_last_time = features[patient,length[patient]-1,10]
        time.append(patient_last_time)
    # np.savetxt("last_time.csv",time,delimiter=',')
    np.savetxt("length.csv",length,delimiter=",")


def read_features1():
    fetaures= np.load("allPatientFeatures_merge.npy")
    print(fetaures.shape)
if __name__ == '__main__':
    # get_all_patients_features()
    # get_all_patients_labels()
    # read_features()
    # read_labels()
    read_features1()
    get_all_patients_labels()
    # get_logistic_features()
    # get_logistic_labels()