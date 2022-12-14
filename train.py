#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle_compat
pickle_compat.patch()
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import pickle
from sklearn.svm import SVC


# In[2]:


def getMealNoMealTime(newData,time):
    mealData=[]
    time1 = newData[0:len(newData)-1]
    time2 = newData[1:len(newData)]
    difference = list(np.array(time1) - np.array(time2))
    Values = list(zip(time1, time2, difference))
    for j in Values:
        if j[2]<time:
            mealData.append(j[0])
    return mealData


# In[11]:



def getMealNomealData(MealTime,S_Time,E_Time,isMealData,GlucoseValues):
    NewMealData = []
    
    for Time in MealTime:
        MealData0= GlucoseValues[GlucoseValues['datetime'].between(Time+ pd.DateOffset(hours=S_Time),Time + pd.DateOffset(hours=E_Time))]
        if MealData0.shape[0] <24:
            continue
        GlucoseData = MealData0['Sensor Glucose (mg/dL)'].to_numpy()
        mean = MealData0['Sensor Glucose (mg/dL)'].mean()
        if isMealData:
            MissingValues = 30 - len(GlucoseData)
            if MissingValues > 0:
                for i in range(MissingValues):
                    GlucoseData = np.append(GlucoseData, mean)
            NewMealData.append(GlucoseData[0:30])
        else:
            NewMealData.append(GlucoseData[0:24])
    return pd.DataFrame(data=NewMealData)


# In[4]:



def Data_alter(InsulinData,GlucoseData):
  M_Data = pd.DataFrame()
  No_M_Data = pd.DataFrame()
  InsulinData= InsulinData[::-1]
  GlucoseData= GlucoseData[::-1]
  GlucoseData['Sensor Glucose (mg/dL)'] = GlucoseData['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
  InsulinData['datetime'] = pd.to_datetime(InsulinData["Date"].astype(str) + " " + InsulinData["Time"].astype(str))
  GlucoseData['datetime'] = pd.to_datetime(GlucoseData["Date"].astype(str) + " " + GlucoseData["Time"].astype(str))
  
  NewData_I = InsulinData[['datetime','BWZ Carb Input (grams)']]
  NewData_G = GlucoseData[['datetime','Sensor Glucose (mg/dL)']]
  
  NewData_I = NewData_I[(NewData_I['BWZ Carb Input (grams)'].notna()) & (NewData_I['BWZ Carb Input (grams)']>0) ]
  
  new_T = list(NewData_I['datetime'])
  
  MealData=[]
  NoMealData =[]
  MealData = getMealNoMealTime(new_T,pd.Timedelta('0 days 120 min'))
  NoMealData = getMealNoMealTime(new_T,pd.Timedelta('0 days 240 min'))
  
  Data_M = getMealNomealData(MealData,-0.5,2,True,NewData_G)
  Data_NM = getMealNomealData(NoMealData,2,4,False,NewData_G)

  Features_Meal = features_Glucose(Data_M)
  Features_NoMeal = features_Glucose(Data_NM)
  
  
  stdScaler = StandardScaler()
  STD_Meal = stdScaler.fit_transform(Features_Meal)
  STD_NoMeal = stdScaler.fit_transform(Features_NoMeal)
  
  pca = PCA(n_components=5)
  pca.fit(STD_Meal)
       
  PCA_Meal = pd.DataFrame(pca.fit_transform(STD_Meal))
  PCA_NoMeal = pd.DataFrame(pca.fit_transform(STD_NoMeal))
  
  PCA_Meal['class'] = 1
  PCA_NoMeal['class'] = 0
  
  Data1 = PCA_Meal.append(PCA_NoMeal)
  Data1.index = [i for i in range(Data1.shape[0])]
  return Data1


# In[5]:


def fn_zero_crossings(row, xAxis):
    slopes = [
     0]
    zero_cross = list()
    zero_crossing_rate = 0
    X = [i for i in range(xAxis)][::-1]
    Y = row[::-1]
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))

    for index in range(0, len(slopes) - 1):
        if slopes[index] * slopes[(index + 1)] < 0:
            zero_cross.append([slopes[(index + 1)] - slopes[index], X[(index + 1)]])

    zero_crossing_rate = np.sum([np.abs(np.sign(slopes[(i + 1)]) - np.sign(slopes[i])) for i in range(0, len(slopes) - 1)]) / (2 * len(slopes))
    if len(zero_cross) > 0:
        return [max(zero_cross)[0], zero_crossing_rate]
    else:
        return [
         0, 0]
    


# In[6]:


def AbsMean(param):
    Mean = 0
    for p in range(0, len(param) - 1):
        Mean = Mean + np.abs(param[(p + 1)] - param[p])
    return Mean / len(param)


# In[7]:


def Entropy(param):
    paramLen = len(param)
    EntropyValue = 0
    if paramLen <= 1:
        return 0
    else:
        value, count = np.unique(param, return_counts=True)
        ratio = count / paramLen
        nonZero_ratio = np.count_nonzero(ratio)
        if nonZero_ratio <= 1:
            return 0
        for i in ratio:
            EntropyValue -= i * np.log2(i)
        return EntropyValue


# In[8]:


def RootMeanSquare(param):
    RootMeanSquare = 0
    for p in range(0, len(param) - 1):
        
        RootMeanSquare = RootMeanSquare + np.square(param[p])
    return np.sqrt(RootMeanSquare / len(param))


# In[9]:


def FastFourier(param):
    FastFourier = fft(param)
    paramLen = len(param)
    t = 2/300
    amplitude = []
    frequency = np.linspace(0, paramLen * t, paramLen)
    for amp in FastFourier:
        amplitude.append(np.abs(amp))
    sortedAmplitude = amplitude
    sortedAmplitude = sorted(sortedAmplitude)
    Amplitude_max = sortedAmplitude[(-2)]
    Frequency_max = frequency.tolist()[amplitude.index(Amplitude_max)]
    return [Amplitude_max, Frequency_max]


# In[12]:


def features_Glucose(Data_meal_NoMeal):
    features_Glucose=pd.DataFrame()
    for i in range(0, Data_meal_NoMeal.shape[0]):
        param = Data_meal_NoMeal.iloc[i, :].tolist()
        features_Glucose = features_Glucose.append({ 
         'Minimum Value':min(param), 
         'Maximum Value':max(param),
         'Mean of Absolute Values1':AbsMean(param[:13]), 
         'Mean of Absolute Values2':AbsMean(param[13:]),  
         'Root Mean Square':RootMeanSquare(param),
         'Entropy':RootMeanSquare(param), 
         'Max FFT Amplitude1':FastFourier(param[:13])[0], 
         'Max FFT Frequency1':FastFourier(param[:13])[1], 
         'Max FFT Amplitude2':FastFourier(param[13:])[0], 
         'Max FFT Frequency2':FastFourier(param[13:])[1]},
          ignore_index=True)
    return features_Glucose

   
if __name__=='__main__':
    InsulinData1=pd.read_csv("Insulin_patient2.csv")
    GlucoseData1=pd.read_csv("CGM_patient2.csv")
    InsulinData2=pd.read_csv("InsulinData.csv",low_memory=False)
    GlucoseData2=pd.read_csv("CGMData.csv",low_memory=False)
    Insulin=pd.concat([InsulinData1,InsulinData2])
    Glucose=pd.concat([GlucoseData1,GlucoseData2])
    Data1= Data_alter(Insulin,Glucose)
    X = Data1.iloc[:, :-1]
    Y = Data1.iloc[:, -1]
    
    model = SVC(kernel='linear',C=1,gamma=0.1)
    kfold = KFold(5, True, 1)
    for tr, tst in kfold.split(X, Y):
        X_train, X_test = X.iloc[tr], X.iloc[tst]
        Y_train, Y_test = Y.iloc[tr], Y.iloc[tst]
        
        model.fit(X_train, Y_train)

    with open('RF_Model.pkl', 'wb') as (file):
        pickle.dump(model, file)





# In[ ]:




