from math import *

import pandas as pd
import matplotlib.pyplot as plt

Data_converted = pd.read_excel('../aerodynamic_properties/converted_data.xlsx', usecols=[1,2,3,4,5,6,7,8,9])
Data = pd.read_excel('FD_AOA.xlsx')

time = list(Data_converted['time'])
th0 = []

for i in range(len(time)):    
    idx = Data[Data['time']==time[i]].index.values[0]
    th0.append(Data['Pitch_angle'][idx])
    
    


#Data_converted['time'][i]
t = Data['time']
alt = Data['fuel_used_l']

plt.scatter(t,alt)
plt.show()