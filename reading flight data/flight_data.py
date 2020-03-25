from math import *

import pandas as pd
import matplotlib.pyplot as plt

Data_converted = pd.read_excel('../converted_data.xlsx', usecols=[1,2,3,4,5,6,7,8,9])
Data = pd.read_excel('FD_AOA.xlsx')
Data = Data.dropna()


Lift_Drag_Polar = Data.loc[Data['display_active_screen'] == 3]
Short_Period = Data.loc[Data['display_active_screen'] == 4]
Phugoid = Data.loc[Data['display_active_screen'] == 5]
Dutch_Roll = Data.loc[Data['display_active_screen'] == 6]
Aperiodic_Roll = Data.loc[Data['display_active_screen'] == 7]
Roll_Stability = Data.loc[Data['display_active_screen'] == 8]
Elevator_Trim_Curve = Data.loc[Data['display_active_screen'] == 9]

plt.scatter(Data['longitude'], Data['lat'], s=0.6)
plt.show()

'''
for i in range(len(Short_Period)):
    if Short_Period['time'][i+1]-Short_Period['time'][i] != 0.1:
        idx = i
'''

pd.DataFrame.to_excel(Lift_Drag_Polar, 'type_of_motion/lift_drag_polar.xlsx')
pd.DataFrame.to_excel(Short_Period, 'type_of_motion/short_period.xlsx')
pd.DataFrame.to_excel(Phugoid, 'type_of_motion/phugoid.xlsx')
pd.DataFrame.to_excel(Dutch_Roll, 'type_of_motion/dutch_roll.xlsx')
pd.DataFrame.to_excel(Aperiodic_Roll, 'type_of_motion/aperiodic_roll.xlsx')
pd.DataFrame.to_excel(Roll_Stability, 'type_of_motion/roll_stability.xlsx')
pd.DataFrame.to_excel(Elevator_Trim_Curve, 'type_of_motion/elevator_trim_curve.xlsx')
print('done')


