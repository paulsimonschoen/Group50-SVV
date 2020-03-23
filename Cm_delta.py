"""
Created on Mon Mar  23 10:53:20 2020

@author: timo
This script calculates the Cm_delta and the Cm_alpha from the flight data provided.
"""

import numpy as np
import pandas as pd
from Delta_e_curves_flightData import *
from acCGcalculator import *



cgData = "Flight_cg_data.xlsx"
seatData = "Flight_Seat_data.xlsx"

def Cm_delta(crewSeat1, crewSeat2, fuelmass, cgData, seatData):
    # constance of the aircraft
    S = 30.00 #m^2
    chord = 2.0569 #m

    # import of the data obtained by changing the cg whitin the aircraft(cgData) with the corresponding position data of the partisipants (seatData)
    cg_data = pd.read_excel(cgData)
    cg_data.columns = ['Altitude_ft', 'IAS', 'a', 'de', 'detr', 'Fe', 'FFL_lbs/hr', 'FFR_lbs/hr', 'F_used', 'TAT_C']

    seat_data = pd.read_excel(seatData)         #Import Excel File with all the data
    seat_data.columns = ['Seat_1', 'Seat_2', 'W_pass']

    # obtaining the right numbers from the cgData, the airspeed as asumed constant and the change in elevator deflection
    V = np.asarray(trim_curve_data['IAS']*0.514444444)[0]              #Speed in [m/s]
    delta_e = np.asarray(trim_curve_data['de'])                     #change of Elevator deflection in [deg]

    # Obtaining the right data from the seat_data to put in the calculateCG



    # converting the final values
    d_delta = delta_e[-1] - delta_e[0]

    cg1, weight = calculateCG(crewSeat1, fuelMass)
    cg2, weight = calculateCG(crewSeat2, fuelMass)
    dcg = cg2 - cg1
    # using the given formula to calculate Cm_delta
    Cn = weight / (0.5 * 1.225 * V^2 * S)
    Cm_delta = - (1/d_delta)*Cn*(dcg/chord)


return Cm_delta

# slope = - 0.4797 # reference data
# slope = - 0.4500 # flight data 20200312_F2

def Cm_alpha(slope, Cm_delta):

    Cm_alpha = - slope * Cm_delta
    return Cm_alpha
