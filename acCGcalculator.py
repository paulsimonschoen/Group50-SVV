# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:35:44 2020

@author: Vince van Deursen

This script calculates the center of gravity for the aircraft, given the fuel mass and crew position
"""

#----- common variables -----
crewMasses = {
        "TI": 80,
        "FR": 80,
        "PA": 80,
        "VT": 80,
        "VI": 80,
        "C1": 80,
        "C2": 80,
        }


seatCG = {
        "1": 131,
        "2": 131,
        "3": 214,
        "4": 214,
        "5": 251,
        "6": 251,
        "7": 288,
        "8": 288,
        "10": 170
        }

emptyMass = 9165.0
emptyCG   = 291.65

fuelMasses = []
fuelMoments = []

f = open("fuelmoments.txt")
lines = f.read().splitlines()
f.close()

for line in lines:
    line = line.replace("\n","").split(" ")
    fuelMasses.append(int(line[0]))
    fuelMoments.append(float(line[1])*100)


#----- FUNCTIONS -----
def calculateCG(crewSeat, fuelMass):    
    #crew contributions
    crewMass   = 0
    crewMoment = 0
    
    for i in crewSeat.items():
        crewMass = crewMass + crewMasses[i[0]] 
        
        crewMoment = crewMoment + crewMasses[i[0]]*seatCG[i[1]]
    
    #fuel contributions - use linear interpolation
    for i in range(len(fuelMasses)):
        if i > 0 and fuelMasses[i-1] <= fuelMass and fuelMasses[i] >= fuelMass:
            fuelMoment = (fuelMoments[i] - fuelMoments[i-1])/(fuelMasses[i] - fuelMasses[i-1])*(fuelMass - fuelMasses[i-1]) + fuelMoments[i-1]
        if fuelMass == 0:
            fuelMoment = 0
    #total cg calculation  
    acMass = fuelMass + emptyMass + crewMass
    acCG = (fuelMoment + emptyMass*emptyCG + crewMoment) / acMass
    

    print(fuelMoment, emptyMass, crewMass, emptyCG, crewMoment)
    return acCG, acMass

#----- SCRIPT -----

fuelMass = 750

crewSeat = {
        "TI": "1",  #timo
        "FR": "2",  #friso
        "PA": "3",  #paul
        "VT": "4",  #vincent
        "VI": "5",  #vince
        "C1": "6",  #crew 1
        "C2": "7"   #crew 2
        }
    

print(calculateCG(crewSeat, fuelMass))
