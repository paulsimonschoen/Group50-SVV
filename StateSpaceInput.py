import numpy as np

fnames = ["reading flight data/input_data/input_short_period1.txt","reading flight data/input_data/input_phugoid.txt","reading flight data/input_data/input_spiral.txt","reading flight data/input_data/input_dutch_roll.txt","reading flight data/input_data/input_aperiodic_roll.txt"]

sp1_ = np.genfromtxt(fnames[0],skip_header=0, skip_footer=285)
sp2_ = np.genfromtxt(fnames[0],skip_header=0, skip_footer=285)
phug_ = np.genfromtxt(fnames[1],skip_header=1775, skip_footer=0)
#phug_t = np.genfromtxt(fname[1],skip_header=219‬, skip_footer=0)
spiral_ = np.genfromtxt(fnames[2])
droll_ = np.genfromtxt(fnames[3])
aroll_ =np.genfromtxt(fnames[4])



print(aroll_[:,1:])

