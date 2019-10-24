#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import sys

loamTrajectory = np.loadtxt(sys.argv[1])
plt.plot(loamTrajectory[:,3],loamTrajectory[:,11]) #x,z
plt.show()
