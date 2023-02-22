#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

if (len(sys.argv) < 3):
    print ("%s [Number of Time Steps] [Number of Processes]\n" % (sys.argv[0]))
    sys.exit()
    
nSteps = int(sys.argv[1])
numproc = int(sys.argv[2])

print(nSteps,numproc)

fig = plt.figure()
ax = plt.axes(projection='3d',xlim =(0, 1),ylim =(0, 1),zlim=(0.1))

def update(timeStep):
    ax.cla()
    for proc in range(numproc):
        fileName = ("Seq_%03d_%02d.txt" % (timeStep,proc))
        if (False == os.path.isfile(fileName)):
            sys.exit("File %s not found\n" % (fileName))

        print("Import %s ..." % (fileName)) 
        pData = np.genfromtxt(fileName,delimiter=' ')

        ax.scatter3D(pData[:,0], pData[:,1], pData[:,2],marker='.')
#
#

anim = animation.FuncAnimation(fig, update, frames=range(nSteps))

mywriter = animation.FFMpegWriter()
anim.save('Seq.mp4', writer=mywriter);
