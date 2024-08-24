import numpy as np
import os
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

rootdir = "HandGestureDataset_SHREC2017"
path = "gesture_1/finger_1/subject_1/essai_1/skeletons_world.txt"

skeleton_positions = np.genfromtxt(os.path.join(rootdir, path))
print(skeleton_positions)
print(skeleton_positions.shape)

idx = 0

# with open(os.path.join(rootdir, path) as f:

def animate(i, x=[], y=[]):
	global idx	

	plt.cla()
	data = skeleton_positions[idx]
	x = data[::3]
	y = data[1::3]
	z = data[2::3]
	
	plt.scatter(x,y,z)
	idx += 1

fig = plt.figure()
ani = FuncAnimation(fig, animate, interval = 100)
plt.show()
