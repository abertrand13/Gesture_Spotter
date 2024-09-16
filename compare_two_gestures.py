import numpy as np
import os
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import json

# Mac
# rootdir = "/Users/alexbertrand/Library/CloudStorage/GoogleDrive-alex.bertrand13@gmail.com/My Drive/Skywhale Labs/datasets/HandGestureDataset_SHREC2017_Stripped"
# path = "gesture_2/finger_1/subject_2/essai_1/skeletons_world.txt/skeletons_world.txt"
# path2 = "gesture_2/finger_1/subject_1/essai_1/skeletons_world.txt/skeletons_world.txt"

# Windows
rootdir = "HandGestureDataset_SHREC2017"
path = "gesture_3/finger_1/subject_1/essai_1/skeletons_world.txt"
# path2 = "gesture_2/finger_1/subject_2/essai_1/skeletons_world.txt"
path2 = "UnityTestGestures/test_gesture03.json"

skeleton1_positions = np.genfromtxt(os.path.join(rootdir, path))
skeleton2_positions = np.genfromtxt(os.path.join(rootdir, path2), delimiter=',')
# f = open(os.path.join(rootdir, path2))
# skeleton2_positions = json.load(f)
print(skeleton2_positions)

idx = 0
hand1_points = None
hand2_points = None
lines = []

# with open(os.path.join(rootdir, path) as f:

def animate(i, x=[], y=[]):
	global idx, hand1_points, hand2_points, lines

	# Erase previous frame
	if hand1_points != None:
		hand1_points.remove()
	if hand2_points != None:
		hand2_points.remove()
	for line in lines:
		for elem in line:
			elem.remove()
	lines = []

	ax = fig.get_axes()[0]

	hand1_points = draw_hand(skeleton1_positions, idx, ax)
	hand2_points = draw_hand(skeleton2_positions, idx, ax, 'r')
	
	idx = (idx + 1) % min(len(skeleton1_positions), len(skeleton2_positions))
	
def draw_hand(skeleton_positions, idx, ax, color='b'):
	data = skeleton_positions[idx]
	x = data[::3]
	y = data[1::3]
	z = data[2::3]
	
	# ax = fig.get_axes()[0]
	points = ax.scatter(x,y,z,c=color)
	lines.append(ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], c=color))
	lines.append(ax.plot([x[0], x[2]], [y[0], y[2]], [z[0], z[2]], c=color))
	lines.append(ax.plot([x[2], x[3]], [y[2], y[3]], [z[2], z[3]], c=color))
	lines.append(ax.plot([x[3], x[4]], [y[3], y[4]], [z[3], z[4]], c=color))
	lines.append(ax.plot([x[4], x[5]], [y[4], y[5]], [z[4], z[5]], c=color))
	lines.append(ax.plot([x[1], x[6]], [y[1], y[6]], [z[1], z[6]], c=color))
	lines.append(ax.plot([x[6], x[7]], [y[6], y[7]], [z[6], z[7]], c=color))
	lines.append(ax.plot([x[7], x[8]], [y[7], y[8]], [z[7], z[8]], c=color))
	lines.append(ax.plot([x[8], x[9]], [y[8], y[9]], [z[8], z[9]], c=color))
	lines.append(ax.plot([x[1], x[10]], [y[1], y[10]], [z[1], z[10]], c=color))
	lines.append(ax.plot([x[10], x[11]], [y[10], y[11]], [z[10], z[11]], c=color))
	lines.append(ax.plot([x[11], x[12]], [y[11], y[12]], [z[11], z[12]], c=color))
	lines.append(ax.plot([x[12], x[13]], [y[12], y[13]], [z[12], z[13]], c=color))
	lines.append(ax.plot([x[1], x[14]], [y[1], y[14]], [z[1], z[14]], c=color))
	lines.append(ax.plot([x[14], x[15]], [y[14], y[15]], [z[14], z[15]], c=color))
	lines.append(ax.plot([x[15], x[16]], [y[15], y[16]], [z[15], z[16]], c=color))
	lines.append(ax.plot([x[16], x[17]], [y[16], y[17]], [z[16], z[17]], c=color))
	lines.append(ax.plot([x[1], x[18]], [y[1], y[18]], [z[1], z[18]], c=color))
	lines.append(ax.plot([x[18], x[19]], [y[18], y[19]], [z[18], z[19]], c=color))
	lines.append(ax.plot([x[19], x[20]], [y[19], y[20]], [z[19], z[20]], c=color))
	lines.append(ax.plot([x[20], x[21]], [y[20], y[21]], [z[20], z[21]], c=color))
	# points = plt.scatter(x,y,z, c=color)

	return points

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim([.30, .55])
ax.set_ylim([-.42, -.20])
ax.set_zlim([.25, .7])
ani = FuncAnimation(fig, animate, interval = 50)
plt.show()
