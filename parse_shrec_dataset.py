import os
import numpy as np
import shutil

rootdir = "HandGestureDataset_SHREC2017"

# for path, dirs, files in os.walk(rootdir):
# 	for filename in files:
# 		print(os.path.join(path, filename))

gestureFile = np.genfromtxt(os.path.join(rootdir, "train_gestures.txt"), dtype=np.intc, delimiter=' ')

for i, row in enumerate(gestureFile):
	# construct file path from reference
	path = ("gesture_" + str(row[0]) + "/"
			"finger_" + str(row[1]) + "/"
			"subject_" + str(row[2]) + "/"
			"essai_" + str(row[3]) + "/"
			"skeletons_world.txt")
	print(path)

	shutil.copy(os.path.join(rootdir, path), "./gestures/gesture_" + str(row[0]) + "_" + str(i) + ".txt")
