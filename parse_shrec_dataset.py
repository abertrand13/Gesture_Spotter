import os
import numpy as np
import shutil


# for path, dirs, files in os.walk(rootdir):
# 	for filename in files:
# 		print(os.path.join(path, filename))


def parseGestures(filename, datasetType="train"):
	data = []
	labels = []
	
	rootdir = "HandGestureDataset_SHREC2017"
	gestureFile = np.genfromtxt(os.path.join(rootdir, filename), dtype=np.intc, delimiter=' ')

	for i, row in enumerate(gestureFile):
		# construct file path from reference
		path = ("gesture_" + str(row[0]) + "/"
				"finger_" + str(row[1]) + "/"
				"subject_" + str(row[2]) + "/"
				"essai_" + str(row[3]) + "/"
				"skeletons_world.txt")
		print(path)

		# shutil.copy(os.path.join(rootdir, path), "./gestures/gesture_" + str(row[0]) + "_" + str(i) + ".txt")

		gestureData = np.genfromtxt(os.path.join(rootdir, path))
		windowFrameCount = 20 # sliding window size

		# parse file
		rows = len(gestureData)
		if rows >= windowFrameCount:
			obj = gestureData[:windowFrameCount]
			print(obj)
			print(obj.shape)
			# push data	
			data.append(gestureData[:windowFrameCount])

			# push label	
			labels.append(row[0])

		np.save(datasetType + "_data", data)
		np.save(datasetType + "_labels", labels)

parseGestures("test_gestures.txt", datasetType="test")
