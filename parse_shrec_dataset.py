import os
import numpy as np
import shutil
import datetime
from tqdm import tqdm

# for path, dirs, files in os.walk(rootdir):
# 	for filename in files:
# 		print(os.path.join(path, filename))


def setSequenceLength(seq, desiredSeqLength):
	if len(seq) < desiredSeqLength:
		padded = np.pad(seq, ((0, desiredSeqLength-len(seq)), (0,0)))
		return padded
	else:
		return seq[:desiredSeqLength]

def parseGestures(filename,
				datasetType="train",
				windowFrameCount=20,
				outputFolder="DatasetParse/"):
	data = []
	labels = []
	
	rootdir = "HandGestureDataset_SHREC2017"
	gestureFile = np.genfromtxt(os.path.join(rootdir, filename), dtype=np.intc, delimiter=' ')

	print("Parsing " + datasetType + " data")

	for _, row in tqdm(enumerate(gestureFile), total=len(gestureFile)):
		# construct file path from reference
		path = ("gesture_" + str(row[0]) + "/"
				"finger_" + str(row[1]) + "/"
				"subject_" + str(row[2]) + "/"
				"essai_" + str(row[3]) + "/"
				"skeletons_world.txt")
		# print(path)

		# shutil.copy(os.path.join(rootdir, path), "./gestures/gesture_" + str(row[0]) + "_" + str(i) + ".txt")

		gestureData = np.genfromtxt(os.path.join(rootdir, path))

		# short circuit, just for now
		# gestureDataWindowed = setSequenceLength(gestureData, windowFrameCount)
		# dataShape = np.shape(gestureDataWindowed)
		# gestureDataWindowedReshaped = np.reshape(gestureDataWindowed, (dataShape[1], dataShape[0]))
		# data.append(gestureDataWindowedReshaped)
		# labels.append(row[0])
		# continue
		

		# parse file
		numRows = len(gestureData)
		if numRows >= windowFrameCount:
			# use only first `windowSize` samples
			# currentGestureData = gestureData[:windowFrameCount]
			# dataShape = np.shape(currentGestureData)
			# shapedGestureData = np.reshape(currentGestureData, (dataShape[1], dataShape[0]))
			# data.append(shapedGestureData)
			# labels.append(row[0])

			# use as many full `windowSize` samples as you can get from full sample
			for i, dataRow in enumerate(gestureData):
				if i + windowFrameCount < numRows:
					# Don't Reshape, or...
					# obj = gestureData[i:i+windowFrameCount]
					# push data

					# Do Reshape
					currentGestureData = gestureData[i:i+windowFrameCount]
					dataShape = np.shape(currentGestureData)
					shapedGestureData = np.reshape(currentGestureData, (dataShape[1], dataShape[0]))
					data.append(shapedGestureData)

					# push label	
					labels.append(row[0]-1) # dataset is 1-indexed for training labels

			

	print("Resulting data shape: " + str(np.shape(data)))
	print("Saving to " + outputFolder)
	np.save(outputFolder + datasetType + "_data", data)
	np.save(outputFolder + datasetType + "_labels", labels)
	return len(labels)

# PARAMS
# ------
datasetOutFolder = "DatasetParse_v9/"
windowLength = 30

if not os.path.isdir(datasetOutFolder):
	os.mkdir(datasetOutFolder)

totalTrainSamples = parseGestures("train_gestures.txt",
				datasetType="train",
				windowFrameCount=windowLength,
				outputFolder=datasetOutFolder)
totalTestSamples = parseGestures("test_gestures.txt",
				datasetType="test",
				windowFrameCount=windowLength,
				outputFolder=datasetOutFolder)

notes = "This should be the same as dataset 4 except without the extra label bucket. So there are 14 gestures and 14 possible labels, as opposed to 15 (which is what there was before)"
f = open(datasetOutFolder + "notes.txt", 'w')
f.write("Date: " + str(datetime.datetime.now()) + "\n")
f.write("Window Size: " + str(windowLength) + "\n")
f.write("Total train samples: " + str(totalTrainSamples) + "\n")
f.write("Total test samples: " + str(totalTestSamples) + "\n")
f.write("Notes: " + notes + "\n")
f.close()
