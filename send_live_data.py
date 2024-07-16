from threading import Thread
from time import sleep
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

run_loop = True

def call_at_interval(period, callback, args):
    while(run_loop):
        sleep(period)
        callback(*args)

def set_interval(period, callback, *args):
    Thread(target=call_at_interval, args=(period, callback, args)).start()

def print_a_thing(str_to_print):
    print(str_to_print)

# def do_the_thing():
# 	num_subjects = 
# 	rootdir = "OnlineDHG/ODHG2016/"
# 	for root, dirs, files in os.walk(rootdir):
# 		for filename in files:
# 			if filename == "skeletons_world.txt":
# 				print(str(os.path.join(root, filename)))
		

# set_interval(1, print_a_thing, "hey its a thread!")
# do_the_thing()


port = 9001
model_name = "my-test-model"
headers = {"content-type": "application/json"}

data_tracked = []
gestures_correct = 0
gestures_incorrect = 0

def send_http_request(gesture_data):
	data_shape = np.shape(gesture_data)
	# print(data_shape)
	data_reshaped = np.reshape(gesture_data, (1, data_shape[0], data_shape[1]))
	data = json.dumps({"signature_name":"serving_default", "instances":data_reshaped.tolist()})
	# print(data)
	endpoint = "http://localhost:" + str(port) + "/v1/models/" + model_name + ":predict"
	# print(endpoint)
	json_response = requests.post(endpoint, data=data, headers=headers)
	# print(json_response)
	# print(dir(json_response))
	# print(json_response.reason)
	# print(json_response.text)
	predictions = json.loads(json_response.text)["predictions"]
	# print("Request returned gesture {}; it was actually {}".format(np.argmax(predictions), true_label))
	# if np.argmax(predictions) == true_label:
	# 	data_tracked.append(0)
	# else:
	# 	data_tracked.append(1)
	return np.argmax(predictions)

# def move_through_gesture(gesture_data, true_label):
# 	send_http_request(gesture_data, true_label)


def identify_gestures(filename, windowFrameCount=20):
	data = []
	labels = []
	
	rootdir = "HandGestureDataset_SHREC2017"
	gestureFile = np.genfromtxt(os.path.join(rootdir, filename), dtype=np.intc, delimiter=' ')

	# print("Parsing " + datasetType + " data")

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

		# gestureDataWindowed = setSequenceLength(gestureData, windowFrameCount)
		# dataShape = np.shape(gestureDataWindowed)
		# gestureDataWindowedReshaped = np.reshape(gestureDataWindowed, (dataShape[1], dataShape[0]))
		# data.append(gestureDataWindowedReshaped)
		# labels.append(row[0])
		# continue
		

		gesture_identifications = [0]*15 # number of possible gestures (+1)
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
					# data.append(shapedGestureData)

					# push label	
					# labels.append(row[0])
					# label = row[0]
					data = shapedGestureData
					predicted_label = send_http_request(data)
					gesture_identifications[predicted_label] += 1

		true_label = row[0]	- 1
		fully_predicted_label = np.argmax(gesture_identifications)
		if(fully_predicted_label != true_label):
			# should probably get a confusion matrix going here, or something
			print("Gesture: predicted {}, actually {}".format(fully_predicted_label, true_label))

		# well this feels wrong:
		global gestures_correct
		global gestures_incorrect
		

		# anyway...	
		if fully_predicted_label == true_label:
			gestures_correct += 1
		else:
			gestures_incorrect += 1

	# print("Resulting data shape: " + str(np.shape(data)))
	# print("Saving to " + outputFolder)
	# np.save(outputFolder + datasetType + "_data", data)
	# np.save(outputFolder + datasetType + "_labels", labels)
	return len(labels)

filename = "test_gestures.txt"
windowLength = 30

totalTestSamples = identify_gestures(filename,
				windowFrameCount=windowLength)

# rootdir = "DatasetParse_v4"
# test_data = np.load(os.path.join(rootdir, "test_data.npy"))
# test_labels = np.load(os.path.join(rootdir, "test_labels.npy"))

# for i, data in enumerate(test_data):
# 	label = test_labels[i]
# 	move_through_gesture(data, label)

print("Overall accuracy: {}/{} = {}%".format(gestures_correct, gestures_correct + gestures_incorrect, gestures_correct / (gestures_correct + gestures_incorrect) * 100.0))
# plt.plot(data_tracked)
# plt.show()
