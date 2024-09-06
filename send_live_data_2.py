from threading import Thread
from time import sleep
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# SERVER LOCATION DEFINITION
port = 9001
model_name = "my-test-model"
headers = {"content-type": "application/json"}

# GLOBAL ACCURACY TRACKING VARS
data_tracked = []
gestures_correct = 0
gestures_incorrect = 0

# Send one block (eg 30 frames) of gesture data to server for classification
def send_http_request(gesture_data):
	data_shape = np.shape(gesture_data)
	data_reshaped = np.reshape(gesture_data, (1, data_shape[0], data_shape[1]))
	data = json.dumps({"signature_name":"serving_default", "instances":data_reshaped.tolist()})
	# print(data)
	
	endpoint = "http://localhost:" + str(port) + "/v1/models/" + model_name + ":predict"
	json_response = requests.post(endpoint, data=data, headers=headers)
	predictions = json.loads(json_response.text)["predictions"]
	predicted_label = np.argmax(predictions)

	return (predicted_label, predictions[0][predicted_label]) # [0]??


# Feed an entire gesture (n frames) through to server
def feed_live_gesture_stream(filepath, window_size):
	# skeleton_path = "subject_{}/sequence_{}/skeletons_world_enhanced.txt"

	# stream_file_data = np.genfromtxt(filepath)
	stream_file_data = np.genfromtxt(filepath, delimiter=',')
	num_frames = len(stream_file_data)

	current_predicted_gesture = -1
	prediction_threshold = .8
	current_recurrence = 0
	recurrence_threshold = 15

	inference_data = []

	for i in range(window_size, num_frames):
		current_gesture_data = stream_file_data[i-window_size:i]
		# print(current_gesture_data[0])
		# print(current_gesture_data[0][0] + current_gesture_data[0][1])
		data_shape = np.shape(current_gesture_data)
		shaped_gesture_data = np.reshape(current_gesture_data, (data_shape[1], data_shape[0]))
		predicted_label, predicted_label_probability = send_http_request(shaped_gesture_data)
		
		have_predicted_gesture = False
		
		if predicted_label_probability > prediction_threshold:
			have_predicted_gesture = True
		else:
			current_predicted_gesture = -1
			current_recurrence = 0
	
		if have_predicted_gesture:
			if predicted_label == current_predicted_gesture:
				current_recurrence += 1
			else:
				current_recurrence = 1

			current_predicted_gesture = predicted_label

		if current_recurrence == recurrence_threshold:
			# we got a gesture!
			# print("Gesture {} detected at frame {}".format(current_predicted_gesture+1, i))
			inference_data.append({"gesture_id":current_predicted_gesture+1, "timestamp": i})
	
	# print(inference_data)
	return inference_data

# things we care about:
# how many of the gestures that did happen, were correctly identified?
# for those correctly identified gestures, what was the average time to detection?
# how many times were gestures incorrectly identified repeatedly?
# how many times was null movement incorrectly identified as a gesture?
# how many gestures were correctly identified as a gesture, but miscategorized?
# how many gestures were missed (and never identified)?
def evaluate_identification_metrics(inference, ground_truth):
	# for every gesture that occurs, check timestamps
	# if the timestamp falls in within one of the correct gestures ranges:
	# 	if the gesture is correctly identified
	# 		if there has not already been a correct gesture identified
	# 			success - calculate time to detection
	# 		else if there has already been a correct gesture identified
	# 			error - repeat
	# 	else if the gesture was incorrectly identified
	# 		error - misidentification
	# if the timestamp does not fall in any correct range
	# 	error - null movement identification
	# 
	# for every timestamp range that did not have a gesture identified during it
	# 	error - missed gesture
	#
	# Data format (ground truth):
	# 
	# [
	# 	{
	# 		timestamp_range: ((start_time_0, end_time_0),
	# 		gesture_id: gesture_id_0,
	# 		gesture_already_identified: gesture_identified_0,
	#		gesture_identified_correctly: gesture_identified_correctly_0
	# 	},
	# 	{
	# 		timestamp_range: ((start_time_1, end_time_1),
	# 		gesture_id: gesture_id_1,
	# 		gesture_already_identified: gesture_identified_1,
	#		gesture_identified_correctly: gesture_identified_correctly_1
	# 	},
	# 	...
	# 	{
	# 		timestamp_range: ((start_time_n, end_time_n),
	# 		gesture_id: gesture_id_n,
	# 		gesture_already_identified: gesture_identified_n,
	#		gesture_identified_correctly: gesture_identified_correctly_n
	# 	}
	# ]
	
	# Data format (inference):	
	# [
	# 	{
	# 		gesture_id: gesture_id_0,
	# 		timestamp: timestamp_0
	# 	},
	# 	{
	# 		gesture_id: gesture_id_1,
	# 		timestamp: timestamp_1
	# 	},
	# 	...
	# 	{
	# 		gesture_id: gesture_id_n
	# 		timestamp: timestamp_n
	# 	}
	# ]

	num_correct_gestures = 0
	total_inference_time = 0
	num_repeat_errors = 0
	num_null_errors = 0
	num_misidentified_errors = 0
	num_missed_errors = 0

	for inf_gesture in inference:
		print(inf_gesture)
		inf_timestamp = inf_gesture["timestamp"]
		for truth_gesture in ground_truth:
			truth_timestamp = truth_gesture["timestamp_range"]
			if inf_timestamp >= truth_timestamp[0] and inf_timestamp <= truth_timestamp[1]:
				if inf_gesture["gesture_id"] == truth_gesture["gesture_id"]:
					if not truth_gesture["gesture_identified_correctly"]:
						num_correct_gestures += 1
						total_inference_time += inf_timestamp - truth_timestamp[0]
						truth_gesture["gesture_identified_correctly"] = True

					else:
						num_repeat_errors += 1
				else:
					num_misidentified_errors += 1
				truth_gesture["gesture_already_identified"] = True
				
				break
			
			elif inf_timestamp < truth_timestamp[0]:
				num_null_errors += 1
				break
	
	for truth_gesture in ground_truth:
		if not truth_gesture["gesture_already_identified"]:
			num_missed_errors += 1

	print("Correctly identified gestures: {}/{}".format(num_correct_gestures, len(ground_truth)))
	print("Average inference time: {}".format(str(total_inference_time / num_correct_gestures)))
	print("Repeat gesture errors: {}".format(num_repeat_errors))
	print("Null id errors: {}".format(num_null_errors))
	print("Misidentification errors: {}".format(num_misidentified_errors))
	print("Missed gesture errors: {}".format(num_missed_errors))

	return (num_correct_gestures, len(ground_truth),
			total_inference_time / num_correct_gestures,
			num_repeat_errors,
			num_null_errors,
			num_misidentified_errors,
			num_missed_errors)


def parse_subject_sequence_info(rootdir, subject, sequence):
	info_path = "subject_{}_infos_sequences.txt"
	
	# info_file_data = np.genfromtxt(os.path.join(rootdir, info_path.format(subject)))
	ground_truth = []
		
	with open(os.path.join(rootdir, info_path.format(subject))) as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			data = [int(x) for x in line.split()]
			# these loop in groups of 3:
			# gesture numbers, eg:					[1 14 8 7 ...]
			# number of fingers, eg:				[1 2  1 1 ...]
			# start and end frame of gesture, eg:	[40 120 135 176 201 265 294 352 ...]
			if i % 3 == 0 and i == sequence*3:
				# parse gesture numbers
				ground_truth = [{"gesture_id": x,
								"gesture_already_identified": False,
								"gesture_identified_correctly": False} for x in data]
			elif i % 3 == 2 and i == sequence*3 + 2:
				# parse frame times
				for j in range(0, len(data), 2):
					ground_truth[int(j/2)]["timestamp_range"] = (data[j], data[j+1])
					
	# print(ground_truth)	
	return ground_truth


num_subjects = 29
num_sequences = 16
# rootdir = "OnlineDHG/ODHG2016"
rootdir = "HandGestureDataset_SHREC2017/UnityTestGestures/"

total_sequences = 0
total_correct_gestures = 0
total_gestures = 0
total_average_inference_time = 0
total_repeat_errors = 0
total_null_errors = 0
total_misidentified_errors = 0
total_missed_errors = 0

	# return (num_correct_gestures, len(ground_truth),
	# 		total_inference_time / num_correct_gestures,
	# 		num_repeat_errors,
	# 		num_null_errors,
	# 		num_misidentified_errors,
	# 		num_missed_errors)

# TODO: Redo this to be more general purpose for file structure parsing. Or something.
# for i in range(1,num_subjects):
# 	for j in range(1, num_sequences):
# 		if not os.path.exists(os.path.join(rootdir, "subject_{}/sequence_{}".format(i,j))):
# 			continue
# 		skeleton_path = "subject_{}/sequence_{}/skeletons_world_enhanced.txt"
# 		ground_truth = parse_subject_sequence_info(rootdir, i, j-1)
# 		print("Feeding through subject {}, sequence {}".format(i, j))
# 		filepath = os.path.join(rootdir, skeleton_path.format(i, j))
# 		inference = feed_live_gesture_stream(filepath, 30)
# 		
# 		results = evaluate_identification_metrics(inference, ground_truth)
# 
# 		total_sequences += 1
# 		total_correct_gestures += results[0]
# 		total_gestures += results[1]
# 		total_average_inference_time = \
# 			(total_average_inference_time * ((total_gestures - results[1]) / total_gestures)) + \
# 			(results[2] * (results[1] / total_gestures))
# 		total_repeat_errors += results[3]
# 		total_null_errors += results[4]
# 		total_misidentified_errors += results[5]
# 		total_missed_errors += results[6]

for gesture_file in os.listdir(rootdir):
	print(gesture_file)
	filepath = os.path.join(rootdir, gesture_file)
	inference = feed_live_gesture_stream(filepath, 30)
	print(inference)


print("TOTALS")
print("Correctly identified gestures: {}/{}".format(total_correct_gestures, total_gestures))
print("Average inference time: {}".format(total_average_inference_time))
print("Repeat gesture errors: {}".format(total_repeat_errors))
print("Null id errors: {}".format(total_null_errors))
print("Misidentification errors: {}".format(total_misidentified_errors))
print("Missed gesture errors: {}".format(total_missed_errors))





# filename = "test_gestures.txt"
# windowLength = 30
# 
# totalTestSamples = identify_gestures(filename,
# 				windowFrameCount=windowLength)

# rootdir = "DatasetParse_v4"
# test_data = np.load(os.path.join(rootdir, "test_data.npy"))
# test_labels = np.load(os.path.join(rootdir, "test_labels.npy"))

# for i, data in enumerate(test_data):
# 	label = test_labels[i]
# 	move_through_gesture(data, label)

# print("Overall accuracy: {}/{} = {}%".format(gestures_correct, gestures_correct + gestures_incorrect, gestures_correct / (gestures_correct + gestures_incorrect) * 100.0))
# plt.plot(data_tracked)
# plt.show()
