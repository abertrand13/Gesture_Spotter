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
	# print(predictions)
	predicted_label = np.argmax(predictions)

	# print(predicted_label)	
	return (predicted_label, predictions[0][predicted_label]) # [0]??

# def move_through_gesture(gesture_data, true_label):
# 	send_http_request(gesture_data, true_label)


def feed_live_gesture_stream(subject, sequence, window_size):
	rootdir = "OnlineDHG/ODHG2016/"
	skeleton_path = "subject_{}/sequence_{}/skeletons_world_enhanced.txt"

	stream_file_data = np.genfromtxt(os.path.join(rootdir, skeleton_path.format(subject, sequence)))
	num_frames = len(stream_file_data)

	current_predicted_gesture = -1
	prediction_threshold = .8
	current_recurrence = 0
	recurrence_threshold = 15


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
			print("Gesture {} detected at frame {}".format(current_predicted_gesture+1, i))


num_subjects = 5
num_sequences = 5
for i in range(1,num_subjects):
	for j in range(1, num_sequences):
		print("Feeding through subject {}, sequence {}".format(i, j))
		feed_live_gesture_stream(i, j, 30)



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
