import os
import numpy as np
import shutil
import datetime
from tqdm import tqdm
import re

# for path, dirs, files in os.walk(rootdir):
#   for filename in files:
#       print(os.path.join(path, filename))


def setSequenceLength(seq, desiredSeqLength):
    if len(seq) < desiredSeqLength:
        padded = np.pad(seq, ((0, desiredSeqLength-len(seq)), (0,0)))
        return padded
    else:
        return seq[:desiredSeqLength]

def parseSingleGestureData(path, delimiter, data, labels, label, windowLength):
    gestureData = np.genfromtxt(path, delimiter=delimiter)

    # short circuit, just for now
    # gestureDataWindowed = setSequenceLength(gestureData, windowLength)
    # dataShape = np.shape(gestureDataWindowed)
    # gestureDataWindowedReshaped = np.reshape(gestureDataWindowed, (dataShape[1], dataShape[0]))
    # data.append(gestureDataWindowedReshaped)
    # labels.append(row[0])
    # continue
    
    # parse file
    numRows = len(gestureData)
    if numRows >= windowLength:
        # use only first `windowSize` samples
        # currentGestureData = gestureData[:windowLength]
        # dataShape = np.shape(currentGestureData)
        # shapedGestureData = np.reshape(currentGestureData, (dataShape[1], dataShape[0]))
        # data.append(shapedGestureData)
        # labels.append(row[0])

        # use as many full `windowSize` samples as you can get from full sample
        for i, dataRow in enumerate(gestureData):
            if i + windowLength < numRows:
                # Don't Reshape, or...
                # obj = gestureData[i:i+windowLength]
                # push data

                # Do Reshape
                currentGestureData = gestureData[i:i+windowLength]
                dataShape = np.shape(currentGestureData)
                shapedGestureData = np.reshape(currentGestureData, (dataShape[1], dataShape[0]))
                data.append(shapedGestureData)

                # push label    
                labels.append(label)

                    
def parseTestAugmentedGestures(windowLength=20):
    data = []
    labels = []

    rootdir = "UnityTestGestures"
    files = [f for f in os.listdir(rootdir) if os.path.isfile(os.path.join(rootdir,f))]
    for file in tqdm(files):
        potentialLabels = re.findall(r'test_gesture(\d+)', file)
        if len(potentialLabels) == 0:
            continue

        # I thought I was being clever by converting the delimiters but well, woops
        # Filter out 'gold' files that use commas
        if len(re.findall(r'scale', file)) == 0: # Fix to be...better. Not this line just like, the whole thing
            continue
        label = int(potentialLabels[0])
        parseSingleGestureData(os.path.join(rootdir, file), ' ', data, labels, label, windowLength)

    return data, labels
        

def parseSHRECGestures(filename,
                datasetType="train",
                windowLength=20):
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
        parseSingleGestureData(os.path.join(rootdir, path), ' ', data, labels, row[0]-1, windowLength) # dataset is 1-indexed for training labels
    
    return data, labels

# so the idea here is we give this a list of functions that will parse individual gesture sets
# each of those will return a list of data and a list of labels, and this will aggregate them all into the right place
# and hopefully that's a relatively scalable way to eventually compose multiple datasets or whatever we need
def parseGestures(windowLength,
                  outputFolder="DatasetParse/"):
    traindata = []
    trainlabels = []

    testdata = []
    testlabels = []
    
    # SHRECdata, SHREClabels = parseSHRECGestures("train_gestures.txt",
    #                                             datasetType="train",
    #                                             windowLength=windowLength)

    # for elem in SHRECdata:
    #     traindata.append(elem)
    # for elem in SHREClabels:
    #     trainlabels.append(elem)

    augdata, auglabels = parseTestAugmentedGestures(windowLength)

    for elem in augdata:
        traindata.append(elem)
    for elem in auglabels:
        # print("Label Element: ", elem) 
        # print(elem.dtype)
        trainlabels.append(elem)


    print("Resulting data shape: " + str(np.shape(traindata)))
    print("Saving to " + outputFolder)
    # np.save(outputFolder + datasetType + "_data", traindata)
    # np.save(outputFolder + datasetType + "_labels", trainlabels)
    np.save(outputFolder + "data", traindata)
    np.save(outputFolder + "labels", trainlabels)



# PARAMS
# ------
datasetOutFolder = "DatasetParse_v11/"
windowLength = 30

if not os.path.isdir(datasetOutFolder):
    os.mkdir(datasetOutFolder)

parseGestures(windowLength, datasetOutFolder)

# totalTrainSamples = parseGestures("train_gestures.txt",
#               datasetType="train",
#               windowLength=windowLength,
#               outputFolder=datasetOutFolder)
# totalTestSamples = parseGestures("test_gestures.txt",
#               datasetType="test",
#               windowLength=windowLength,
#               outputFolder=datasetOutFolder)

notes = "This is exclusively the augmented gestures, made by taking original gestures from Unity and then scaling/noising/shifting them"
f = open(datasetOutFolder + "notes.txt", 'w')
f.write("Date: " + str(datetime.datetime.now()) + "\n")
f.write("Window Size: " + str(windowLength) + "\n")
# f.write("Total train samples: " + str(totalTrainSamples) + "\n")
# f.write("Total test samples: " + str(totalTestSamples) + "\n")
f.write("Notes: " + notes + "\n")
f.close()
