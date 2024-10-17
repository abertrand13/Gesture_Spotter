import os
import sys
import fileinput

if len(sys.argv) < 2:
    print("USAGE: python Unity2JSON.py <unity-recorded-file-to-convert>")
    print("script assumes that the unity file will have a '.gold' extension")

filename = sys.argv[1]

with open(filename, 'r') as file:
    filedata = file.read()

# for line in filedata:
#     line = line.replace('][', '\n') # most lines
#     line = line.replace('[', '') # first line
#     line = line.replace(']', '') # last line

filedata = filedata.replace('][', '\n') # most lines
filedata = filedata.replace('[', '') # first line
filedata = filedata.replace(']', '') # last line


filename = filename.strip(".gold") + ".json"

with open(filename, 'w') as file:
    file.write(filedata)
