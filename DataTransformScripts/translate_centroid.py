import os
import sys
import numpy as np
import json

if(len(sys.argv) < 3):
	print("Usage: python translate_centroid.py <source-data> <output-file>")
	quit()

source_data = np.genfromtxt(sys.argv[1], delimiter=',')

target_centroid = (.43, -.3, .5)
source_centroid = [0, 0, 0]
total_points = 0

# calculate centroid of source data
for row in source_data:
	x = row[::3]
	y = row[1::3]
	z = row[2::3]
	source_centroid[0] += sum(x)
	source_centroid[1] += sum(y)
	source_centroid[2] += sum(z)
	total_points += len(x)
	
source_centroid[0] /= total_points
source_centroid[1] /= total_points
source_centroid[2] /= total_points

centroid_diff = (target_centroid[0] - source_centroid[0], target_centroid[1] - source_centroid[1], target_centroid[2] - source_centroid[2])

for i in range(len(source_data)):
	source_data[i] = [x + centroid_diff[j%3] for j, x in enumerate(source_data[i])]

np.savetxt(sys.argv[2], source_data, delimiter=',')
