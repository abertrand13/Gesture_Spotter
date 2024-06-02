import os
import shutil

rootdir = "HandGestureDataset_SHREC2017"
newrootdir = "HandGestureDataset_SHREC2017_Stripped"

# if not os.path.isdir(newrootdir):
# 	os.mkdir(newrootdir)

for path, dirs, files in os.walk(rootdir):
	for filename in files:
		if filename == "skeletons_world.txt":
			filepath = os.path.join(path, filename)
			relpath = os.path.relpath(filepath, rootdir)
			print(relpath)
			newrelpath = os.path.join(newrootdir, relpath)
			print(newrelpath)
			os.makedirs(newrelpath, exist_ok=True)
			shutil.copy(filepath, newrelpath)
