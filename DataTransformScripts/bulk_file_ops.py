import os
import subprocess
import sys
from tqdm import tqdm

# subprocess.run(["python", "augment.py", "UnityTestGestures_2/01_Grab/gesture1_01.gold", "3", "UnityTestGestures_2_Output/"], shell=True)
# alias_command = "python"
# alias_expansion = os.popen("alias " + alias_command).read().split("=")[-1].strip('"\'')
# subprocess.call(alias_expansion.split(), shell=False)

# subprocess.call(["../Intention-Scripts/intention-venv/bin/python"], shell=True)

for root, dirs, files in tqdm(os.walk("UnityTestGestures_2")):
    for file in files:
        # Code to convert all fold files into json files 
        # subprocess.call([sys.executable, "DataTransformScripts/Unity2JSON.py", os.path.join(root, file)])
        # Code to augment all jsons into a separate dataset folder
        if(file.endswith(".json")):
            subprocess.call([sys.executable, "DataTransformScripts/augment.py", os.path.join(root, file), "3", "UnityTestGestures_2_Output/"])
