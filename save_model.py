import sys
import os
import tensorflow as tf

print("THIS CURRENTLY DOESN'T DO ANYTHING: USE testmodel.py TO SAVE A MODEL")

if len(sys.argv) < 2: # 
    print("Usage: python save_model.py <checkpoint-file-location>")
    exit()

# find checkpoint file (make sure it exists)
cp_file = sys.argv[1]
if os.path.isfile(cp_file):
    print("Loading checkpoint file at ", os.path.abspath(cp_file))
    dir_name, f_name = os.path.split(os.path.abspath(cp_file))
    print(dir_name)
    print(f_name)




