import numpy as np
import random
import sys

def data_aug(skeleton, compoent_num, noise_val, shift_val, scale_val):
    joints = 3
    do_noise = False
    do_scale = False
    do_shift = False


    if noise_val !=0:
        do_noise = True
    if scale_val !=0:
        do_scale = True
    if shift_val !=0:
        do_shift = True

    def scale(skeleton):
        ratio = scale_val
        low = 1 - ratio
        high = 1 + ratio
        factor = np.random.uniform(low, high)
        video_len = skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(compoent_num):
                skeleton[t][j_id] *= factor
        skeleton = np.array(skeleton)
        return skeleton

    def shift(skeleton):
        high = shift_val
        low = -high
        offset = np.random.uniform(low, high, joints)
        video_len = skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(compoent_num):
                skeleton[t][j_id] += offset
        skeleton = np.array(skeleton)
        return skeleton

    def noise(skeleton):
        high = noise_val
        low = -high
        #select 4 joints
        all_joint = list(range(compoent_num))
        random.shuffle(all_joint)
        selected_joint = all_joint[0:4]

        for j_id in selected_joint:
            noise_offset = np.random.uniform(low, high, joints)
            for t in range(len(skeleton)):
                skeleton[t][j_id] += noise_offset
        skeleton = np.array(skeleton)
        return skeleton


    skeleton = np.array(skeleton).reshape(len(skeleton),compoent_num,joints)
    if do_noise:
        skeleton = noise(skeleton)
    if do_scale:
        skeleton = scale(skeleton)
    if do_shift:
        skeleton = shift(skeleton)
    # skeleton -= skeleton[0][1]
    skeleton = np.array(skeleton).reshape(len(skeleton),compoent_num*joints)

    return skeleton

joints = 22
scale = .2
shift = .1
noise = .1

if(len(sys.argv) < 3):
    print("Usage: python augment.py <filename> <number-of-augmented-files-to-produce>")

for i in range(int(sys.argv[2])):
    filename = sys.argv[1]
    skeleton = np.genfromtxt(filename, delimiter=',')
    new_skeleton = data_aug(skeleton, joints, scale, shift, noise)
    new_filename = filename.removesuffix(".json") \
                    + ("_scale{}".format(scale) if scale else "") \
                    + ("_shift{}".format(shift) if shift else "") \
                    + ("_noise{}".format(noise) if noise else "") \
                    + "_" + str(i) + ".json"
    np.savetxt(new_filename, new_skeleton, delimiter=" ") # match SHREC dataset format
