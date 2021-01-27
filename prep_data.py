import numpy as np
import cv2
import useful_funcs
import os
input_parent_folder="../../filer"
input_folders =["pasta", "burgers", "pizza"]

input_folders_refined = {i:[] for i in input_folders}
output_parent_folder="data"
image_size_x = 128
image_size_y = 128

print(input_folders_refined)


paths = []

for input_folder in list(input_folders_refined.keys()):
    current_folder = list(filter(lambda x: useful_funcs.find_score(x,"","") >= 0,
                                 useful_funcs.list_dir_recursive(os.path.join(input_parent_folder,
                                                                              "reddit_sub_" + input_folder))))
    if len(current_folder) >= 1000:
        input_folders_refined[input_folder] = current_folder
    else:
        del input_folders_refined[input_folder]

for output_folder in input_folders_refined.keys():
    os.makedirs(os.path.join(output_parent_folder,output_folder),exist_ok=True)
print(paths)

images = []
index = 0

folder_index = 0
for folder, path_group in input_folders_refined.items():
    print(path_group)
    for path in path_group:
        if(not useful_funcs.is_jpg(path)):
            continue
        try:
            image = cv2.imread(path)
            output_name = os.path.join(output_parent_folder,folder, str(useful_funcs.find_score(path)) + "_" + str(index)+".jpg")
            image = useful_funcs.pad_to_size(image,image_size_x,image_size_y)
            cv2.imwrite(output_name,image)
            index += 1
            if index%1000 == 0:
                print("index:", index)
        except Exception as e:
            print(e)

    folder_index += 1
