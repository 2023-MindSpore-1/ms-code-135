import os
import shutil
import pandas as pd

train = False
root = '/opt/data/private/wenyu/dataset/CUB_200_2011/'

img_folder = os.path.join(root, "images")
img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,
                         names=['idx', 'label'])
train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,
                               names=['idx', 'train_flag'])
data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
data = data[data['train_flag'] == train]
file_path = data['path'].tolist()


old_file_root = os.path.join(root, "images")
if train:
    new_file_root = os.path.join(root, "train")
else:
    new_file_root = os.path.join(root, "test")

for i in range(len(file_path)):
    print(file_path[i])
    print(os.path.join(old_file_root,file_path[i]))
    print(os.path.join(new_file_root,file_path[i].split("/")[1]))
    shutil.copy(os.path.join(old_file_root,file_path[i]), os.path.join(new_file_root,file_path[i].split("/")[1]))
