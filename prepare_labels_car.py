import pandas as pd
import os
import glob
import cv2

from tqdm import tqdm

show_info = True
images_with_required_classes = 0
total_images = 0
labels = {
    'vehicle': 6
}



annotation_root = '../input/lisa_traffic_light_dataset/lisa-traffic-light-dataset/Annotations/Annotations'
image_root = '../input/lisa_traffic_light_dataset/lisa-traffic-light-dataset'


def get_coords(tag, x_min, y_min, x_max, y_max, images_with_required_classes):
        """
        We will return a single digit for each label.
        Also we will return normalized x_center, y_center, 
        width, and height. We will divice the x_center and width by 
        image width and y_center and height by image height to
        normalize. Each image is 1280 in width and 960 in height. 
        """
        if tag in labels:
            if tag == 'vehicle':
                label = labels['vehicle']
                color = (0, 255, 0)

            x_center = ((x_max + x_min) / 2) / 1280 
            y_center = ((y_max + y_min) / 2) / 720
            w = (x_max - x_min) / 1280
            h = (y_max - y_min) / 720
            return label, x_center, y_center, w, h
        else:
            label = ''
            x_center = ''
            y_center = ''
            w = ''
            h = ''
            return label, x_center, y_center, w, h

# for root_folder_name in root_folder_names:
#     folder_names = os.listdir(f"{annotation_root}/{root_folder_name}")
#     num_folders = len(folder_names)
#     mapped_clip = root_folder_name_mapper[root_folder_name]

    ##for i in range(1,  num_folders+1):
print('##### NEW CSV AND IMAGES ####')
# read the annotation CSV file
df = pd.read_csv(f"./input/vehicle/archive/nexet/nexet/train_boxes1.csv",
                 delimiter=',')
# get all image paths
image_paths = glob.glob(f"./input/vehicle/archive/nexet/nexet/nexet_2017_1/*jpg")
image_paths.sort()

total_images += len(image_paths)

if show_info:
    print('NUMBER OF IMAGE AND UNIQUE CSV FILE NAMES MAY NOT MATCH')
    print('NOT A PROBLEM')
    print(f"Total objects in current CSV file: {len(df)}")
    print(f"Unique Filenames: {len(df['image_filename'].unique())}")
    print(df.head())
    print(f"Total images in current folder: {len(image_paths)}")

tags = df['label'].values
x_min = df['x0'].values
y_min = df['y0'].values
x_max = df['x1'].values
y_max = df['y1'].values

#file_counter = 0  # to counter through CSV file
# iterate through all image paths
for file_counter, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    image_name = image_path.split(os.path.sep)[-1]
    #print('img',image_name)
    # iterate through all CSV rows
    for j in range(len(df)):
        if file_counter < len(df):
            print(df.loc[file_counter]['image_filename'])
            file_name = df.loc[5]['image_filename']
            #print('fl', file_name)
            if file_name == image_name:

                label, x, y, w, h = get_coords(tags[file_counter],
                                               x_min[file_counter],
                                               y_min[file_counter],
                                               x_max[file_counter],
                                               y_max[file_counter],
                                               images_with_required_classes)
                with open(f"./input/vehicle/input/labels/{image_name.split('.')[0]}.txt",
                          'a+') as f:
                    if type(label) == int:
                        f.writelines(f"{label} {x} {y} {w} {h}\n")
                        f.close()
                    else:
                        f.writelines(f"")
                        f.close()
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    cv2.imwrite(f"./input/vehicle/input/images/{image_name}", image)
                    #file_counter += 1
                # continue
            if file_name != image_name:
                break

print(f"Total images parsed through: {total_images}")
# print(f"Total images with desired classes: {images_with_required_classes}")