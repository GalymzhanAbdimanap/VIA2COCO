import os
import json
import cv2



label_map = {'helmet': 1, 'head': 0}
image_dir = "../dataset_safety_clothes/kioge_new"
json_path = os.path.join(image_dir, "annotations.json")
yolo_ann_dir = "../dataset_safety_clothes/kioge_new"

with open(json_path, "r") as read_file:
    data_infos = json.load(read_file)


for v in data_infos.values():
    filename = v['filename']
    img_path = os.path.join(image_dir, filename)
    
    height, width = cv2.imread(img_path).shape[:2]

    yolo_ann_file = os.path.join(yolo_ann_dir, os.path.splitext(filename)[0] + '.txt')
    ann_file = open(yolo_ann_file, 'w')
    
    for obj in v['regions']:
        label = label_map[obj["region_attributes"]["class"]]
        obj = obj['shape_attributes']
        x = obj['x']
        y = obj['y']
        w = obj['width']
        h = obj['height']
        cx = (x + w/2)
        cy = (y + h/2)

        line = f'{label} {cx/width} {cy/height} {w/width} {h/height}\n'
        ann_file.write(line)

    ann_file.close()
    print(filename)
