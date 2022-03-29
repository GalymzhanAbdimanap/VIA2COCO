import os
import json
import cv2



label_map = {'hawser': 0}
image_dir = "./hawser_dataset"
json_path = os.path.join(image_dir, "hawser_all_ann.json")
yolo_ann_dir = "./hawser_dataset"

with open(json_path, "r") as read_file:
    data_infos = json.load(read_file)


for v in data_infos.values():
    filename = v['filename']
    img_path = os.path.join(image_dir, filename)
    
    height, width = cv2.imread(img_path).shape[:2]

    
    
    if len(v['regions']) == 0:
        continue


    yolo_ann_file = os.path.join(yolo_ann_dir, os.path.splitext(filename)[0] + '.txt')
    ann_file = open(yolo_ann_file, 'w')
    
    for obj in v['regions']:
        # label = label_map[obj["region_attributes"]["class"]]
        label = 0
        obj = obj['shape_attributes']
       
        x = min(obj['all_points_x'])  # MAX MIN
        y = min(obj['all_points_y'])
        w = max(obj['all_points_x']) - x
        h = max(obj['all_points_y']) - y
        cx = (x + w/2)
        cy = (y + h/2)

        line = f'{label} {cx/width} {cy/height} {w/width} {h/height}\n'
        ann_file.write(line)
    


    ann_file.close()
    print(filename)
