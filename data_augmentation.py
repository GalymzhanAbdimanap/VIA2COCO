import os
import cv2
import json


data_dir = 'dataset_3_helmets/all'
out_dir = 'dataset_3_helmets/all_augmented'
ann_file = 'dataset_3_helmets/all/all_annotations.json'

with open(ann_file, 'r') as f:
    data = json.load(f)

result_json = dict()



for filename in os.listdir(data_dir):
    if os.path.splitext(filename.lower())[1] != '.jpg':
        continue

    image = cv2.imread(os.path.join(data_dir, filename))
    size = os.stat(os.path.join(data_dir, filename)).st_size.__str__()
    ann = data.get(filename + size)
    
    # ------------------------------
    # 0 original image
    cv2.imwrite(os.path.join(out_dir, '0__' + filename), image)
    size = os.stat(os.path.join(out_dir, '0__' + filename)).st_size.__str__()
    if ann:
        result_json['0__' + filename + str(size)] = {
            "filename":'0__' + filename,
            "size":size,
            "regions":ann['regions'],
            "file_attributes":{}
        }

    # -------------------------------
    # 1 horizontal flip original
    flipped = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(out_dir, '1__' + filename), flipped)
    size = os.stat(os.path.join(out_dir, '1__' + filename)).st_size.__str__()
    height, width = flipped.shape[:2]
    if ann:
        regions = []
        for attr in ann['regions']:
            bbox = attr['shape_attributes']
            x = width - (bbox['x'] + bbox['width'])
            y = bbox['y']
            w = bbox['width']
            h = bbox['height'] 
            regions.append({"shape_attributes":{"name":"rect","x":x,"y":y,"width":w,"height":h},
                            "region_attributes":attr["region_attributes"]})
        result_json['1__' + filename + str(size)] = {
            "filename":'1__' + filename,
            "size":size,
            "regions":regions,
            "file_attributes":{}
        }

  
with open(os.path.join(out_dir, 'aug_all_annotations.json'), 'w') as outfile:
    json.dump(result_json, outfile)
