# This code is based on https://github.com/ultralytics/JSON2YOLO
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

def make_dirs(dir='../../yolox-pytorch/dataset/ours/annotations_txt/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
        
    dir.mkdir(parents=True, exist_ok=True)  # make dir
    
    return dir


def convert_coco_json(json_dir='../../yolox-pytorch/dataset/ours/annotations', use_segments=False, cls91to80=False):
    save_dir = make_dirs()  # output directory

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob('*.json')):
        fn = Path(save_dir) / json_file.stem.replace('instances_', '') # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {x['id']: x for x in data['images']}

        # Write labels file
        for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
            if x['iscrowd']:
                continue

            img = images[x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']
            
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            # Segments
            if use_segments:
                segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
                s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                cls = x['category_id'] - 1
                
                line = cls, *(s if use_segments else box)  # cls, box or segments
                with open((fn / f).with_suffix('.txt'), 'a') as file:
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':
    source = 'COCO'

    if source == 'COCO':
        convert_coco_json('../../yolox-pytorch/dataset/ours/annotations')  # directory with *.json
