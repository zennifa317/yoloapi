import argparse
import glob

from yolo_api import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_file', type=str, default=None)
    parser.add_argument('--txt_file', type=str, default=None)
    parser.add_argument('--json_path', type=str, default=None)

    arg = parser.parse_args()

    yolo = Yolo()

    img_file = arg.img_file
    txt_file = arg.txt_file
    output = arg.json_path

    img_path = []
    txt_path = []
    img_path = glob.glob(os.path.join(img_file, '*.jpg'))
    txt_path = glob.glob(os.path.join(txt_file, '*.txt'))

    yolo.load_imgs_annos(images_file=img_path, annotations_file=txt_path)
    yolo.create_json(json_path=output)