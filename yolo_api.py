from collections import defaultdict
import json
import os

import cv2
import numpy as np

from general import xywh2xyX4, adjust_corner, xyX42xywh, denormalize, normalize

class Yolo:
    def load_imgs_annos(self, images_file=None, annotations_file=None):
        self.imgs, self.anns = dict(), dict() 
        if images_file is not None:
            print('loading images file...')
            imgs = {}
            for img_path in images_file:
                img_info = {}
                img = cv2.imread(img_path)
                img_name = os.path.basename(img_path)
                img_id = os.path.splitext(img_name)[0]

                img_info['img_name'] = img_name                
                img_info['path'] = img_path
                img_info['height'] = img.shape[0]
                img_info['width'] = img.shape[1]

                imgs[img_id] = img_info
            
            self.imgs = imgs
            print('Done')

        if annotations_file is not None:
            print('loading annotations file...')
            anns = defaultdict(list)
            for ann_path in annotations_file:
                img_id = os.path.splitext(os.path.basename(ann_path))[0]
                with open(ann_path, 'r') as f:
                    ann = f.readlines()
                if ann != []:
                    for info in ann:
                        ann_info = {}

                        cat_id, bbox = info.split(' ', 1)
                        cat_id = int(cat_id)
                        bbox = bbox.split(' ')
                        bbox = list(map(float, bbox))

                        ann_info['cat_id'] = cat_id
                        ann_info['bbox'] = bbox

                        anns[img_id].append(ann_info)
                else:
                    ann_info = {}
                    ann_info['cat_id'] = []
                    ann_info['bbox'] = []

                    anns[img_id].append(ann_info)
            
            self.anns = anns
            print('Done')

        diff = set(imgs) ^ set(anns)
        if diff != set():
            raise IndexError(f'画像データとアノテーションデータの数が一致していません\nimg_id:{diff}')
    
    def load_json(self, *json_paths):
        self.imgs, self.anns = dict(), dict()
        print('loading json file...')
        for json_path in json_paths:
            with open(json_path, mode='r') as j:
                info = json.load(j)

            info_img_id = list(info['images'])
            img_id = list(self.imgs)
            inter = set(info_img_id) & set(img_id)
            if inter == set():
                self.imgs.update(info['images'])
                self.anns.update(info['annotations'])
            else:
                raise IndexError(f'img_idが重複しています\nimg_id:{inter}')
            
        print('Done')

    def create_json(self, json_path):
        info = {'images': self.imgs, 'annotations': self.anns}
        with open(json_path, mode='w') as f:
            json.dump(info, f, indent=1)

    def get_imgid(self):
        return list(self.imgs)
    
    def get_impath(self, img_id):
        return self.imgs[img_id]['path']
    
    def get_shape(self, img_id):
        width = self.imgs[img_id]['width']
        height = self.imgs[img_id]['height']
        return width, height
    
    def load_img(self, img_id):
        img_path = self.get_impath(img_id)
        img = cv2.imread(img_path)

        return img

    def load_anns(self, img_id):
        anns = self.anns[img_id]

        return anns

    def save(self, img_id, output, draw_bbox=False):
        image_path = os.path.join(output, 'images')
        label_path = os.path.join(output, 'labels')

        os.makedirs(output, exist_ok=True)
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        if not os.path.exists(label_path):
            os.mkdir(label_path)

        img_info = self.imgs[img_id]
        img = cv2.imread(img_info['path'])

        if draw_bbox:
            width = img_info['width']
            height = img_info['height']
            for ann in self.anns[img_id]:
                de_ann = []
                for point, scale in zip(ann['bbox'], (width, height, width, height)):
                    de_ann.append(round(point * scale))
                corner = xywh2xyX4(de_ann)
                cv2.rectangle(img, corner[0], corner[2], (255, 0, 0))

        cv2.imwrite(os.path.join(image_path, img_info['img_name']), img)

        anns_info = self.anns[img_id]
        anns = []
        for ann_info in anns_info:
            bbox = ' '.join(map(str, ann_info['bbox']))
            ann = str(ann_info['cat_id']) + ' ' +bbox
            anns.append(ann)
        with open(os.path.join(label_path, img_id+'.txt'), mode='w') as f:
            f.write('\n'.join(anns))
    
    def show(self, img_id, draw_bbox=False):
        img_info = self.imgs[img_id]
        img_path = img_info['path']
        img = cv2.imread(img_path)
        
        if draw_bbox:
            width = img_info['width']
            height = img_info['height']
            for ann in self.anns[img_id]:
                de_ann = denormalize(ann['bbox'], width, height)
                corner = xywh2xyX4(de_ann)
                cv2.rectangle(img, corner[0], corner[2], (255, 0, 0))

        cv2.imshow(img_id, img)
        cv2.waitKey(0)
        cv2.destroyWindow(img_id)

    def trans_img(self, img_id, trans, height=None, width=None):
        img = self.load_img(img_id)
        if height is None:
            height = self.imgs[img_id]['height']
        if  width is None:
            width =self.imgs[img_id]['width']
        transed_img =cv2.warpAffine(img, trans, (width, height))

        return transed_img

    def trans_ann(self, img_id, trans):
        anns = self.load_anns(img_id)
        transed_anns = []
        np_trans = np.array(trans)
        width =self.imgs[img_id]['width']
        height = self.imgs[img_id]['height']

        for ann in anns:
            bbox = ann['bbox']
            cat_id = ann['cat_id']
            transed_ann = {}
            de_bbox = denormalize(bbox, width, height)
            corner = xywh2xyX4(de_bbox)

            np_corner = np.array(corner)
            np_corner = np.append(np_corner, np.ones((4, 1)), axis=1)
            np_transed_corner = np.dot(np_trans, np_corner.T)
            transed_corner = np_transed_corner.T.tolist()

            adj_corner = adjust_corner(transed_corner, width, height)
            transed_bbox = xyX42xywh(adj_corner)
            nor_bbox = normalize(transed_bbox, width, height)
            transed_ann['cat_id'] = cat_id
            transed_ann['bbox'] = nor_bbox
            transed_anns.append(transed_ann)

        return transed_anns
    
    def resize(self, img_id, size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR):
        img = self.load_img(img_id)
        resize_img = cv2.resize(img, size, fx, fy, interpolation)

        return resize_img