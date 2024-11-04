import glob
import json
import os

import cv2
from sklearn.model_selection import train_test_split

from trans_matrix import trans_matrix
from yolo_api import *

if __name__ == '__main__':
    with open('./dataset.json') as f:
        dataset = json.load(f)

    yolo = Yolo()
    yolo.load_json(dataset['json_paths'])
    img_id = yolo.get_imid()

    train_rate = dataset['train']['train_rate']
    valid_rate = dataset['valid']['valid_rate']
    test_rate = dataset['test']['test_rate']
    if train_rate + valid_rate + test_rate != 1:
        raise ValueError('それぞれの割合の合計が1になっていません')
    train_imids, validtest_imids = train_test_split(img_id, train_size=train_rate, random_state=0)
    valid_imids, test_imids = train_test_split(validtest_imids, train_size=valid_rate/(valid_rate+test_rate), random_state=0)

    train_paths = yolo.get_impath(train_imids)
    valid_paths = yolo.get_impath(valid_imids)
    test_paths = yolo.get_impath(test_imids)

    with open(dataset['train']['train_txt'], mode='w') as g:
        g.write('\n'.join(train_paths))
    with open(dataset['valid']['valid_txt'], mode='w') as h:
        h.write('\n'.join(valid_paths))
    with open(dataset['test']['test_txt'], mode='w') as i:
        i.write('\n'.join(test_paths))
    
    if dataset['Data_Aug']['bool'] == 'True':
        data_aug = dataset['Data_Aug']

        img_path = os.path.join(data_aug['output'], 'images')
        txt_path = os.path.join(data_aug['output'], 'labels')
        os.makedirs(data_aug['output'], exist_ok=True)
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)

        for i in range(len(data_aug['processes'])):
            if data_aug['processes'][i]['rate'] == 1:
                aug = train_imids
            elif data_aug['processes'][i]['rate'] < 1:
                aug, rest = train_test_split(train_imids, train_size=data_aug['processes'][i]['rate'], random_state=0)
            else:
                raise ValueError('割合が1を超えています')
            
            trans = trans_matrix(data_aug['processes'][i])

            for img_id in aug:
                file_name = data_aug['processes'][i]['name'] + '_' + img_id
                img_name = file_name + '.jpg'
                txt_name = file_name + '.txt'

                transed_img = yolo.trans_img(img_id=img_id, trans=trans)
                cv2.imwrite(os.path.join(img_path, img_name), transed_img)

                transed_anns_info = yolo.trans_ann(img_id=img_id, trans=trans)
                with open(os.path.join(txt_path, txt_name), mode='w') as j:
                    anns = []
                    for ann_info in transed_anns_info:
                        ann = str(ann_info['cat_id']) + ' ' + ' '.join(map(str, ann_info['bbox']))
                        anns.append(ann)
                        j.write('\n'.join(anns))

            aug, rest = [], []

        img_file = glob.glob(os.path.join(img_path, '*.jpg'))
        txt_file = glob.glob(os.path.join(txt_path, '*.txt'))

        yolo_aug = Yolo()
        yolo_aug.load_imgs_annos(images_file=img_file, annotations_file=txt_file)
        yolo_aug.create_json(json_path=os.path.join(data_aug['output'], 'data_aug.json'))
        aug_paths = yolo.get_impath(yolo_aug.get_imid())
        with open(dataset['train']['train_txt'], mode='a') as k:
            k.write('\n'.join(aug_paths))
    
    elif dataset['Data_Aug']['bool'] == 'False':
        pass
    else:
        raise ValueError('True,Falseで指定してください')