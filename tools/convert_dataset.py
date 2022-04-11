import os
import argparse
import xml.etree.ElementTree as ET


VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')


def convert_voc(data_root, output_file, mode='trainval', ignore_difficult=True):
    assert mode in ('train', 'val', 'trainval', 'test')

    txt_file = os.path.join(data_root, f'ImageSets/Main/{mode}.txt')
    with open(txt_file) as f:
        img_ids = f.read().splitlines()

    class_to_idx = {_class: i for i, _class in enumerate(VOC_CLASSES)}
    anns = []
    for img_id in img_ids:
        filename = f'JPEGImages/{img_id}.jpg'
        xml_path = os.path.join(data_root, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = set()
        labels_difficult = set()
        for obj in root.findall('object'):
            label_name = obj.find('name').text
            if label_name not in VOC_CLASSES:
                continue
            label = class_to_idx[label_name]
            difficult = int(obj.find('difficult').text)
            if difficult:
                labels_difficult.add(label)
            else:
                labels.add(label)
        
        if not ignore_difficult:
            labels = labels.union(labels_difficult)
        else:
            labels_difficult -=labels
        gt_labels = ['0' for _ in range(len(VOC_CLASSES))]
        for l in list(labels):
            gt_labels[l] = '1'
        if mode == 'test' and ignore_difficult:
            for l in list(labels_difficult):
                gt_labels[l] = '-1'
        ann = filename + '\t' + ','.join(gt_labels)
        anns.append(ann)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(anns))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data/VOC2007', help='VOC data root')
    parser.add_argument('--out_file', default='./data/VOC2007/labels.txt', help='output label file')
    parser.add_argument('--mode', default='trainval', help='data subset')
    args = parser.parse_args()

    convert_voc(args.data_root, args.out_file, args.mode)
