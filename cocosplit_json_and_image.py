import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import time
import os
import glob
import shutil


def stop():
	time.sleep(999)

def save_coco(file, images, annotations, categories):
# def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)
        # json.dump({ """'info': info, 'licenses': licenses,""" 'images': images, 
        #    'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):

    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def read_json_and_batch_image(json_path, save_image_folder_path, image_dir_path):
    save_image_folder_path = save_image_folder_path + "/"
    image_dir_path = image_dir_path + "/"
    image_list = glob.glob(image_dir_path+"*.JPG")
    file_name_list = [ os.path.basename(item) for item in image_list]

    coco = None
    with open(json_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        #info = coco['info']
        #licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        for image in images:
            if image['file_name'] in file_name_list:

                image['path'] = save_image_folder_path + image['file_name']
                shutil.copy2(image_dir_path + image['file_name'], save_image_folder_path)

    with open(json_path, 'wt', encoding='UTF-8') as f:
        json.dump(coco, f, indent=2, sort_keys=True)    


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
# parser.add_argument('train', type=str, help='Where to store COCO training annotations')
# parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('coco_dir_path', type=str, help='Where to organize folders.')
parser.add_argument('image_dir_path', type=str, help='image directory path.')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                    help='Split a multi-class dataset while preserving class distributions in train and test sets')



args = parser.parse_args()

def main(args):
    if args.coco_dir_path[-1] == "/":
        args.coco_dir_path = args.coco_dir_path[:-1]

    if args.image_dir_path[-1] == "/":
        args.image_dir_path = args.image_dir_path[:-1]

    try:
        if not os.path.exists(args.coco_dir_path):
            os.makedirs(args.coco_dir_path)
            os.makedirs(args.coco_dir_path + "/annotations")
            os.makedirs(args.coco_dir_path + "/train")
            os.makedirs(args.coco_dir_path + "/val")
    except OSError:
        print("Error: Creating directory. " + args.coco_dir_path)

    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        #info = coco['info']
        #licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        train_json_path = args.coco_dir_path + "/annotations/train.json"
        val_json_path = args.coco_dir_path + "/annotations/val.json"
   
        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        if args.multi_class:

            annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)

            #bottle neck 1
            #remove classes that has only one sample, because it can't be split into the training and testing sets
            annotation_categories =  funcy.lremove(lambda i: annotation_categories.count(i) <=1  , annotation_categories)

            annotations =  funcy.lremove(lambda i: i['category_id'] not in annotation_categories  , annotations)


            X_train, y_train, X_test, y_test = iterative_train_test_split(np.array([annotations]).T,np.array([ annotation_categories]).T, test_size = 1-args.split)

            save_coco(train_json_path, filter_images(images, X_train.reshape(-1)), X_train.reshape(-1).tolist(), categories)
            save_coco(val_json_path, filter_images(images, X_test.reshape(-1)), X_test.reshape(-1).tolist(), categories)
            # save_coco(args.train, info, licenses, filter_images(images, X_train.reshape(-1)), X_train.reshape(-1).tolist(), categories)
            # save_coco(args.test, info, licenses, filter_images(images, X_test.reshape(-1)), X_test.reshape(-1).tolist(), categories)

            print("Saved {} entries in {} and {} in {}".format(len(X_train), train_json_path, len(X_test), val_json_path))
            
        else:

            X_train, X_test = train_test_split(images, train_size=args.split)

            anns_train = filter_annotations(annotations, X_train)
            anns_test=filter_annotations(annotations, X_test)


            save_coco(train_json_path, X_train, anns_train, categories)
            save_coco(val_json_path, X_test, anns_test, categories)
            # save_coco(args.train, X_train, anns_train, categories)
            # save_coco(args.test, X_test, anns_test, categories)
            # save_coco(args.train, info, licenses, X_train, anns_train, categories)
            # save_coco(args.test, info, licenses, X_test, anns_test, categories)

            print("Saved {} entries in {} and {} in {}".format(len(anns_train), train_json_path, len(anns_test), val_json_path))
    read_json_and_batch_image(train_json_path, args.coco_dir_path + "/train", args.image_dir_path)            
    read_json_and_batch_image(val_json_path, args.coco_dir_path + "/val", args.image_dir_path)


if __name__ == "__main__":
    main(args)
