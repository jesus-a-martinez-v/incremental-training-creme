import argparse
import os
import random

import numpy as np
from imutils import paths
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to the input dataset.')
argument_parser.add_argument('-c', '--csv', required=True, help='Path to the output csv file.')
argument_parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size for the network')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading VGG-16 network...')
model = VGG16(weights='imagenet', include_top=False)
batch_size = arguments['batch_size']

image_paths = list(paths.list_images(arguments['dataset']))
random.seed(84)
random.shuffle(image_paths)

labels = [path.split(os.path.sep)[-2] for path in image_paths]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

csv_columns = ['class'] + [f'feature_{i}' for i in range(512 * 7 * 7)]

with open(arguments['csv'], 'w') as f:
    f.write(f'{",".join(csv_columns)}\n')

    for batch_number, index in enumerate(range(0, len(image_paths), batch_size)):
        print(f'[INFO] Processing batch {batch_number + 1}/{int(np.ceil(len(image_paths) / float(batch_size)))}')

        batch_paths = image_paths[index: index + batch_size]
        batch_labels = labels[index: index + batch_size]
        batch_images = []

        for image_path in batch_paths:
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)

            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            batch_images.append(image)

        batch_images = np.vstack(batch_images)
        features = model.predict(batch_images, batch_size=batch_size)
        features = features.reshape((features.shape[0], 7 * 7 * 512))

        for label, vector in zip(batch_labels, features):
            vector = ','.join([str(v) for v in vector])
            f.write(f'{label},{vector}')
