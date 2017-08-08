import csv
import os
from PIL import Image

# CIFAR-10 classes that we want to keep
CLASSES = ['cat', 'dog', 'deer', 'bird', 'horse', 'frog']

IMAGE_COUNTER = 1

# open csv file containing id -> class
with open('raw/food/trainLabels.csv', 'rb') as f:
    READER = csv.reader(f)

    # iterate over rows
    for row in READER:
        # check if it's a class that we want to keep
        if row[1] in CLASSES:
            # load image and save as jpg
            im = Image.open('raw/food/%s.png' % row[0])

            output_filename = "%s.jpg" % IMAGE_COUNTER
            im.save(os.path.join('data', 'others', output_filename), \
            "JPEG", quality=70)

            # increase image counter
            IMAGE_COUNTER = IMAGE_COUNTER + 1
