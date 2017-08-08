import os
import os.path
import requests
import time
from PIL import Image
from glob import glob  # used for finding the path names
SIZE = 32, 32

os.chdir('train/nothotdogs')
IMAGE_FILES = glob('**/*.jpg', recursive=True)
IMAGE_FILES.extend(glob('**/*.jpeg',recursive=True))
IMAGE_FILES.extend(glob('**/*.png',recursive=True))
IMAGE_COUNTER = 1

for image_file in IMAGE_FILES:
    # open file and resize
    try:
        im = Image.open(image_file)
        im = im.resize(SIZE, Image.ANTIALIAS)
        output_filename = "%s.jpg" % IMAGE_COUNTER
        im.save(os.path.join('..', 'processed_notDogs', output_filename), \
                "JPEG", quality=70)

        IMAGE_COUNTER = IMAGE_COUNTER + 1
    except:
        IMAGE_COUNTER = IMAGE_COUNTER + 1
        # save locally

if __name__ == "__main__":
    pass
