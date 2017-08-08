# Processes all images into size of 32*32 pixels for the network =
from glob import glob  # used for finding the path names
import os
import os.path
from PIL import Image

SIZE = 32, 32

# set directory
os.chdir('train/hotdogs')

# filter all jpg and png images
IMAGE_FILES = glob('*.jpg')
IMAGE_FILES.extend(glob('*.jpeg'))
IMAGE_FILES.extend(glob('*.png'))

# Image files stores a list of all the files ending with that one.

IMAGE_COUNTER = 1

for image_file in IMAGE_FILES:
    # open file and resize
    try:
        im = Image.open(image_file)
        im = im.resize(SIZE, Image.ANTIALIAS)
        output_filename = "%s.jpg" % IMAGE_COUNTER
        im.save(os.path.join('..', 'processed_dogs', output_filename), \
                "JPEG", quality=70)

        IMAGE_COUNTER = IMAGE_COUNTER + 1
    except:
        IMAGE_COUNTER = IMAGE_COUNTER + 1
        # save locally

if __name__ == "__main__":
    pass
