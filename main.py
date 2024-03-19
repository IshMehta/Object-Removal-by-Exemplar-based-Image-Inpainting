import cv2
import numpy as np
import os
import errno
import time

from object_removal import objectRemoval



# change these folders as needed
SOURCE_FOLDER = "images/source/"
OUT_FOLDER = "images/result/"


if __name__ == "__main__":

    # make the images/results folder
    output_dir = os.path.join(OUT_FOLDER)
    
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    
    for i in range(1,4):
        # image names should be input_1, input_2, input_3
        start = time.time()
        print("\nProcessing file set", i)
        image_name = SOURCE_FOLDER + 'input_' + str(i) + '.png'
        mask_name = SOURCE_FOLDER + 'mask_' + str(i) + '.png'

        image = cv2.imread(image_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)  # one channel only
        
        print(image_name)
        print('image', image.shape, image.size / 3e3, 'kB')
        print('mask', mask.shape)
        
        output = objectRemoval(image, mask, setnum=i, window=(9,9))
        cv2.imwrite(OUT_FOLDER + 'result_' + str(i) + '.png', output)
        end = time.time()
        print('image completed, elapsed time:', np.round(end-start,3))
