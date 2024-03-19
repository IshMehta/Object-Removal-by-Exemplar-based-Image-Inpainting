
import numpy as np
import scipy as sp
import cv2
import scipy.signal                     
from matplotlib import pyplot as plt     



def objectRemoval(image, mask, setnum=0, window=(9,9)):
    """ ALL IMAGES SUPPLIED OR RETURNED ARE UINT8. 
    This function will be called three times; once for each 
    image/mask pairs to produce result images.
    
    
    Parameters
    ----------
    image : numpy.ndarray (dtype=uint8)
        Three-channel color image of shape (r,c,3)
        
    mask: numpy.array (dtype=uint8)
        Single channel B&W image of shape (r,c) which defines the
        target region to be removed.
        
    setnum: (integer)
        setnum=0 is for use with the autograder
        setnum=1,2,3 are your three required image sets
 
    window: default = (9, 9), tuple of two integers
        The dimensions of the target window described at the beginning of
        Section III of the paper.
        You can optimize the window for each image. If you do this, 
        then include the final value in your code, where it can be activated 
        by setnum. Then when we run your code, we should get the same results.
        
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        Three-channel color image of same shape as input image (r, c, 3)
        Make sure you deal with any needed normalization or clipping, so that
        your image array is complete on return.
    """

    # WRITE YOUR CODE HERE.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.uint8) 

    if setnum == 0:   
        patch_height = window[0]    
        patch_width = window[1]
    if setnum == 1:
        # patch_height = 43    
        # patch_width = 23
        patch_height = 13 
        patch_width = 11
    if setnum == 2:
        # patch_height = 15    
        # patch_width = 21
        patch_height = 34    
        patch_width = 34
    if setnum == 3:
        patch_height = 21    
        patch_width = 37
    
    
    mask = np.where(cv2.GaussianBlur(mask, (3,3), 0)<128, 0, 255).astype(np.uint8)
   


    # Find regions
    target_region = np.where(mask > 128)
    source_region = np.where(mask < 128)

   

    # Confidence initialisaiton
    confidence = np.ones((mask.shape[0], mask.shape[1]))
    confidence[target_region] = 0

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F,  dx = 1, dy = 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F,  dx = 0, dy = 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
    isophote_vectors = np.dstack((grad_y, -grad_x))
    isophote_vectors /= np.linalg.norm(isophote_vectors)

    # Find contour front using LoG
    edge = cv2.Canny( mask, 200, 255) 
    contour_front, hireachy = cv2.findContours(edge, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
    
    # cv2.drawContours(image, contour_front, -1, (0,255,0), 1)
    contours = np.vstack(contour_front).squeeze()
    contours = contours[: ,::-1]


     # apply mask on image to black out source region
    working_image = image.copy()
    working_image[target_region] = 255



    # Data Term initialisation
    # First find the normal vector
    data = np.zeros((mask.shape[0], mask.shape[1]))
    grad_x_frontier = cv2.Sobel(mask, cv2.CV_64F,  dx = 1, dy = 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    grad_y_frontier = cv2.Sobel(mask, cv2.CV_64F,  dx = 0, dy = 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
    normal_vector = np.dstack((grad_x_frontier, grad_y_frontier)) 
    normal_vector /= np.linalg.norm(normal_vector)
    data = np.zeros((mask.shape[0], mask.shape[1]))
    for pixel in contours:
        data[pixel[0], pixel[1]] = 1
    data = np.multiply(data, np.abs(np.sum(normal_vector * isophote_vectors, axis=-1))/255.0)


    # Priority Initialisation
    priority = np.zeros((mask.shape[0], mask.shape[1]))
    for pixel in contours:
        priority[pixel[0], pixel[1]] = 1
    priority = np.multiply(priority,  np.multiply(data, confidence))

    
    while len(contours)!=0:
        # find highest priorty pixel
        highest_priority  = -np.inf
        highest_priority_pixel = None
        for index in contours:
            if priority[index[0], index[1]] > highest_priority:
                highest_priority = priority[index[0], index[1]]
                highest_priority_pixel = index

        highest_priority_patch = get_patch(highest_priority_pixel, patch_height , patch_width , image.shape)
        source_patch = find_source(highest_priority_patch, working_image, image,  patch_height, patch_width, mask)
        

        # Once we have found the source patch, we update the image
        mask[highest_priority_patch[0][0] : highest_priority_patch[0][1], 
              highest_priority_patch[1][0] : highest_priority_patch[1][1]] = 0

        image[highest_priority_patch[0][0] : highest_priority_patch[0][1], 
              highest_priority_patch[1][0] : highest_priority_patch[1][1], :] = image[source_patch[0][0] : source_patch[0][1], source_patch[1][0] : source_patch[1][1], :]
        # update confidence
        confidence[highest_priority_patch[0][0] : highest_priority_patch[0][1], 
              highest_priority_patch[1][0] : highest_priority_patch[1][1]] = np.sum(confidence[source_patch[0][0] : source_patch[0][1], source_patch[1][0] : source_patch[1][1]])/(patch_height*patch_width)
        
        


        
        # calculate frontier again by first changing mask then finding contours again

        mask = np.where(cv2.GaussianBlur(mask, (5,5), 0)<128, 0, 255).astype(np.uint8)
        edge = cv2.Canny( mask, 150, 255) 
        contour_front, hireachy = cv2.findContours(edge, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
        
        if len(contour_front) != 0:
            contours = np.vstack(contour_front).squeeze()
            contours = contours[: ,::-1]
        else:
            contours = []


        # update data
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F,  dx = 1, dy = 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F,  dx = 0, dy = 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
        isophotes_magnitude = np.sqrt(np.square(grad_x/255) + np.square(grad_y/255))
        isophote_vectors = np.dstack((grad_y, -grad_x))
        # isophote_vectors = np.asarray([grad_y, -grad_x])

        grad_x_frontier = cv2.Sobel(mask, cv2.CV_64F,  dx = 1, dy = 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
        grad_y_frontier = cv2.Sobel(mask, cv2.CV_64F,  dx = 0, dy = 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
        normal_vector = np.dstack((grad_x_frontier, grad_y_frontier))
        data = np.zeros((mask.shape[0], mask.shape[1]))
        for pixel in contours:
            data[pixel[0], pixel[1]] = 1
        data = np.multiply(data, np.abs(np.sum(normal_vector * isophote_vectors, axis=-1)/255))

        
        
        # Priority Initialisation
        priority = np.zeros((mask.shape[0], mask.shape[1]))
        for pixel in contours:
            priority[pixel[0], pixel[1]] = 1
        priority = np.multiply(priority,  np.multiply(data, confidence))


    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
    


def get_patch(point, patch_height,  patch_width, imageShape):
    

    patch = [
        [point[0] - (patch_height//2), point[0] + (patch_height//2) +1 ],
        [point[1] - (patch_width//2),  point[1] + (patch_width//2) + 1] 
        
    ] 

    return np.array(patch)
    

def find_source_patch(patch, mask, image, patch_height, patch_width):

    minSSD = 0
    minFound = None
    for y in range(mask.shape[0] - patch_height + 1):
        for x in range(mask.shape[1] - patch_width + 1):
            potential_patch_coordinates = [[y, y+patch_height ], [x, x+patch_width]]
            # print(potential_patch_coordinates)

            # make sure this patch is not in the target region -> check if sum of patch in mask is == 0
            if mask[potential_patch_coordinates[0][0] : potential_patch_coordinates[0][1] , potential_patch_coordinates[1][0] : potential_patch_coordinates[1][1] ].sum() == 0:
                ssd = np.sum((image[potential_patch_coordinates[0][0] : potential_patch_coordinates[0][1], 
                                   potential_patch_coordinates[1][0] : potential_patch_coordinates[1][1], :] - 
                                   image[patch[0][0] : patch[0][1], patch[1][0]:patch[1][1], :])**2)
                if minFound is None:
                    minFound = potential_patch_coordinates
                    minSSD = ssd
                elif ssd < minSSD:
                    minSSD = ssd
                    minFound = potential_patch_coordinates


    return minFound

def find_source(patch, working_image, image, patch_height, patch_width, mask):
    # Match patch using SSD

    temp_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
    temp_mask[np.where(mask > 128)] = 0
    temp_mask[np.where(mask < 128)] = 1
    img_copy = image.copy()
    result = cv2.matchTemplate(working_image, image[patch[0][0] : patch[0][1] , patch[1][0]:patch[1][1] , :] , cv2.TM_SQDIFF, mask=temp_mask[patch[0][0] : patch[0][1] , patch[1][0]:patch[1][1]])
    # Find location of best match
    result[np.where(mask > 128)] = np.inf   # set mask area to inf
    result[patch[0][0] : patch[0][1] , patch[1][0]:patch[1][1]] = np.inf # set patch area to inf

    min_val, _, min_loc, _ = cv2.minMaxLoc(result)
    temp = [[min_loc[1], min_loc[1] + patch_height ], [min_loc[0], min_loc[0] + patch_width ]]
    plt.imshow(image[patch[0][0] : patch[0][1] , patch[1][0]:patch[1][1] , :])

    return  temp




    
 
    