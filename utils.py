import cv2
import numpy as np
import scipy
from numba import jit
import sys
import matplotlib.pyplot as plt

            
    
def match_features(descriptors1, descriptors2):
    matches = []
    for i in range(len(descriptors1)):
        best_match = -1
        second_best_match = -1
        min_sq_diff = float('inf')

        for j in range(len(descriptors2)):
            sq_diff = np.sum(np.square(descriptors1[i] - descriptors2[j]))

            if sq_diff < min_sq_diff:
                second_best_match = best_match
                best_match = j
                min_sq_diff = sq_diff

        if min_sq_diff < 0.75:  # Ratio test
            ratio = min_sq_diff / np.sum(np.square(descriptors1[i] - descriptors2[second_best_match]))
            if ratio < 0.75:  # Keep matched pair based on ratio
                matches.append((i, best_match))

    return matches

def feature_descriptors(img, img_g, Nbest_corners, patch_size):
    descriptors = []
    x = Nbest_corners[:,1]
    y = Nbest_corners[:,2]

    for i in range(len(Nbest_corners)):
        y_i = x[i]          #reverse the co-ordinates again
        x_i = y[i]
        gray = copy.deepcopy(img_g)
        gray = np.pad(img_g, ((patch_size, patch_size), (patch_size, patch_size)), mode='constant', constant_values=0) #pad the image by 40 on all sides
        x_start = int(x_i + patch_size/2)
        y_start = int(y_i + patch_size/2)
        descriptor = gray[x_start: x_start+patch_size, y_start:y_start+patch_size]  #40X40 descriptor pf one point
        descriptor = cv2.GaussianBlur(descriptor, (7, 7), cv2.BORDER_DEFAULT)
        descriptor = descriptor[::5, ::5]
        descriptor_1 = descriptor.reshape((64, 1))
        descriptor_std = (descriptor_1 - descriptor_1.mean())/descriptor_1.std()
        descriptors.append(descriptor_std)  

    return descriptors


@jit
def ANMS(img, img_h, n_best, coords):
    num = len(coords)
    inf = sys.maxsize
    r = inf * np.ones((num,3))
    ED = 0
    for i in range(num):
        for j in range(num):
            x_i = coords[i][1]              #We take x_cordinate of one corner point
            y_i = coords[i][0]              #We take x_cordinate of one corner point
            neighbors_x = coords[j][1]      #x_cordinate of other points
            neighbors_y = coords[j][0]

            if img_h[y_i, x_i] > img_h[neighbors_y, neighbors_x]:
                ED = (neighbors_x - x_i)**2 + (neighbors_y - y_i)**2

            if ED < r[i, 0]:
                r[i, 0] = ED
                r[i, 1] = x_i
                r[i, 2] = y_i
    arr = r[:,0]
    feature_sorting = np.argsort(-arr)      #We get the index of biggest that is the resaon of -ve sign (Descending order index)
    feature_coord = r[feature_sorting]
    Nbest_corners = feature_coord[:n_best,:] #We also can find min of (n_best, num_of_feature_cordinates we got)
    for i in range(len(Nbest_corners)):
        cv2.circle(img, (int(Nbest_corners[i][1]), int(Nbest_corners[i][2])), 3, 255, -1)
    plt.imshow(img)
    plt.savefig("anms.png")
    return Nbest_corners


def corners(img, img_g, choice):
    '''
    Corner Detector to find corners of an image, Choice 1 = Shi-Tomasi Choice 2 = Harris
    '''

    if choice == 1:
        dst = cv2.goodFeaturesToTrack(img_g, 1000, 0.05, 10)
        img_h = cv2.dilate(dst, None, iterations=2)
        
        

    elif choice == 2:
        img_g = np.float32(img_g)
        dst = cv2.cornerHarris(img_g, 2, 3, 0.04)
        img_h = cv2.dilate(dst, None, iterations=2)
        # img[img_h > 0.001 * img_h.max()] = [255,0,0]
	    # plt.imshow(dst)
	    # plt.show()
    


    else:
        print("Wrong choice entered for corner detection method")
        return None

    lm = scipy.ndimage.maximum_filter(img_h, 10)
    msk = (img_h == lm)
    ls = scipy.ndimage.minimum_filter(img_h, 10)
    diff = ((lm - ls) > 20000)
    msk[diff == 0] = 0
    img[img_h > 0.01*img_h.max()] = [255, 0, 0]
    # plt.imshow(img)
	# plt.savefig("harris.png")
	# plt.show()
    coords = []
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            if msk[i][j] == True:
                coords.append((i,j))

    return coords, dst
    

def generate_keypoints(corners):
    # corners_img = corners(img, choice, Nbest, max_pts)

    keypoints = [cv2.KeyPoint(x=float(kp[0][0]), y=float(kp[0][1]), size=20) for kp in corners]
    return keypoints

# def corners(img, choice, Nbest):
#     '''
#     Corner Detector to find corners of an image, Choice 1 = Shi-Tomasi Choice 2 = Harris
#     '''

#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     if (choice == 1):
#         '''
#         Shi-Tomasi corner detection
#         '''
#         corners_img = cv2.goodFeaturesToTrack(gray_img, 1000, 0.05, 10)
#         corners_img = np.int0(corners_img)
        
#         for corners in corners_img:
#             x, y = corners.ravel()
#             cv2.circle(img, (x, y), 3, [0,255,0], -1)


#     elif (choice == 2):
#         '''
#         Harris corner detection
#         '''
#         gray_img = np.float32(gray_img)

#         corners_img = cv2.cornerHarris(gray_img, 3, 3, 0.04)

#         img[corners_img>0.001*corners_img.max()] = [0, 255, 0]

#     else:
#         print("Wrong choice entered for corner detection method")
#         return img

#     Cimg = cv2.cornerMinEigenVal(gray_img, blockSize=3, ksize=3)
#     Cimg_norm = cv2.normalize(Cimg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#     _, thresh = cv2.threshold(Cimg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     local_maxima = np.zeros_like(Cimg_norm, np.uint8)
#     local_maxima[thresh == 255] = 255
#     coordinates = np.column_stack(np.where(local_maxima != 0))
#     # local_maxima = cv2.dilate((Cimg == cv2.dilate(Cimg, None)), None)
#     # coordinates = np.argwhere(local_maxima > 0)

#     Nstrong = len(coordinates)
#     r = np.full(Nstrong, np.inf)

#     for i in range(Nstrong):
#         for j in range(Nstrong):
#             if Cimg[coordinates[j][0], coordinates[j][1]] > Cimg[coordinates[i][0], coordinates[i][1]]:
#                 ED = (coordinates[j][1] - coordinates[i][1])**2 + (coordinates[j][0] - coordinates[i][0])**2
#                 if ED < r[i]:
#                     r[i] = ED

#     sorted_indices = np.argsort(r)[::-1][:Nbest]
#     selected_coordinates = coordinates[sorted_indices]

#     for coord in selected_coordinates:
#         x, y = coord[1], coord[0]
#         cv2.circle(img, (x, y), 3, [255, 0, 0], -1)
    
    
#     return img


def visualize_matches(img1, keypoints1, img2, keypoints2, matches):
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show(img):
    '''
    Display image
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
