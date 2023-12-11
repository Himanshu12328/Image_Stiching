import cv2
import numpy as np
import scipy
from numba import jit
import sys
import matplotlib.pyplot as plt
import random
import copy

def keypoint(points):
    kp1 = []
    for i in range(len(points)):
        kp1.append(cv2.KeyPoint(int(points[i][0]), int(points[i][1]), 3))
        return kp1
    
def matches(points):
    m = []
    for i in range(len(points)):
        m.append(cv2.DMatch(int(points[i][0]), int(points[i][1]), 2))
        return m
    
def draw_matches(images, matched_pairs):
    img1 = copy.deepcopy(images[0])
    img2 = copy.deepcopy(images[1])
    key_points_1 = [x[0] for x in matched_pairs]
    keypoints1 = keypoint(key_points_1)
    key_points_2 = [x[1] for x in matched_pairs]
    keypoints2 = keypoint(key_points_2)
    matched_pairs_idx = [(i,i) for i,j in enumerate(matched_pairs)]
    matches1to2 = matches(matched_pairs_idx)
    out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, None, flags =2)
    plt.imshow(out)
    plt.show()
	
def wraptwoimages(images, H):
    img1 = copy.deepcopy(images[1])
    img2 = copy.deepcopy(images[0])
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    pts_2 = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts_2), axis=0)
    [Xmin, Ymin] = np.int32(pts.min(axis=0).ravel())
    [Xmax, Ymax] = np.int32(pts.max(axis=0).ravel())
    t = [-Xmin, -Ymin]
    Ht = np.array([[1, 0, t[0]], [0,1,t[1]], [0, 0, 1]])
    result = cv2.warpPerspective(img2, Ht.dot(H), (Xmax-Xmin, Ymax-Ymin), flags=cv2.INTER_LINEAR)
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def dot_product(h_mat, keypoint):
    keypoint = np.expand_dims(keypoint, 1)
    keypoint = np.vstack([keypoint, 1])
    product = np.dot(h_mat, keypoint)
    if product[2] !=0:
        product = product/product[2]
    
    return product[0:2,:]

def homography(point1, point2):
    h_matrix = cv2.getPerspectiveTransform(np.float32(point1), np.float32(point2))
    return h_matrix

def ransac(matched_pairs, threshold):
    inliers = []
    COUNT = []
    for i in range(1000):
        keypoints1 = [x[0] for x in matched_pairs]
        keypoints2 = [x[1] for x in matched_pairs]
        length = len(keypoints1)

        randomlist = random.sample(range(0, length), 4)
        points_1 = [keypoints1[idx] for idx in randomlist]
        points_2 = [keypoints2[idx] for idx in randomlist]

        h_matrix = homography(points_1, points_2)
        points = []
        count_inliers = 0
        for i in range(length):
            a = (np.array(keypoints2[i]))
            ssd = np.linalg.norm(np.expand_dims(np.array(keypoints2[i]), 1) - dot_product(h_matrix, keypoints1[i]))
            if ssd < threshold:
                count_inliers += 1
                points.append((keypoints1[i], keypoints2[i]))
        COUNT.append(-count_inliers)
        inliers.append((h_matrix, points))
    max_count_idx = np.argsort(COUNT)
    max_count_idx = max_count_idx[0]
    final_matched_pairs = inliers[max_count_idx][1]

    pts_1 = [x[0] for x in final_matched_pairs]
    pts_2 = [x[1] for x in final_matched_pairs]
    h_final_matrix, status = cv2.findHomography(np.float32(pts_1), np.float32(pts_2))
    return h_final_matrix, final_matched_pairs
           
    
def feature_matching(imgs, gray_imgs, img_desc, best_corners, match_ratio):
    f1 = img_desc[0]
    f2 = img_desc[1]
    corners1 = best_corners[0]
    corners2 = best_corners[1]
    matched_pairs = []
    for i in range(0, len(f1)):
        sqr_diff = []
        for j in range(0, len(f2)):
            diff = np.sum((f1[i] - f2[j])**2)
            sqr_diff.append(diff)
        sqr_diff = np.array(sqr_diff)
        diff_sort = np.argsort(sqr_diff)
        sqr_diff_sort = sqr_diff[diff_sort]
        ratio = sqr_diff_sort[0]/(sqr_diff_sort[1])
        if ratio < match_ratio:
            matched_pairs.append((corners1[i,1:3], corners2[diff_sort[0], 1:3]))

    return matched_pairs

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
            neighbours_x = coords[j][1]      #x_cordinate of other points
            neighbours_y = coords[j][0]

            if img_h[y_i, x_i] > img_h[neighbours_y, neighbours_x]:
                ED = (neighbours_x - x_i)**2 + (neighbours_y - y_i)**2

            if ED < r[i, 0]:
                r[i, 0] = ED
                r[i, 1] = x_i
                r[i, 2] = y_i
    arr = r[:,0]
    feature_sorting = np.argsort(-arr)      #We get the index of biggest that is the resaon of -ve sign (Descending order index)
    feature_cord = r[feature_sorting]
    Nbest_corners = feature_cord[:n_best,:] #We also can find min of (n_best, num_of_feature_cordinates we got)
    for i in range(len(Nbest_corners)):
        cv2.circle(img, (int(Nbest_corners[i][1]), int(Nbest_corners[i][2])), 3, 255, -1)
    plt.imshow(img)
    plt.savefig("anms.png")
    return Nbest_corners


def img_corners(img, img_g, choice):
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
    plt.imshow(img)
	plt.savefig("harris.png")
	plt.show()
    coords = []
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            if msk[i][j] == True:
                coords.append((i,j))

    return coords, dst
    
def show(img):
    '''
    Display image
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
