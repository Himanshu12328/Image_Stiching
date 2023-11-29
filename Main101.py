import cv2
import copy
from utils import ANMS, corners, feature_descriptors


def __main__():

    images = []
    gray_images = []

    for i in range (1, 6):
        img = cv2.imread('data/{i}.jpg')
        images.append(img)
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(img_g)

    img_count = len(images)
    corners = 1000
    match_ratio = 0.4
    threshold = 30
    c1 = 0
    c2 = 0
    c3 = 0
    for j in range(img_count-1):
        img_desc = []
        best_corners = []
        c2 = j+1
        imgs = [images[c1], images[c2]]
        print("Matching image", c1+1, "and", c2+1)
        gray_imgs = [gray_images[c1], gray_images[c2]]
        for i in range(2):
            img = imgs[i]
            img_g = gray_imgs[i]
            coords, img_h = corners(copy.deepcopy(img), img_g, 2)
            Nbest_corners = ANMS(copy.deepcopy(img), img_h, corners, coords)
            best_corners.append(Nbest_corners)
            feature_vectors = feature_descriptors(copy.deepcopy(img), img_g, Nbest_corners, 40)
            img_desc.append(feature_vectors)

    


    
    # img1_corners = corners(img1, 1, 100, 150)
    # img2_corners = corners(img2, 1, 100, 150)

    # img1_keypoints = generate_keypoints(img1_corners)
    # img2_keypoints = generate_keypoints(img2_corners)

    # descriptors1 = describe_feature(img1, img1_keypoints)
    # descriptors2 = describe_feature(img2, img2_keypoints)

    # matches = match_features(descriptors1, descriptors2)

    # visualize_matches(img1, img1_keypoints, img2, img2_keypoints, matches)

    # show(img1_corner)

if __name__ == '__main__':
    __main__()