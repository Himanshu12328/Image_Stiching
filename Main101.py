import cv2
from utils import corners, show, describe_feature, match_features, visualize_matches, generate_keypoints


def __main__():


    img1 = cv2.imread('data/1.jpg')
    img2 = cv2.imread('data/2.jpg')
    
    img1_corners = corners(img1, 1, 100, 150)
    img2_corners = corners(img2, 1, 100, 150)

    img1_keypoints = generate_keypoints(img1_corners)
    img2_keypoints = generate_keypoints(img2_corners)

    descriptors1 = describe_feature(img1, img1_keypoints)
    descriptors2 = describe_feature(img2, img2_keypoints)

    matches = match_features(descriptors1, descriptors2)

    visualize_matches(img1, img1_keypoints, img2, img2_keypoints, matches)

    # show(img1_corner)

if __name__ == '__main__':
    __main__()