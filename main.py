import cv2
import copy
from utils import ANMS, img_corners, feature_descriptors, feature_matching, ransac, wraptwoimages, draw_matches
import matplotlib.pyplot as plt

def __main__():

    images = []
    gray_images = []

    for i in range (1, 6):
        images.append(cv2.imread(f'data/{i}.jpg'))
        gray_images.append(cv2.cvtColor(images[i-1], cv2.COLOR_BGR2GRAY))

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
            coords, img_h = img_corners(copy.deepcopy(img), img_g, 2)
            Nbest_corners = ANMS(copy.deepcopy(img), img_h, corners, coords)
            best_corners.append(Nbest_corners)
            feature_vectors = feature_descriptors(copy.deepcopy(img), img_g, Nbest_corners, 40)
            img_desc.append(feature_vectors)

        matched_pairs = feature_matching(imgs, gray_imgs, img_desc, best_corners, match_ratio)
        print("Matched pairs", len(matched_pairs))
        # draw_matches(images, matched_pairs)
        if len(matched_pairs) > 20:
            c1 = c2
            final_h_mat, final_matched = ransac(matched_pairs, threshold)
            wraped = wraptwoimages(imgs, final_h_mat)
            plt.imshow(wraped)
            plt.show()
        else:
            c3 += 1
            continue

        images[c1] = wraped
        gray_images[c1] = cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    __main__()