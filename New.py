import cv2
import numpy as np

def resize_image(image):
    return cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def harris_corner_detection(gray_image):
    return cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

def dilate_corners(corners):
    return cv2.dilate(corners, None)

def find_local_maxima(corners, threshold):
    local_max = cv2.compare(corners, threshold, cv2.CMP_GT)
    coordinates = np.argwhere(local_max > 0)
    coordinates[:, [0, 1]] = coordinates[:, [1, 0]]  # Swap columns for (x, y) format
    return coordinates

def adaptive_non_maximal_suppression(corners, coordinates):
    N_best = 100  # Change this value to the number of best corners needed
    N_strong = len(coordinates)
    r = np.ones(N_strong) * np.inf

    for i in range(N_strong):
        for j in range(N_strong):
            if corners[coordinates[j][1], coordinates[j][0]] > corners[coordinates[i][1], coordinates[i][0]]:
                euclidean_dist = (coordinates[j][0] - coordinates[i][0])**2 + (coordinates[j][1] - coordinates[i][1])**2
                if euclidean_dist < r[i]:
                    r[i] = euclidean_dist

    sorted_indices = np.argsort(r)
    return [tuple(coordinates[idx]) for idx in sorted_indices[:N_best]]

def draw_corners(image, corners):
    corner_image = image.copy()
    for corner in corners:
        cv2.circle(corner_image, corner, 3, (0, 0, 255), -1)  # Red color for corners
    return corner_image

def describe_keypoints(image, keypoints):
    descriptors = []
    for point in keypoints:
        x, y = point[0], point[1]
        patch = image[y - 20:y + 21, x - 20:x + 21]  # Extract 41x41 patch centered around the keypoint
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)  # Apply Gaussian blur
        subsampled_patch = cv2.resize(blurred_patch, (8, 8))  # Sub-sample to 8x8
        descriptor = subsampled_patch.flatten().reshape(-1, 1)  # Reshape to obtain a 64x1 vector
        standardized_descriptor = (descriptor - np.mean(descriptor)) / np.std(descriptor)  # Standardize
        descriptors.append(standardized_descriptor)
    return descriptors

def match_descriptors(descriptors1, descriptors2):
    matches = []
    ratio = 0.7  # Adjust this ratio as needed
    for i, desc1 in enumerate(descriptors1):
        best_match_distance = float('inf')
        second_best_match_distance = float('inf')
        best_match_idx = -1

        for j, desc2 in enumerate(descriptors2):
            ssd = np.sum((desc1 - desc2) ** 2)
            
            if ssd < best_match_distance:
                second_best_match_distance = best_match_distance
                best_match_distance = ssd
                best_match_idx = j
            elif ssd < second_best_match_distance:
                second_best_match_distance = ssd

        if best_match_distance < ratio * second_best_match_distance:
            matches.append(cv2.DMatch(i, best_match_idx, 0))

    return matches

def detect_and_match_features(image_path1, image_path2):
    original_image1 = cv2.imread(image_path1)
    original_image2 = cv2.imread(image_path2)

    if original_image1 is None or original_image2 is None:
        print("Image not found. Please provide valid image paths.")
        return

    resize_img1 = resize_image(original_image1)
    resize_img2 = resize_image(original_image2)

    gray1 = convert_to_gray(resize_img1)
    gray2 = convert_to_gray(resize_img2)

    corners1 = harris_corner_detection(gray1)
    corners2 = harris_corner_detection(gray2)

    corners1 = dilate_corners(corners1)
    corners2 = dilate_corners(corners2)

    threshold1 = 0.01 * corners1.max()
    threshold2 = 0.01 * corners2.max()

    coordinates1 = find_local_maxima(corners1, threshold1)
    coordinates2 = find_local_maxima(corners2, threshold2)

    best_corners1 = adaptive_non_maximal_suppression(corners1, coordinates1)
    best_corners2 = adaptive_non_maximal_suppression(corners2, coordinates2)

    keypoints1 = [cv2.KeyPoint(int(x), int(y), 1) for (x, y) in best_corners1]
    keypoints2 = [cv2.KeyPoint(int(x), int(y), 1) for (x, y) in best_corners2]

    descriptors1 = describe_keypoints(gray1, best_corners1)
    descriptors2 = describe_keypoints(gray2, best_corners2)

    print("Keypoints 1:", len(best_corners1))
    print("Keypoints 2:", len(best_corners2))
    print("Descriptors 1:", len(descriptors1))
    print("Descriptors 2:", len(descriptors2))

    matches = match_descriptors(descriptors1, descriptors2)

    print("Matches:", len(matches))

    matched_img = cv2.drawMatches(resize_img1, keypoints1, resize_img2, keypoints2, matches, None)
    cv2.imshow('matches', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = 'data/1.jpg'
img2 = 'data/2.jpg'
detect_and_match_features(img, img2)