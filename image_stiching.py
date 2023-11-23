import cv2
import numpy as np

def compute_homography(src_pts, dst_pts):
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)[0]

def calculate_ssd(point1, point2, homography):
    transformed_point = np.dot(homography, np.array([point1[0], point1[1], 1]))
    transformed_point = transformed_point / transformed_point[2]
    return np.sum((point2 - transformed_point[:2])**2)


def match_feature_points(processed_img1_patches, processed_img2_patches):
    matches = []

    for i, patch1 in enumerate(processed_img1_patches):
        best_match_distance = float('inf')
        second_best_match_distance = float('inf')
        best_match_idx = -1

        for j, patch2 in enumerate(processed_img2_patches):
            ssd = np.sum((patch1 - patch2) ** 2)

            if ssd < best_match_distance:
                second_best_match_distance = best_match_distance
                best_match_distance = ssd
                best_match_idx = j
            elif ssd < second_best_match_distance:
                second_best_match_distance = ssd

        ratio = 0.7
        if best_match_distance < ratio * second_best_match_distance:
            matches.append([i, best_match_idx])

    return matches


def draw_matches(img1, img2, matches, img1_corners, img2_corners):
    img1_kps = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in img1_corners]
    img2_kps = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in img2_corners]

    matches_to_draw = [cv2.DMatch(m[0], m[1], 1) for m in matches]

    img_matches = cv2.drawMatches(img1, img1_kps, img2, img2_kps, matches_to_draw, None)

    cv2.imshow('matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_patch(img, corner_mask):
    patches = []

    for corner in corner_mask:
        x, y = corner  # Update unpacking to handle two values (x, y)

        # Check if patch falls within image boundaries
        if x - 20 < 0 or y - 20 < 0 or x + 20 >= img.shape[1] or y + 20 >= img.shape[0]:
            continue

        patch = img[int(y - 20):int(y + 21), int(x - 20):int(x + 21)]
        patches.append(patch)

    return patches


def process_patch(patches):
    processed_patches = []

    for patch in patches:
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)

        subsampled_patch = cv2.resize(blurred_patch, (8, 8))

        vector = subsampled_patch.flatten().reshape(-1, 1)

        standardized_vector = (vector - np.mean(vector)) / np.std(vector)

        processed_patches.append(standardized_vector)

    return processed_patches


def adaptive_non_max_suppression(corners, max_corners):
    # Use k-d tree to improve efficiency
    tree = cv2.flann_based_matcher({'algorithm': 'kdtree', 'trees': 5})

    # Calculate distances between corners
    matches = tree.knnMatch(corners, corners, k=2)

    # Apply non-maximum suppression
    selected_corners = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance:
            selected_corners.append(corners[m[0].queryIdx])

    # Select top `max_corners` corners
    selected_corners = selected_corners[:max_corners]

    return selected_corners


def image_corners(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)

    # Use Shi-Tomasi corner detection for faster performance
    image_corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=100, qualityLevel=0.01, blockSize=2, minDistance=10)

    return image_corners.reshape(-1, 2)

def show_image(img):
    '''
    Show the image
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def __main__():
    img1 = cv2.imread('data/1.jpg') #first image
    img2 = cv2.imread('data/2.jpg') #second image
    
    img_1_corner_mask = image_corners(img1)
    img_2_corner_mask = image_corners(img2)

    image_1_patches = extract_patch(img1, img_1_corner_mask)
    image_2_patches = extract_patch(img2, img_2_corner_mask)

    process_img1_patches = process_patch(image_1_patches)
    process_img2_patches = process_patch(image_2_patches)

    matches = match_feature_points(process_img1_patches, process_img2_patches)

    threshold = 100
    Nmax = 1000
    percentage_inliers = 0.9

    inliers = []
    best_homography = []
    max_inliers = 0

    for _ in range(Nmax):
        random_matches = np.random.choice(len(matches), 4, replace=False)
        src_pts = np.array([img_1_corner_mask[matches[random_matches[i]][0]] for i in range(4)])
        dst_pts = np.array([img_2_corner_mask[matches[random_matches[i]][1]] for i in range(4)])

        H = compute_homography(src_pts, dst_pts)

        current_inliers = []

        for idx, match in enumerate(matches):
            src_point = img_1_corner_mask[match[0]]
            dst_point = img_2_corner_mask[match[1]]

            ssd = calculate_ssd(src_point, dst_point, H)

            if ssd < threshold:
                current_inliers.append(match)

        if len(current_inliers) > max_inliers:
            max_inliers = len(current_inliers)
            inliers = current_inliers
            best_homography = H

        if len(inliers) > len(matches) * percentage_inliers:
            break

    inliers_src_pts = np.array([img_1_corner_mask[m[0]] for m in inliers])
    inliers_dst_pts = np.array([img_2_corner_mask[m[1]] for m in inliers])

    final_homography = compute_homography(inliers_src_pts, inliers_dst_pts)
    
    draw_matches(img1, img2, inliers, img_1_corner_mask, img_2_corner_mask)

if __name__ == '__main__':
    __main__()