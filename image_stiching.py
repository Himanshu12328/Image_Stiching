import cv2
import numpy as np


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


# def image_resizing(img, scale):
    # Use bilinear interpolation for better

# import cv2
# import numpy as np

# def match_feature_points(processed_img1_patches, processed_img2_patches):
#     matches = []

#     for i, patch1 in enumerate(processed_img1_patches):
#         best_match_distance = float('inf')
#         second_best_match_distance = float('inf')
#         best_match_idx = -1

#         for j, patch2 in enumerate(processed_img2_patches):
#             ssd = np.sum((patch1 - patch2) ** 2)

#             if ssd < best_match_distance:
#                 second_best_match_distance = best_match_distance
#                 best_match_distance = ssd
#                 best_match_idx = j
#             elif ssd < second_best_match_distance:
#                 second_best_match_distance = ssd

#         ratio = 0.7
#         if best_match_distance < ratio * second_best_match_distance:
#             matches.append([i, best_match_idx])
#     return matches

# def draw_matches(img1, img2, matches, img1_corners, img2_corners):
#     img1_kps = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in img1_corners]
#     img2_kps = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in img2_corners]

#     matches_to_draw = [cv2.DMatch(m[0], m[1], 1) for m in matches]
    
#     img_matches = cv2.drawMatches(img1, img1_kps, img2, img2_kps, matches_to_draw, None)
    
#     cv2.imshow('matches', img_matches)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def extract_patch(img, corner_mask):
#     '''
#     Extract patches from the image
#     '''
#     # patch_size = 41
#     patches = []

#     for corner in corner_mask:
#         x, y = corner  # Update unpacking to handle two values (x, y)
#         if x - 20 < 0 or y - 20 < 0 or x + 20 >= img.shape[1] or y + 20 >= img.shape[0]:
#             continue

#         patch = img[y - 20:y + 21, x - 20:x + 21]
#         patches.append(patch)

#     return patches

# def process_patch(patches):
#     '''
#     Process the patch
#     '''
#     processed_patches = []
#     for patch in patches:
#         blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)

#         subsampled_patch = cv2.resize(blurred_patch, (8, 8))

#         vector = subsampled_patch.flatten().reshape(-1, 1)

#         standardized_vector = (vector - np.mean(vector)) / np.std(vector)

#         processed_patches.append(standardized_vector)

#     return processed_patches

# def adaptive_non_max_suppression(corners, max_corners):
#     '''
#     Adaptive non-maximum suppression to get the best corners
#     '''

#     local_maxima = cv2.dilate(corners, None) == corners
#     coordinates = np.argwhere(local_maxima)
#     coordinates[:, [0, 1]] = coordinates[:, [1, 0]]

#     corners_strong = len(coordinates)
#     r = np.ones(corners_strong) * np.inf

#     for i in range(corners_strong):
#         x, y = coordinates[i]
#         for j in range(corners_strong):
#             a, b = coordinates[j]
#             if corners[b, a] > corners[y, x]:
#                 ED = (a - x) ** 2 + (b - y) ** 2
#                 if ED < r[i]:
#                     r[i] = ED

#     finate_indices = np.where(np.isfinite(r))[0]
#     sorted_indices = finate_indices[np.argsort(r[finate_indices])]
#     selected_corners = coordinates[sorted_indices[:max_corners]]

#     return selected_corners


# def image_corners(img):
#     '''
#     Detecting corners of the image
#     '''
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_gray = np.float32(img_gray)
#     image_corners = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    
#     # Tunning the threshold depending on the image
#     threshold = 0.01 * image_corners.max()

#     image_corner_mask = np.zeros_like(image_corners)
#     image_corner_mask[image_corners > threshold] = 255
    
#     # Get coordinates of the corners
#     corner_coords = np.argwhere(image_corner_mask == 255)
#     corner_coords[:, [0, 1]] = corner_coords[:, [1, 0]]  # Swap columns for (x, y) format
    
#     max_corners = 100
#     selected_corners = corner_coords[:max_corners]

#     return selected_corners



# def image_resizing(img):
#     '''
#     Resize 1/5 of the original image
#     '''
#     original_height, original_width = img.shape[:2]
#     new_width = original_width // 5
#     new_height = original_height // 5
#     resized_img = cv2.resize(img, (new_width, new_height))
#     return resized_img

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

    # resized_img1 = image_resizing(img1)
    # resized_img2 = image_resizing(img2)
    
    img_1_corner_mask = image_corners(img1)
    img_2_corner_mask = image_corners(img2)

    image_1_patches = extract_patch(img1, img_1_corner_mask)
    image_2_patches = extract_patch(img2, img_2_corner_mask)

    process_img1_patches = process_patch(image_1_patches)
    process_img2_patches = process_patch(image_2_patches)

    matches = match_feature_points(process_img1_patches, process_img2_patches)

    draw_matches(img1, img2, matches, img_1_corner_mask, img_2_corner_mask)

if __name__ == '__main__':
    __main__()