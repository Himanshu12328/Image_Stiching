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
    img1_kps = [cv2.KeyPoint(x[0], x[1], 1) for x in img1_corners]
    img2_kps = [cv2.KeyPoint(x[0], x[1], 1) for x in img2_corners]

    matches_to_draw = [cv2.DMatch(m[0], m[1], 1) for m in matches]
    
    img_matches = cv2.drawMatches(img1, img1_kps, img2, img2_kps, matches_to_draw, None)
    
    cv2.imshow('matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_patch(img, corner_mask):
    '''
    Extract patches from the image
    '''
    # patch_size = 41
    patches = []

    for corner in corner_mask:
        x, y, _ = corner
        # x = int(x)
        # y = int(y)
        if x - 20 < 0 or y - 20 < 0 or x + 20 >= img.shape[1] or y + 20 >= img.shape[0]:
            continue

        # half_patch_size = patch_size // 2
        # if x - half_patch_size < 0 or y - half_patch_size < 0 or x + half_patch_size >= img.shape[1] or y + half_patch_size >= img.shape[0]:
        #     continue #skip the corner if it is too close to the borders

        patch = img[y - 20:y + 21, x - 20:x + 21]
        patches.append(patch)

    return patches

def process_patch(patches):
    '''
    Process the patch
    '''
    processed_patches = []
    for patch in patches:
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)

        subsampled_patch = cv2.resize(blurred_patch, (8, 8))

        vector = subsampled_patch.flatten().reshape(-1, 1)

        standardized_vector = (vector - np.mean(vector)) / np.std(vector)

        processed_patches.append(standardized_vector)

    return processed_patches

def adaptive_non_max_suppression(corners, max_corners):
    '''
    Adaptive non-maximum suppression to get the best corners
    '''

    local_maxima = cv2.dilate(corners, None) == corners
    coordinates = np.argwhere(local_maxima)
    coordinates[:, [0, 1]] = coordinates[:, [1, 0]]

    corners_strong = len(coordinates)
    r = np.ones(corners_strong) * np.inf

    for i in range(corners_strong):
        x, y = coordinates[i]
        for j in range(corners_strong):
            a, b = coordinates[j]
            if corners[b, a] > corners[y, x]:
                ED = (a - x) ** 2 + (b - y) ** 2
                if ED < r[i]:
                    r[i] = ED

    finate_indices = np.where(np.isfinite(r))[0]
    sorted_indices = finate_indices[np.argsort(r[finate_indices])]
    selected_corners = [tuple(coordinates[idx] for idx in sorted_indices[:max_corners])]

    return selected_corners


def image_corners(img):
    '''
    detecting corners of the image
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)
    image_corners = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    #tunning the threshold depending on image
    threshold = 0.01 * image_corners.max()

    image_corner_mask = np.zeros_like(image_corners)
    image_corner_mask[image_corners > threshold] = 255
    
    Cimg = image_corners

    max_corners = 100
    select_corners = adaptive_non_max_suppression(Cimg, max_corners)

    return select_corners


def image_resizing(img):
    '''
    Resize 1/5 of the original image
    '''
    original_height, original_width = img.shape[:2]
    new_width = original_width // 5
    new_height = original_height // 5
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def show_image(img):
    '''
    Show the image
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def __main__():
    img1 = cv2.imread('Dataset/0.jpg') #first image
    img2 = cv2.imread('Dataset/1.jpg') #second image

    resized_img1 = image_resizing(img1)
    resized_img2 = image_resizing(img2)
    
    img_1_corner_mask = image_corners(resized_img1)
    img_2_corner_mask = image_corners(resized_img2)

    image_1_patches = extract_patch(resized_img1, img_1_corner_mask)
    image_2_patches = extract_patch(resized_img2, img_2_corner_mask)

    process_img1_patches = process_patch(image_1_patches)
    process_img2_patches = process_patch(image_2_patches)

    matches = match_feature_points(process_img1_patches, process_img2_patches)

    draw_matches(resized_img1, resized_img2, matches, img_1_corner_mask, img_2_corner_mask)

if __name__ == '__main__':
    __main__()