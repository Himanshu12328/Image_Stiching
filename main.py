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
    patch_size = 41
    patches = []

    for corner in corner_mask:
        x, y, _ = corner
        x = int(x)
        y = int(y)

        half_patch_size = patch_size // 2
        if x - half_patch_size < 0 or y - half_patch_size < 0 or x + half_patch_size >= img.shape[1] or y + half_patch_size >= img.shape[0]:
            continue #skip the corner if it is too close to the borders

        patch = img[y - half_patch_size:y + half_patch_size + 1, x - half_patch_size:x + half_patch_size + 1]
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
    # sort corners by strength
    # sorted_corners = sorted(corners, key=lambda x: x[2], reverse=True)
    
    #Initialize the list of selected corners with the strongest corner
    selected_corners = [corners[0]]

    for i in range(len(corners)):
        current_corner = corners[i]
        add_corner = True

        for j in range(len(selected_corners)):
            #calculate the minimum seperation distance
            distance = np.sqrt((current_corner[0] - selected_corners[j][0]) ** 2 + (current_corner[1] - selected_corners[j][1]) ** 2)
            min_distance = 1000000


            for k in range(len(selected_corners)):
                current_distance = np.sqrt((current_corner[0] - selected_corners[k][0]) ** 2 + (current_corner[1] - selected_corners[k][1]) ** 2)
                min_distance = min(min_distance, current_distance)
                #check if the corner is too close to a previously selected corner
                # if sorted_corners[k][2] > current_corner[2]:
                #     current_distance = np.sqrt((current_corner[0] - selected_corners[k][0]) ** 2 + (current_corner[1] - selected_corners[k][1]) ** 2)
                #     min_distance = min(min_distance, current_distance)

            
            threshold = min_distance * 0.9

            if distance <= threshold:
                add_corner = False
                break
                    
        if add_corner:
            selected_corners.append(current_corner)

        #stop when the list is long enough
        if len(selected_corners) >= max_corners:
            break

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

    image_corner_mask = np.uint8(image_corner_mask)

    # find corners
    corners = []
    for y in range(image_corner_mask.shape[0]):
        for x in range(image_corner_mask.shape[1]):
            if image_corner_mask[y, x] == 255:
                corners.append((x, y, 0))

    max_corners = 100
    select_corners = adaptive_non_max_suppression(corners, max_corners)


    # draw corners
    # for corner in select_corners:
    #     cv2.circle(img, (corner[0], corner[1]), 3, (0, 0, 255), -1)

    # img_with_corners = cv2.cvtColor(image_corner_mask, cv2.COLOR_GRAY2BGR)
    # img_with_corners[image_corner_mask != 0] = [0, 0, 255]

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
    img1 = cv2.imread('dataset/0.jpg') #first image
    img2 = cv2.imread('dataset/1.jpg') #second image

    resized_img_1 = image_resizing(img1)
    resized_img_2 = image_resizing(img2)

    img_1_corner_mask = image_corners(resized_img_1)
    img_2_corner_mask = image_corners(resized_img_2)

    # print(f"Image 1 Corner Mask: {img_1_corner_mask}")
    # print(f"Image 2 Corner Mask: {img_2_corner_mask}")

    image_1_patches = extract_patch(img1, img_1_corner_mask)
    image_2_patches = extract_patch(img2, img_2_corner_mask)

    process_img1_patches = process_patch(image_1_patches)
    process_img2_patches = process_patch(image_2_patches)

    matches = match_feature_points(process_img1_patches, process_img2_patches)

    draw_matches(resized_img_1, resized_img_2, matches, img_1_corner_mask, img_2_corner_mask)

    # show_image(resized_img)
    # show_image(image_1_corner_mask)

if __name__ == '__main__':
    __main__()