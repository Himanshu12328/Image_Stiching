import cv2
import numpy as np

def describe_feature(img, keypoints):
    descriptors = []

    for kp in keypoints:
        x, y = kp.pt[0], kp.pt[1]

        # Check if patch is within image boundaries
        if (y - 20 < 0) or (y + 21 > img.shape[0]) or (x - 20 < 0) or (x + 21 > img.shape[1]):
            continue

        patch = img[max(0, int(y - 20)):min(int(y + 21), img.shape[0]), max(0, int(x - 20)):min(int(x + 21), img.shape[1])]

        if patch.shape != (41, 41):  # Ensure the patch is 41x41
            continue

        blurred_patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=2)
        subsampled_patch = cv2.resize(blurred_patch, (8, 8))
        descriptor = subsampled_patch.flatten()

        # Standardize the vector
        descriptor = (descriptor - np.mean(descriptor)) / np.std(descriptor)

        descriptors.append(descriptor)

    return np.array(descriptors)
            
    
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
    

def ANMS(corners, max_pts):
    # Finding the x, y coordinates from corners
    coords = np.array([[corner[0][0], corner[0][1]] for corner in corners])

    # Calculate Euclidean distances between points
    distances = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            distances[i][j] = np.linalg.norm(coords[i] - coords[j])

    # Find the furthest and closest neighbors for each point
    furthest = np.max(distances, axis=1)
    closest = np.min(distances + np.eye(len(coords)) * np.max(distances), axis=1)

    # Compute the suppression criteria
    suppression = furthest / closest

    # Select the points with highest suppression criteria
    indices = np.argsort(suppression)[::-1][:max_pts]
    selected_corners = [corners[i] for i in indices]

    return selected_corners

def corners(img, choice, Nbest, max_pts):
    '''
    Corner Detector to find corners of an image, Choice 1 = Shi-Tomasi Choice 2 = Harris
    '''

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if choice == 1:
        corners_img = cv2.goodFeaturesToTrack(gray_img, 1000, 0.05, 10)
        
        selected_corners = ANMS(corners_img, max_pts)

    elif choice == 2:
        gray_img = np.float32(gray_img)
        corners_img = cv2.cornerHarris(gray_img, 3, 3, 0.04)
        img[corners_img > 0.001 * corners_img.max()] = [0, 255, 0]

        selected_corners = ANMS(corners_img, max_pts)

    else:
        print("Wrong choice entered for corner detection method")
        return None

    return selected_corners
    

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
