import cv2
import numpy as np

def image_corners(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)
    image_corners = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    #tunning the threshold depending on image
    threshold = 0.01 * image_corners.max()

    image_corner_mask = np.zeros_like(image_corners)
    image_corner_mask[image_corners > threshold] = 255

    image_corner_mask = np.uint8(image_corner_mask)

    contours, _ = cv2.findContours(image_corner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # img_with_corners = cv2.cvtColor(image_corner_mask, cv2.COLOR_GRAY2BGR)
    # img_with_corners[image_corner_mask != 0] = [0, 0, 255]

    return img


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
    img = cv2.imread('dataset/1.jpg')

    resized_img = image_resizing(img)

    image_corner_mask = image_corners(resized_img)
    
    # show_image(resized_img)
    show_image(image_corner_mask)

if __name__ == '__main__':
    __main__()