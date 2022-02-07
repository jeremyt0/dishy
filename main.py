import cv2
import numpy as np
from PIL import Image
import imageio


def main():

    img = cv2.imread("leaf_example1.JPG")
    h, w = img.shape[:2]
    
    # Convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ImageUtils.saveimg_rgb(hsv_img, "img_hsv.png")


    l1 = (1870,510)
    l2 = (4350,1130)

    leaf_to_use = l2
    leaf_hsv = get_circle(hsv_img, centroid=leaf_to_use, radius=150)
    leaf_rgb = get_circle(img, centroid=leaf_to_use, radius=150)
    # img = img[h//2:w//2]
    ImageUtils.saveimg_rgb(img=leaf_rgb, filename="leaf_rgb.png")
    ImageUtils.saveimg_rgb(img=leaf_hsv, filename="leaf_hsv.png")


    # Calculate total brown from area 
    brown_in_img = total_brown(hsv_img=leaf_hsv, rgb_img=leaf_rgb)


    pass

class Leaf:
    total_leaf_pixels = 0

class ImageUtils:
    def saveimg_rgb(img, filename):
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(filename)

    def saveimg(img, filename):
        Image.fromarray(img, cv2.COLOR_BGR2RGB).save(filename)


def get_circle(img, centroid, radius=50):
    x, y = centroid
    h, w = img.shape[:2]
    circle_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(circle_mask, centroid, radius, (255,255,255), -1)
    ImageUtils.saveimg_rgb(circle_mask, "circle_mask.png")

    leaf_only = cv2.bitwise_and(img, circle_mask)
    ImageUtils.saveimg_rgb(leaf_only, "leaf_only.png")
    # TODO: Convert RGB pixels to Binary to get total white pixel mask
    # leaf_only_binary = 1.0 * (leaf_only > 0)
    # leaf_only_binary = leaf_only_binary[leaf_only_binary>0]
    # print(leaf_only_binary.shape, leaf_only_binary.dtype)
    # import sys
    # sys.exit(0)
    Leaf.total_leaf_pixels = np.count_nonzero(circle_mask)
    return leaf_only

def total_brown(hsv_img, rgb_img):
    """Calculate total brown pixels using HSV channel (Hue: 0-30)

    Args:
        img (image_region): Image region
    """

    # Get Hue channel
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    
    # Get boundaries for brown in hue channel
    hsv_lower = np.array((10, 100, 20))
    hsv_higher = np.array((20, 255, 200))
    brown = cv2.inRange(hsv_img, hsv_lower, hsv_higher)
    # Convert mask single channel to 3 channel
    brown_mask_3 = np.stack((brown,)*3, axis=-1)    
    ImageUtils.saveimg_rgb(brown, "mask_brown.png")

    # Mask out brown in original image
    overlay_img = np.bitwise_and(rgb_img, brown_mask_3)
    ImageUtils.saveimg_rgb(overlay_img, "img_brown.png")
    

    # Calculate total brown pixels
    print(f'Brown shape: {brown.shape}')
    leaf_size = Leaf.total_leaf_pixels//3
    print(leaf_size)
    total_brown = np.count_nonzero(brown)
    total_brown_ratio = total_brown/leaf_size
    print(total_brown, leaf_size)

    print(f'Ratio of brown in image: {total_brown_ratio}')
    return total_brown_ratio



if __name__=="__main__":
    main()