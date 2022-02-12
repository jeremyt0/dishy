
import imageio
from PIL import Image
import cv2
import os
import numpy as np

class ImageUtils:

    @staticmethod    
    def read_image(image_path):
        return cv2.imread(image_path)

    @staticmethod
    def channels_0_to_3(img):
        # Return image with 3 channels instead of 0, binary to RGB
        return np.dstack((img, img, img)).astype(np.uint8)*255 if len(img.shape) == 2 else img



    @staticmethod
    def get_brown_mask(hsv_img):
        # Get boundaries for brown in hue channel
        hsv_lower = np.array((0, 100, 20))
        hsv_higher = np.array((25, 255, 200))
        brown = cv2.inRange(hsv_img, hsv_lower, hsv_higher)
        return brown

    @staticmethod
    def total_brown(brown_mask, leaf_mask):
        """Calculate total brown pixels using HSV channel (Hue: 0-30)

        Args:
            brown_mask (mask): Brown region of image
            leaf_mask (mask): Leaf mask of image
        """
        # Calculate total brown pixels
        leaf_size = np.count_nonzero(leaf_mask)
        total_brown = np.count_nonzero(brown_mask)
        total_brown_ratio = total_brown/leaf_size
        return total_brown_ratio

    @staticmethod
    def get_mask_of_region(region, blank_img):
        ROI_tmp_img = blank_img.copy()
        ROI_coord = region.coords.astype(int)
        ROI_tmp_img[ROI_coord[:, 0], ROI_coord[:, 1]] = 255
        return ROI_tmp_img

    @staticmethod
    def get_cropped_region_img(region, img):
        y1, x1, y2, x2 = region.bbox
        return img[y1:y2, x1:x2]


    @staticmethod
    def remove_all_black_pixels(img: np.array) -> np.array:
        # 1. Get mask of any black pixels
        mask = np.all(img == [0, 0, 0], axis=-1)
        # 2. Convert single bool value to rgb (3 channel)
        mask = np.dstack((mask, mask, mask))
        # 3. Flatten
        mask_flat = mask.flatten()
        # 4. Delete
        pixels_flat = np.delete(img, mask_flat)
        # 5. Reshape
        all_non_black_pixels = np.reshape(pixels_flat, (len(pixels_flat)//3, 3))
        return all_non_black_pixels

    @staticmethod
    def binary_to_rgb(img):
        # Return image with 3 channels instead of 0
        return np.dstack((img, img, img)).astype(np.uint8)*255 if len(img.shape) == 2 else img
    
    @staticmethod
    def binary_to_rgb_bool(img):
        # Return image with 3 channels instead of 0
        return np.dstack((img, img, img)) if len(img.shape) == 2 else img



    @staticmethod
    def max_resize_img(img, max_length=32000):
        if img.dtype == "bool":
            img = img.astype(np.uint8)*255
        # If larger than 32k then resize
        width, height = img.shape[:2]
        if width > height:
            if width > max_length:
                delta = width/max_length
                # print("Width:",(max_length, int(height/delta)))
                return cv2.resize(img, (int(height/delta), max_length), interpolation=cv2.INTER_AREA)
        elif height > width:
            if height > max_length:
                delta = height/max_length
                # print("Height:",(int(width/delta), max_length))
                return cv2.resize(img, (max_length, int(width/delta)), interpolation=cv2.INTER_AREA)
        elif height == width:
            print("Square shaped.")
            pass
            # Check if greater than 32000
        # Fine
        return img

    @staticmethod
    def try_save_img(image, image_path):
        try:
            Image.fromarray(image).save(image_path)
        except Exception as e:
            print(e)
            try:
                imageio.imsave(image_path, image)
            except Exception as e2:
                print(e2)

    @staticmethod
    def try_save_img_small(image, image_path, max_size=4000):
        new_image = ImageUtils.max_resize_img(image, max_size)
        ImageUtils.try_save_img(new_image, image_path)

    @staticmethod
    def try_save_img_if_not_exists(image, image_path):
        if os.path.isfile(image_path):
            return
        ImageUtils.try_save_img(image, image_path)


    @staticmethod
    def get_mask_cropped_img(to_crop_img, grid_mask):
        mask = grid_mask.copy().astype("uint8")
        img = to_crop_img.copy()

        height, width = img.shape[:2]
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

        cropped_img = cv2.bitwise_and(img, img, mask=mask)
        return cropped_img

