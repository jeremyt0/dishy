from utilities.imageutils import ImageUtils
from utilities.logg import LOGGER

from collections import deque
import os
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import regionprops, label
from scipy import ndimage as ndi




class Detector(object):
    def __init__(self) -> None:
        self.image = None
        self.rows = 5
        self.columns = 5

        self.results = {}

        self.MAX_IMG_SIZE = 2000



    def run(self, image_path):
        '''
        Given an image, process
        '''
        self.prerequisites(image_path)

        self.start_analysis(image_path)

        return None

    def prerequisites(self, image_path):
        def make_dir_if_not_exists(dir):
            if not os.path.exists(dir):
                os.mkdir(dir)
        
        base_path, self.image_name = os.path.split(image_path)
        self.image_name_pure = os.path.splitext(self.image_name)[0]

        # Add empty list for image
        self.results[self.image_name_pure] = { "brown": [] }

        # Output directory    
        self.output_dirpath = os.path.join(base_path, 'output')
        make_dir_if_not_exists(self.output_dirpath)

        # Output Images directory
        self.output_images_dirpath = os.path.join(self.output_dirpath, self.image_name_pure)
        make_dir_if_not_exists(self.output_images_dirpath)
        
        # Output Region Images directory
        self.tmp_region_images_dirpath = os.path.join(self.output_images_dirpath, "regions")
        make_dir_if_not_exists(self.tmp_region_images_dirpath)

        return None



    def start_analysis(self, image_path):
        # 1. Get image
        LOGGER.debug("# Step 1 - Reading image #")
        self.image = ImageUtils.read_image(image_path)  

        # Convert RGB to HSV, LAB
        LOGGER.debug("# Step 2 - Converting to different colour channels #")
        self.image_HSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.image_GRAY = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_RGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        ImageUtils.try_save_img_small(self.image_HSV, os.path.join(self.output_images_dirpath, "0_HSV.png"), max_size=1000)
        ImageUtils.try_save_img_small(self.image_GRAY, os.path.join(self.output_images_dirpath, "0_GRAY.png"), max_size=1000)
        ImageUtils.try_save_img_small(self.image_RGB, os.path.join(self.output_images_dirpath, "0_RGB.png"), max_size=1000)

        # 3. Detect each leaf
        LOGGER.debug("# Step 3 - Getting leaf regions #")
        self.leaf_regions = self.detect_leaves()  

        # 4. Get values from final leaf mask
        LOGGER.debug("# Step 4 - Saving results! #")
        results = self.output_results()  

        LOGGER.debug("# Step 5 - Done! #")


    def detect_leaves(self):
        '''
        Detecting leaves
        '''
        ### Get mask from HSV ###
        h, w = self.image.shape[:2]
        img_mask = self.get_hsv_mask(self.image_HSV) 

        save_img_name = "1_HSV_mask.png"
        LOGGER.debug(f"\tSaving image: {save_img_name}")
        ImageUtils.try_save_img_small(img_mask, os.path.join(self.output_images_dirpath, save_img_name))


        ### Clean ###
        kernel = np.ones((5,5))
        img_dilate = cv2.dilate(img_mask.astype(np.uint8), kernel, iterations=2)  # Dilate
        # mask_img = remove_small_holes(img_dilate, 500)  # Remove small holes
        mask_img = remove_small_objects(img_dilate, 500)  # Remove small objects
        mask_img = ndi.binary_fill_holes(mask_img).astype(int)
        mask_img = cv2.erode(mask_img.astype(np.uint8), kernel, iterations=3)  # Erode

        save_img_name = "2_cleaned.png"
        LOGGER.debug(f"\tSaving image: {save_img_name}")
        ImageUtils.try_save_img_small(mask_img.astype(bool), os.path.join(self.output_images_dirpath, save_img_name))

        
        ### Filter leaf/not leaf ###
        leaf_regions = deque()
        labelled_mask, _ = ndi.measurements.label(mask_img)

        for region in regionprops(labelled_mask):
            if region.area < 1000 or region.eccentricity >= 0.8 or region.solidity <= 0.4:
                continue
            leaf_regions.append(region)
            continue

        ### Sort leaves by row ###
        sorted_leaf_regions = self.sort_regions_order(leaf_regions)

        ### Get each round leaf region ###
        rectangle_mask = self.image_RGB.copy()
        final_mask = np.zeros((h,w))
        blank_img = np.zeros((h,w))

        # Sort each leaf by order and create leaf object
        for n, leaf_region in enumerate(sorted_leaf_regions):
            progress = ((n+1)/len(leaf_regions))*100
            sys.stdout.write("\tSaving leaf regions: %d%%   \r" % (progress) )
            # Get mask of region
            ROI_tmp_img = ImageUtils.get_mask_of_region(leaf_region, blank_img)

            ### RGB ###
            # 1. Get only mask regions of RGB image
            ROI_rgb = self.image_RGB*ImageUtils.channels_0_to_3(ROI_tmp_img)

            # 2. Get cropped mask regions RGB
            ROI_rgb_cropped = ImageUtils.get_cropped_region_img(leaf_region, ROI_rgb)
            ImageUtils.try_save_img(ROI_rgb_cropped, os.path.join(self.tmp_region_images_dirpath, f'{n+1}.png'))

            ### HSV ### 
            # 1. Get only mask regions of HSV image
            ROI_hsv = self.image_HSV*ImageUtils.channels_0_to_3(ROI_tmp_img)
            
            # 2. Get cropped mask regions HSV
            ROI_hsv_cropped = ImageUtils.get_cropped_region_img(leaf_region, ROI_hsv)

            # 3. Get brown mask of leaf
            ROI_brown = ImageUtils.get_brown_mask(ROI_hsv_cropped)
            ImageUtils.try_save_img(ROI_brown, os.path.join(self.tmp_region_images_dirpath, f'{n+1}_brown.png'))
            ROI_brown_ratio = ImageUtils.total_brown(ROI_brown, ROI_tmp_img)
            self.results[self.image_name_pure]['brown'].append(ROI_brown_ratio)


            ### Output mask image ###
            # Add to final mask
            final_mask = np.logical_or(ROI_tmp_img, final_mask)
            # Draw box around output image
            y1, x1, y2, x2 = leaf_region.bbox
            cv2.rectangle(rectangle_mask, (x1, y1), (x2, y2), (255,0,0), 2) 

        sys.stdout.flush()
        print()
        LOGGER.info("\tFinished leaf regions.")

        ### Save images ###
        save_img_name = "3_ROI.png"
        LOGGER.debug(f"\tSaving image: {save_img_name}")
        ImageUtils.try_save_img_small(rectangle_mask, os.path.join(self.output_images_dirpath, save_img_name), max_size=self.MAX_IMG_SIZE)

        save_img_name = "4_final_mask.png"
        LOGGER.debug(f"\tSaving image: {save_img_name}")
        ImageUtils.try_save_img_small(final_mask.astype(np.uint8)*255, os.path.join(self.output_images_dirpath, save_img_name), max_size=self.MAX_IMG_SIZE)

        # Output numbered regions with highlights
        LOGGER.debug("\tOutputting numbered regions")
        final_mask_cropped = ImageUtils.get_mask_cropped_img(self.image_RGB.copy(), final_mask)
        output_img = self.output_numbered_squares(sorted_leaf_regions, final_mask_cropped)

        save_img_name = "5_final_ordered.png"
        LOGGER.debug(f"\tSaving image: {save_img_name}")
        ImageUtils.try_save_img_small(output_img, os.path.join(self.output_images_dirpath, save_img_name), max_size=self.MAX_IMG_SIZE)

        return sorted_leaf_regions


    def output_results(self):
        '''
        Loop through leaf regions
        Get specific values for each leaf and add to dataframe
        '''
        brown_ratios = self.results[self.image_name_pure]['brown']
        leaf_id = [n for n in range(1, len(brown_ratios)+1)]

        df = pd.DataFrame({'leaf_id': leaf_id,
                   'Brown Ratio': brown_ratios,
                   })

        df.to_csv(os.path.join(self.output_images_dirpath, "results.csv"))

        return True
        


    ### Methods ###

    def sort_regions_order(self, regions, order="row"):
        '''
        regions: list of regions
        Sort list of regions by rows
        '''
        if order == "row":
            x1, y1 = 0, 1
        else:  # col
            y1, x1 = 0, 1
        # 1. Sort y-centroid
        sorted_regions = deque(sorted(regions, key=lambda r: r["centroid"][x1], reverse=False))

        # 2. Sort x-centroid per row
        for num_rows in range(self.rows):
            row_regions = []
            for num_cols in range(self.columns):
                row_regions.append(sorted_regions.popleft())
            # Sort against index 1 (x value)
            sorted_row = sorted(row_regions, key=lambda r: r["centroid"][y1], reverse=False)
            sorted_regions.extend(sorted_row)
        return sorted_regions


    def output_numbered_squares(self, squares, template_img):
        template_img = template_img.copy()  # Output image for reference
        for n, sq in enumerate(squares):
            if len(sq.bbox) == 6:
                y1, x1, _, y2, x2, _ = sq.bbox
            else:
                y1, x1, y2, x2 = sq.bbox

            if len(sq.centroid) == 3:
                centroid_y, centroid_x, _ = sq.centroid
            else:
                centroid_y, centroid_x = sq.centroid
            org = (int(centroid_x), int(centroid_y))
            cv2.rectangle(template_img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(template_img, f'{n+1}', org, cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)

        return template_img


    ##

    def get_hsv_mask(self, image_HSV, lower=[60, 255, 180], higher=[0, 80, 65]):
        # Brown:
        # lower_values = np.array([6, 63, 0])
        # upper_values = np.array([23, 255, 81])
        # Boundaries are green
        # lower_hsv = np.array([60, 255, 180])
        # upper_hsv = np.array([0, 100, 65])
        lower_hsv = np.array(lower)
        upper_hsv = np.array(higher)

        mask = cv2.inRange(image_HSV, upper_hsv, lower_hsv)
        return mask

    def remove_melanin_center_circle(self, img, centroid, radius=54):
        h, w = img.shape[:2]
        a, b = centroid
        y, x = np.ogrid[-a:h-a, -b:w-b]
        mask = x**2 + y**2 <= radius**2

        img[mask] = 0
        return img



    


