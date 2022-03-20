from analysis.detector import Detector
from utilities.logg import LOGGER

import glob
import os


class Automator:

    def __init__(self, project_dir):
        self.project_dir = project_dir  # Images directory path


    def run(self, steps=[1]):
        '''
        Auto run from directory
        '''
        LOGGER.debug("### Running Detector ###")

        # Pre-requisites
        self.step_0()

        # Loop through functions
        for step in steps:
            self.steps[step]()

        LOGGER.success("### Finished running Detector ###")

    def make_steps(self):
        self.steps = {
            1: self.step_1,  # Run Colour images
        }

    def step_0(self):
        LOGGER.debug("## Step 0 - Pre-requisites ##")
        # Make paths
        self.colour_image_paths = self.find_all_images_in_dir(self.project_dir)
        
        # Add note to folder if no images
        if not self.colour_image_paths:
            self.write_note()
        else:
            self.remove_note()

        # Make steps
        self.make_steps()
        LOGGER.success("## Step 0 - Finished ##")


    def step_1(self):
        img_colour = "colour"
        detector = self.get_detector(img_colour)()
        # For each image
        for n, img in enumerate(self.colour_image_paths): 
            LOGGER.debug(f'Running image ({n+1}/{len(self.colour_image_paths)}): {img}')
            detector.run(img)
            
                

    def find_all_images_in_dir(self, dir):
        '''
        Returns list of image paths
        '''
        img_types = ("*.png", "*.jpg", "*.jpeg")
        image_files = []

        for ext in img_types:
            image_files.extend(glob.glob(os.path.join(dir, ext)))
        return image_files


    def get_detector(self, colour) -> object:
        detector = {
            "colour": Detector
        }
        
        return detector[colour] if colour in detector else Detector


    def write_note(self):
        filename = os.path.join(self.project_dir, "PLEASE COPY IMAGES HERE.txt")
        with open(filename, 'w') as f:
            f.write(f"Please add your colour images in this directory. Thank you.")

    def remove_note(self):
        filename = "PLEASE COPY IMAGES HERE.txt"
        text_file = glob.glob(os.path.join(self.project_dir, filename))
        if text_file:
            os.remove(text_file[0])
        

