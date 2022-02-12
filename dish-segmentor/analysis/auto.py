from analysis.detector import Detector

import glob
import os


class Automator:

    def __init__(self, project_dir):
        self.project_dir = project_dir  # Images directory path


    def run(self, steps=[1]):
        '''
        Auto run from directory
        '''
        print("### Running Detector ###\n")

        # Pre-requisites
        self.step_0()

        # Loop through functions
        for step in steps:
            self.steps[step]()

        print("### Finished running Detector ###\n")

    def make_steps(self):
        self.steps = {
            1: self.step_1,  # Run Colour images
        }

    def step_0(self):
        # Make paths
        self.colour_image_paths = self.find_all_images_in_dir(self.project_dir)
        # print(f'All image paths: {self.colour_image_paths}')

        # Make steps
        self.make_steps()


    def step_1(self):
        img_colour = "colour"
        detector = self.get_detector(img_colour)()
        # For each image
        for n, img in enumerate(self.colour_image_paths): 
            print(f'Running image ({n+1}/{len(self.colour_image_paths)}): {img}')
            detector.run(img)
            
                

    def find_all_images_in_dir(self, dir):
        '''
        Returns list of image paths
        '''
        image_files = glob.glob(os.path.join(dir, "*.png"))
        return image_files


    def get_detector(self, colour) -> object:
        detector = {
            "colour": Detector
        }
        
        return detector[colour] if colour in detector else Detector


