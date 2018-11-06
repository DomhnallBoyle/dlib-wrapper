import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

classification_results = sys.argv[1]

print current_directory + 'semantic_segmentation_output.jpeg'
