import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--img_path', type=str, default='data/dog.jpg')
argparser.add_argument('--points', type=str, default='1 2 3 4')
vars = argparser.parse_args()
print(np.__version__)
print(f"IMG PATH = {vars.img_path}")
print(f"POINTS = {vars.points}")


#####################################
# HOW TO RUN ########################
#####################################
# python3 python_test.py
# python3 python_test.py --img_path abc.jpg --points '1 2 56'