from ast import main
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import json
import helpers

PAD = 50
THRESH = 0.8

def perform_inputs_preprocessing(image, marked_points):
    bbox = helpers.get_bbox(
        image, points=marked_points, pad=PAD, zero_pad=True)
    crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
    resize_image = helpers.fixed_resize(
        crop_image, (512, 512)).astype(np.float32)
    #  Generate extreme point heat map normalized to image values
    extreme_points = marked_points - \
        [np.min(marked_points[:, 0]), np.min(
            marked_points[:, 1])] + [PAD, PAD]
    extreme_points = (
        512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
    extreme_heatmap = helpers.make_gt(
        resize_image, extreme_points, sigma=10)
    extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)
    input_dextr = np.concatenate(
        (resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
    inputs = input_dextr.transpose((2, 0, 1))[np.newaxis, ...]
    return inputs, bbox

def get_marked_points(points=None):
    return np.array(points).astype(np.int)

def prepate_and_store_input(img_path=None, points=None):
    
    if img_path:
        img = Image.open(img_path)
        rgb_im = img.convert('RGB')
        image = np.array(rgb_im)
        marked_points = get_marked_points(points)
        prepared_inputs, bbox = perform_inputs_preprocessing(image, marked_points)

        # print("prepared_input.shape: ", prepared_inputs.shape)
        with open('./scripts/input_arr.json', 'w') as js_file:
            js_file.write(json.dumps(prepared_inputs.flatten().tolist()))
        with open('./scripts/bbox.json', 'w') as js_file:
            js_file.write(json.dumps({"bbox": [int(box_pt) for box_pt in bbox]}))
        # print(json.dumps(prepared_inputs.tolist()))
        np.save("./scripts/input_arr.npy", prepared_inputs)
        np.save("./scripts/bbox.npy", bbox)
        # print(image.shape)
        # print(marked_points)
        # print(bbox)
        # return image, prepared_inputs, bbox

def main():
    args_parser = ArgumentParser()
    args_parser.add_argument("--img_path", required=True)
    args_parser.add_argument("--points", required=True)
    args = args_parser.parse_args()

    IMG_PATH = args.img_path
    POINTS = json.loads(args.points)
    prepate_and_store_input(img_path=IMG_PATH, points=POINTS)

if __name__ == "__main__":
    main()

# ##############################################################################################
# FIRST TRY THIS TO READ FILES: https://github.com/aplbrain/npyjs
# TEST  THIS https://stackoverflow.com/questions/60281255/create-tensors-from-image-for-onnx-js
# #############################################################################################