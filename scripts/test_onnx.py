import numpy as np
import onnxruntime as ort
from PIL import Image
from dataloaders import helpers as helpers
from skimage.transform import resize
from matplotlib import pyplot as plt

# import onnx
# onnx_model = onnx.load("model.onnx")
# onnx.checker.check_model(onnx_model)

IMG_PATH = 'ims/dog-cat.jpg'
PAD = 50
THRESH = 0.8
POINTS_TO_CHOOSE = 'dog' # 'dog' or 'cat'

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


def get_marked_points():
    
    if POINTS_TO_CHOOSE == 'dog':
        points_list = [[147, 23], [52, 362], [0, 273], [328, 154]]
    elif POINTS_TO_CHOOSE == 'cat':
        points_list = [[325, 118], [271, 301], [363, 372], [498, 239]]
    else:
        raise ValueError("Invalid POINTS_TO_CHOOSE value")

    return np.array(points_list).astype(np.int)


def get_input(img_path=None):
    
    if img_path:
        image = np.array(Image.open(img_path))
        marked_points = get_marked_points()
        prepared_inputs, bbox = perform_inputs_preprocessing(image, marked_points)

        # print("prepared_input.shape: ", prepared_inputs.shape)
        # print(marked_points)
        return image, prepared_inputs, bbox

    return np.random.random((1, 4, 512, 512)).astype('float32')


def perform_post_processing(image, results, bbox):
    # outputs = upsample(outputs, size=(512, 512),
    #                        mode='bilinear', align_corners=True)

    print(results[0][0].shape)
    outputs = resize(results[0][0], (512, 512))
    outputs = np.expand_dims(outputs, axis=[0, 1])
    print(outputs.shape)


    pred = np.transpose(outputs[0, ...], (1, 2, 0))
    pred = 1 / (1 + np.exp(-pred))
    pred = np.squeeze(pred)
    result = helpers.crop2fullmask(
            pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=PAD) > THRESH
    return result

def print_inputs_outputs(session):
    all_ips = session.get_inputs()
    all_outs = session.get_outputs()
    # input dimensions (important for debugging)
    print()
    print("Inputs:")
    for ip in all_ips:
        print(f"Name: {ip.name} -- Shape: {ip.shape}")
    print()
    print("Outputs:")
    for op in all_outs:
        print(f"Name: {op.name} -- Shape: {op.shape}")
    print()


def test_dextr_model():
    image, model_input, ip_bbox = get_input(img_path=IMG_PATH)
    plt.axis('off')
    plt.imshow(image)
    sess = ort.InferenceSession('model_dextr02.onnx',
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    print_inputs_outputs(session=sess)
    results = sess.run(["outputs"], {"inputs": model_input})[0]
    prepared_outputs = perform_post_processing(image, results, ip_bbox)
    print(prepared_outputs.shape)
    plt.imshow(helpers.overlay_masks(image / 255, prepared_outputs))
    plt.show()


if __name__ == "__main__":
    test_dextr_model()
