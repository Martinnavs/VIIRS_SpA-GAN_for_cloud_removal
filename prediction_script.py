import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def predict(args):

    # img = cv.imread('Sample VIIRS/jan721.tiff')
    img = cv.imread(args.input_image)

    M = 512
    N = M

    if os.path.exists("prediction_inputs"):
        os.system("rm -r prediction_inputs/")
    os.mkdir("prediction_inputs")
    os.mkdir("prediction_inputs/cloudy_image")

    i = 0
    arr = []

    # slicing images into 512 x 512, with iteration moving at 512/4
    for x in range(0,img.shape[0] // M + 2):
        for y in range(0,img.shape[1] // N + 2):
            x1 = x * M if x * M < img.shape[0] and x * M + M < img.shape[0] else img.shape[0]- M
            y1 = y * N if y * N < img.shape[1] and y * N + N < img.shape[1] else img.shape[1] - N
            # getting the image slice
            sampimg = img[x1:x1+M,y1:y1+N]
            # recording image name, and location where it will be placed in the new image
            arr.append([str(i)+".tiff", x1, x1+M, y1, y1+N])
            # writing it into the prediction directory
            cv.imwrite(os.path.join("prediction_inputs/cloudy_image", arr[i][0]), sampimg)
            i += 1

    # Python prediction script. Use based on whether you have a GPU or not
    if not args.gpu: # w/o GPU
        os.system("python predict_general.py --config pretrained_models/Denoised/config_cpu.yml --pretrained pretrained_models/Denoised/gen_model_epoch_60.pth ")
    else: # w/ GPU
        os.system("python predict_general.py --config pretrained_models/Denoised/config_gpu.yml --pretrained pretrained_models/Denoised/gen_model_epoch_60.pth  --cuda")

    # base arrays for reconstructing image
    new_img = np.zeros(img.shape)
    attentionmap = np.zeros(img.shape)

    # getting the 2nd part of the three images
    # recall: 1st image - cloudy, 2nd image - cleaned, 3rd image - attention map
    for item in arr:
        val = cv.imread(os.path.join("prediction_inputs/epoch_0001", item[0]))
        h, w, channels = val.shape
        trio = w//3
        cleaned = val[:, 1*trio:2*trio] 
        # adding the image based on its stored position
        img[item[1]:item[2], item[3]:item[4]] = cleaned
        attentionmap[item[1]:item[2], item[3]:item[4]] = val[:, 2*trio:]

    cv.imwrite(args.output_image, img)
    
    print("Prediction complete.")
    
    if os.path.exists("prediction_inputs"):
        os.system("rm -r prediction_inputs/")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=False)
    parser.add_argument('--output_image', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    predict(args)