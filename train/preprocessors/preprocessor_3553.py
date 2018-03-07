import numpy as np
from skimage import io

from train.util import annotation_extractor
import os

np.set_printoptions(threshold=np.inf) #print full ndarray, not shortened version

res = 32  # resolution of the square that will be taken from training data.

toronto = "Toronto_ISPRS"
toronto_set1 = "03553"

region = toronto
set = toronto_set1

source_path = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc/datasets/ground_truth_sets/%s/%s.png" % (region, set)
cars_annotated_path = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc/datasets/ground_truth_sets/%s/%s_Annotated_Cars.png" % (region, set)
other_annotated_path = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc/datasets/ground_truth_sets/%s/%s_Annotated_Negatives.png" % (region, set)


def extract_annotations(path):
    image = io.imread(path)
    i = 1;
    pixel_list = np.empty(shape=[0, 2], dtype=int)
    print("Finding annotations in %s  - %dx%d pixels" % (path, image.shape[0], image.shape[1]))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y][3] > 0:
                i = i + 1

                if i % 100 == 0:
                    print("found %d annotations" % i)
                pixel_list = np.vstack([pixel_list, np.array([x, y])])

    # #10 images for debugging
    # for x in range(1000, image.shape[0]):
    #     for y in range(1000, image.shape[1]):
    #         if image[x][y][3] > 0:
    #             i = i + 1
    #             print('%d - %d, %d' % (i, x, y))
    #             print(pixel_list.shape, " ", np.array([x,y]).shape)
    #
    #             pixel_list = np.vstack([pixel_list, np.array([x, y])])
    #
    #         if i > 10: break
    #     if i > 10: break
    print("Found %d annotations in total" % (pixel_list.shape[0]))
    return pixel_list


car_centers = extract_annotations(cars_annotated_path)
neg_centers = extract_annotations(other_annotated_path)

source_image = io.imread(source_path)



def crop_image(image, center_coordinate, window_size):
    offset = int(window_size / 2)
    if center_coordinate[0] - offset < 0 or center_coordinate[0] + offset >= image.shape[0] - 1 or center_coordinate[1] - offset < 0 or center_coordinate[1] + offset >= image.shape[1] - 1:
        print("out of bounds crop, %d, %d skipped" % (center_coordinate[0], center_coordinate[1]))
        return None
    else:
        cropped = image[center_coordinate[0] - offset: center_coordinate[0] + offset, center_coordinate[1] - offset: center_coordinate[1] + offset]
        if cropped.shape[2] > 3:  # remove alpha channel if it exists.
            cropped = cropped[:, :, :3]
            # training_images = np.vstack([training_images, [cropped]])
        return cropped


save_path_car = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/%s/%s/car/px%s" % (region, set, "%d/%d.png")
save_path_neg = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/%s/%s/neg/px%s" % (region, set, "%d/%d.png")
dump_path_car = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/all/car/%s_%s_%s" % (region, set, "%d_%d.png")
dump_path_neg = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/all/neg/%s_%s_%s" % (region, set, "%d_%d.png")

print("Starting cropping, scaling, normalizing and saving car images")

if not os.path.exists("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/%s/%s/car/px%d/" % (region, set, res)):
    os.makedirs("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/%s/%s/car/px%d/" % (region, set, res))
if not os.path.exists("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/%s/%s/neg/px%d/" % (region, set, res)):
    os.makedirs("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/%s/%s/neg/px%d/" % (region, set, res))
if not os.path.exists("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/all/car/"):
    os.makedirs("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/all/car/")
if not os.path.exists("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/all/neg/"):
    os.makedirs("C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/all/neg/")

for i in range(car_centers.shape[0]):
    car = crop_image(source_image, car_centers[i], res)
    if(car is not None):
        car_scaled = annotation_extractor.resize_images(car, res, 299)
        io.imsave(save_path_car % (res, i), car_scaled)
        io.imsave(dump_path_car % (res, i), car_scaled)
        if i % 100 == 0:
            print("done with %d or %d images" % (i, car_centers.shape[0]))

print("Starting cropping, scaling, normalizing and saving negatie images")
for i in range(neg_centers.shape[0]):
    neg = crop_image(source_image, neg_centers[i], res)
    if neg is not None:
        neg_scaled = annotation_extractor.resize_images(neg, res, 299)
        io.imsave(save_path_neg % (res, i), neg_scaled)
        io.imsave(dump_path_neg % (res, i), car_scaled)
        if i % 100 == 0:
            print("done with %d or %d images" % (i, neg_centers.shape[0]))