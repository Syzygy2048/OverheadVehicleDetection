#This class takes the path to an annotated image as input and returns a list of coordinates based on the annotation.
#The annotated image should be in the format of a transparent image with a single pixel to mark the location of an annotated object.
#Input: image path
#Output: list of x/y coordinates (might be y/x? need to check)

from skimage import io
import numpy as np

def extract_annotations(image):
    i = 1;
    pixel_list = np.empty(shape=[0, 2], dtype=int)
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if image[x][y][0] > 0:
                i = i + 1
                # print('%d - %d, %d' % (i, x, y))
                # print(pixel_list.shape, " ", np.array([x,y]).shape)
                pixel_list = np.vstack([pixel_list, np.array([x, y])])
        #     if i > 10: break
        # if i > 10: break

    return pixel_list


def crop_images(image, list_of_centers, window_size):
    training_images = np.empty(shape=[0, 50, 50, 4], dtype=int)
    offset = int(window_size/2)
    for coord in list_of_centers:
        if coord[0] - offset < 0 or coord[0] + offset >= image.shape[0] - 1 or coord[1] - offset < 0 or coord[1] + offset >= image.shape[1]-1:
            print("out of bounds crop, %d, %d skipped" % (coord[0], coord[1]))
        else:
            # print("%d/%d, %d/%d added" % (coord[0], image.shape[0], coord[1], image.shape[1]))
            # print("cropping from %d %d to %d %d" % (coord[0] - offset , coord[0] + offset, coord[1] - offset , coord[1] + offset))
            cropped = image[coord[0] - offset : coord[0] + offset, coord[1] - offset : coord[1] + offset]
            training_images = np.vstack([training_images, [cropped]])
    return training_images


def extract():
    image = io.imread("data/cowc/datasets/ground_truth_sets/Toronto_ISPRS/03559_Annotated_Cars.png")
    list_of_centers = extract_annotations(image)
    image = io.imread("data/cowc/datasets/ground_truth_sets/Toronto_ISPRS/03559.png")
    training_images = crop_images(image, list_of_centers, 50)

    return training_images
# print(training_images.shape)
# for img in training_images:
#     print("trying to show image")
#     io.imshow(img)
# io.show()


