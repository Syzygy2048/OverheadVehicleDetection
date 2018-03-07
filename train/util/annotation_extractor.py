# This class takes the path to an annotated image as input and returns a list of coordinates based on the annotation.
# The annotated image should be in the format of a transparent image with a single pixel to mark the location of an annotated object.
# Input: image path
# Output: list of x/y coordinates (might be y/x? need to check)

from skimage import io
import numpy as np

from scipy.ndimage.interpolation import zoom


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


def _crop_images(path, list_of_centers, window_size):
    image = io.imread(path)
    training_images = np.empty(shape=[0, window_size, window_size, 3], dtype=int)
    offset = int(window_size / 2)
    print("Cropping images: ")
    for coord in list_of_centers:
        if coord[0] - offset < 0 or coord[0] + offset >= image.shape[0] - 1 or coord[1] - offset < 0 or coord[
            1] + offset >= image.shape[1] - 1:
            print("out of bounds crop, %d, %d skipped" % (coord[0], coord[1]))
        else:
            cropped = image[coord[0] - offset: coord[0] + offset, coord[1] - offset: coord[1] + offset]
            if cropped.shape[2] > 3:  # remove alpha channel if it exists.
                cropped = cropped[:, :, :3]
            training_images = np.vstack([training_images, [cropped]])
    print("Cropped %d images" % (training_images.shape[0]))
    return training_images


def extract(size, source_path, possitive_annotation_path, negative_annotated_path):
    list_of_car_centers = _extract_annotations(possitive_annotation_path)
    list_of_neg_centers = _extract_annotations(negative_annotated_path)
    car_images = _crop_images(source_path, list_of_car_centers, size)
    other_images = _crop_images(source_path, list_of_neg_centers, size)

    # return car_images, car_images
    return car_images, other_images


# print(training_images.shape)
# for img in training_images:
#     print("trying to show image")
#     io.imshow(img)
# io.show()


def resize_images(image, source_resolution, target_resolution):
    resize_factor = (target_resolution / source_resolution)
    reshaped = (image - np.mean(image)) / np.std(image)     # why? - (standard normal distribution)this causes the mean to be 0 and the variance to be 1, resulting in better recognition results - technically not resizing, but w/e, the mean and std are calculatied from the whole trainings set, and are applied to everything put into the network. training and production. #### should be backed by a source? i was told this by someone in ##machinelearning on freenode
                                            # additional infos at: https://www.tensorflow.org/tutorials/image_recognition#usage_with_the_c_api
                                            # We also need to scale the pixel values from integers that are between 0 and 255 to the floating point values that the graph operates on. We control the scaling with the input_mean and input_std flags: we first subtract input_mean from each pixel value, then divide it by input_std.
                                            # These values probably look somewhat magical, but they are just defined by the original model author based on what he/she wanted to use as input images for training. If you have a graph that you've trained yourself, you'll just need to adjust the values to match whatever you used during your training process.



    resized = zoom(reshaped, [resize_factor, resize_factor, 1.0])

    resized = resized / np.max(np.abs(resized))  # get values between + and - 1 so they can be saved with skimage, this is only necessary for preprocessing
    return resized
