import numpy as np
from PIL import Image
import time
import os


def is_valid_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        return False

    # Check if the path is a regular file
    if not os.path.isfile(file_path):
        return False

    # Add additional checks if needed
    # For example, check for file extension, file size, etc.

    # If all checks pass, the file is considered valid
    return True


def input_data():
    # Input the filename
    while True:
        filename = input('Enter name of an image: ')
        typefile = filename[len(filename)-3:]
        if ((typefile == 'jpg' or typefile == 'png') and (is_valid_file(filename))):
            break

    while True:
        # Enter the number of clusters
        k_clusters = int(input('Enter the number of clusters (max 1000): '))

        # Enter the number of iterations
        iteration = int(
            input('Enter the max number of iterations (max 1000): '))
        if ((iteration > 0 and iteration <= 1000)
                and (k_clusters > 0 and k_clusters <= 1000)):
            break

    # Choose type of centroids
    type_centroids = input(
        'Enter the type of centroid (random or in_pixels): ')
    if (type_centroids != 'random'):
        type_centroids = 'in_pixels'

    return filename, k_clusters, iteration, type_centroids


def output_data(filename, k_clusters, outputImg):
    # Save image
    print("\nCompression done. \n")
    while True:
        outputType = input(
            "Enter the type of image output: (jpg/png/pdf): ")
        if (outputType == 'jpg' or outputType == 'png' or outputType == 'pdf'):
            break

    # Export image to file
    outputImgName = filename[:len(filename)-4]+"_" + type_centroids+'_k' + \
        str(k_clusters) + '.' + outputType
    Image.fromarray(outputImg.astype(np.uint8)).save(outputImgName)


def init_centroid(flat_img, k_cluster, type_centroid):
    # type centroid
    # 1: random
    # 2: in pixel

    centroids = np.zeros((k_cluster, 3))

    if type_centroid == 'random':
        for i in range(k_cluster):
            rand_color = np.zeros(3)
            # in case color already exist
            while(rand_color in centroids):
                for j in range(len(flat_img[0])):
                    rand_color[j] = np.random.randint(0, 256)
            centroids[i] = rand_color
        return centroids
    elif type_centroid == 'in_pixels':
        for i in range(k_cluster):
            # in case color already exist
            choice_color = np.random.choice(flat_img.shape[0])
            while(choice_color in centroids):
                choice_color = np.random.choice(flat_img.shape[0])
            centroids[i] = flat_img[choice_color]
        return centroids
    else:
        return None


def assign_labels(flat_img, centroids):
    # Tính toán khoảng cách giữa flat_img và centroids
    distances = np.linalg.norm(flat_img[:, None] - centroids, axis=2)

    # Gán nhãn cho mỗi pixel dựa trên khoảng cách nhỏ nhất
    labels = np.argmin(distances, axis=1)

    return labels


def update_centroids(flat_img, labels_arr, centroids_info):
    new_centroids = np.zeros(centroids_info)  # create matrix zeros

    for i in range(centroids_info[0]):
        # update pixel in every cluster
        pixel_cluster = flat_img[labels_arr == i]
        if len(pixel_cluster) > 0:
            # take mean value of col of cluster i
            new_centroids[i] = np.mean(pixel_cluster, axis=0)
    return new_centroids


def kmeans(flat_img, k_clusters, max_iteration, type_centroids):
    # initialize centroids
    centroids = init_centroid(flat_img, k_clusters, type_centroids)

    # initalize 0 for all elements in flat image
    labels = np.full(flat_img.shape[0], 0)

    for _ in range(max_iteration):
        old_centroids = centroids
        labels = assign_labels(flat_img, centroids)
        centroids = update_centroids(flat_img, labels, centroids.shape)
        if np.allclose(old_centroids, centroids, rtol=10e-5, equal_nan=False):
            break
    return centroids, labels


def change_2d_to_1d(filename):
    # Open image
    image = Image.open(filename)
    # Convert to numpy array (3D matrix)
    image = np.array(image)
    # Preprocessing - Flatten image to a 1D array
    rows, cols, channels = image.shape
    flatImage = image.reshape(rows * cols, channels)
    return flatImage, image


def change_1d_to_2d(centroids, labels, image):
    result = centroids[labels].astype(np.uint8)
    return result.reshape(image.shape)


if __name__ == '__main__':
    filename, k_clusters, iteration, type_centroids = input_data()

    start_time = time.time()
    # Preprocessed
    flatImg, image = change_2d_to_1d(filename)

    # Apply K-means clustering algorithm
    centroids, labels = kmeans(
        flatImg, k_clusters, iteration, type_centroids)

    # Reshape to the original image
    outputImg = change_1d_to_2d(centroids, labels, image)
    end_time = time.time()

    output_data(filename, k_clusters, outputImg)

    execution_time = end_time - start_time
    print("Execution time:", round(execution_time, 2), "s")
