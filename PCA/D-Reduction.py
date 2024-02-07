import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

img = cv2.imread("Your_img_file.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_double = gray_img.astype(np.float64)
mean_column = np.mean(gray_double, axis=0)
img_mean_subtracted = gray_double - mean_column
cov_matrix = np.cov(img_mean_subtracted.T)
value, vector = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(value)[::-1]
sorted_vector = vector[:, sorted_indices]
num_components = [10, 20, 30, 40, 50, 60, 91]
Output_images = []
for i in num_components:
    selected_components = sorted_vector[:, :i]
    projected_data = np.dot(img_mean_subtracted, selected_components)
    reconstructed_image = np.dot(projected_data, selected_components.T) + mean_column
    Output_images.append(reconstructed_image.astype(np.uint8))

plt.figure(figsize=(20, 10))  # Increase the size of the figure
plt.suptitle("Dimensionality Reduction using PCA")
for i, reconstructed_image in enumerate(Output_images):
    plt.subplot(1, len(num_components), i+1)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"{num_components[i]} components")
    plt.axis('off')

plt.show()

pca = PCA(n_components=91)
pca.fit(img_mean_subtracted)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio using 91 components:", np.sum(explained_variance_ratio))





