import numpy as np

img1 = np.load(f"final_project/image_processing/data_blurring/outputimage_own 3Memristors [12]_0.npy")
img2 = np.load(f"final_project/image_processing/data_blurring/outputimage_own Aprox [11]_0.npy")



print(np.sum(img1 - img2))

