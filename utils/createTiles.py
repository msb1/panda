import cv2
import imutils
import skimage.io
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Location of the training images
data_dir = 'D:\\******************************\\train_images'
mask_dir = 'D:\\******************************\\train_label_masks'
# mask color blackouts by site
karoliska_mask = {0: [0, 0, 0], 1: [0, 0, 255], 2: [255, 0, 0]}
radboud_mask = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 255], 3: [255, 255, 0], 4: [255, 128, 0], 5: [255, 0, 0]}
tile_mask = {0: [0, 0, 0], 1: [255, 255, 255], 2: [0, 0, 255], 3: [255, 255, 0], 4: [255, 128, 0], 5: [255, 0, 0]}
lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
tile = 224
tile_area = tile * tile

def main():

    panda_tiles = open("D:\\******************************\\panda_tiles.txt", 'w')

    # read in Radboud image info from file
    radboud_images = []
    with open("D:\\******************************\\radboud_list.txt", 'r') as rfile:
        radboud_images = [img.strip() for img in list(rfile.readlines())]

    # read in Radboud images and write image tile info to file
    for idx, img_name in enumerate(radboud_images):
        # print("Radboud image: ", img_name)
        biopsy = skimage.io.MultiImage(data_dir + '\\' + img_name + '.tiff')
        img = biopsy[-1]
        biopsy_mask = skimage.io.MultiImage(mask_dir + '\\' + img_name.strip() + '_mask.tiff')
        mask = biopsy_mask[-1]
        # print(img.shape, mask.shape)

        _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220, 255, cv2.THRESH_BINARY)
        img[thresh == 255] = 0

        ht = img.shape[0] // tile
        wt = img.shape[1] // tile
        hr = tile - img.shape[0] % tile
        wr = tile - img.shape[1] % tile
        img = cv2.copyMakeBorder(img, 0, hr, 0, wr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, hr, 0, wr, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # print(ht, wt, hr, wr)
        if idx % 50 == 0:
            print("Iteration: {} / {}".format(idx, len(radboud_images)))

        # overlay = np.zeros(img.shape, dtype=np.uint8)
        for col in range(wt + 1):
            jdx = col * tile
            for row in range(ht + 1):
                kdx = row * tile
                t_img = cv2.cvtColor(img[kdx: kdx + tile, jdx: jdx + tile, :], cv2.COLOR_BGR2GRAY)
                t_mask = mask[kdx: kdx + tile, jdx: jdx + tile, 0]
                mmax = np.amax(t_mask)
                result = 0
                if mmax == 1 or mmax == 2:
                    result = 1
                if mmax > 2:
                    result = 2
                if cv2.countNonZero(t_img) > 0.15 * tile * tile:
                    # print(cv2.countNonZero(tile_img))
                    img[kdx: kdx + tile, jdx: jdx + tile, 2] = 255
                    t_info = "{},{},{},{}\n".format(img_name, kdx, jdx, result)
                    panda_tiles.write(t_info)  

    # read in Karolinska image info from file
    karolinska_images = []
    with open("D:\\******************************\\karolinska_list.txt", 'r') as rfile:
        karolinska_images = [img.strip() for img in list(rfile.readlines())]

    # read in Karolinska images and write image tile info to file
    for idx, img_name in enumerate(karolinska_images):
        # print("Karolinska image: ", img_name)
        biopsy = skimage.io.MultiImage(data_dir + '\\' + img_name + '.tiff')
        img = biopsy[-1]
        biopsy_mask = skimage.io.MultiImage(mask_dir + '\\' + img_name.strip() + '_mask.tiff')
        mask = biopsy_mask[-1]
        # print(img.shape, mask.shape)

        _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220, 255, cv2.THRESH_BINARY)
        img[thresh == 255] = 0

        ht = img.shape[0] // tile
        wt = img.shape[1] // tile
        hr = tile - img.shape[0] % tile
        wr = tile - img.shape[1] % tile
        img = cv2.copyMakeBorder(img, 0, hr, 0, wr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, hr, 0, wr, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # print(ht, wt, hr, wr)
        if idx % 50 == 0:
            print("Iteration: {} / {}".format(idx, len(karolinska_images)))

        for col in range(wt + 1):
            jdx = col * tile
            for row in range(ht + 1):
                kdx = row * tile
                t_img = cv2.cvtColor(img[kdx: kdx + tile, jdx: jdx + tile, :], cv2.COLOR_BGR2GRAY)
                t_mask = mask[kdx: kdx + tile, jdx: jdx + tile, 0]
                mmax = np.amax(t_mask)
                result = mmax
                if mmax > 2:
                    result = 2
                if cv2.countNonZero(t_img) > 0.15 * tile * tile:
                    # print(cv2.countNonZero(tile_img))
                    img[kdx: kdx + tile, jdx: jdx + tile, 2] = 255
                    t_info = "{},{},{},{}\n".format(img_name, kdx, jdx, result)
                    panda_tiles.write(t_info)

    panda_tiles.close()


if __name__ == "__main__":
    main()