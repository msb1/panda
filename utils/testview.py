import skimage.io
import seaborn as sns
import numpy as np
import random as rv
import matplotlib.pyplot as plt
import cv2

# Location of the training images
data_dir = 'D:\\******************************\\train_images'
mask_dir = 'D:\\******************************\\train_label_masks'
# mask color blackouts by site
overlay_mask = {0: [0, 0, 0], 1: [0, 0, 0], 2: [255, 0, 0] }
radboud_mask = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 255], 3: [255, 255, 0], 4: [255, 128, 0], 5: [255, 0, 0] }
# tile_mask = {0: [0, 0, 0], 1: [255, 255, 255], 2: [0, 0, 255], 3: [255, 255, 0], 4: [255, 128, 0], 5: [255, 0, 0] }
lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
tile = 224

def main():

    # read in Radboud image info from file
    radboud_images = []
    with open("D:\\******************************\\radboud_list.txt", 'r') as rfile:
        radboud_images = [img.strip() for img in list(rfile.readlines())]

    # read in Karolinska image info from file
    karolinska_images = []
    with open("D:\\******************************\\karolinska_list.txt", 'r') as rfile:
        karolinska_images = [img.strip() for img in list(rfile.readlines())]

    num_radboud = len(radboud_images)
    num_karolinska = len(karolinska_images)

    img_name = ""
    flag = True
    while True:
        if flag:
            idx = rv.randint(0, num_radboud)
            img_name = radboud_images[idx]
            flag = False
        else:
            idx = rv.randint(0, num_karolinska)
            img_name = karolinska_images[idx]
            flag = True

        title = "Karolinska Image: " if flag else "Radboud Image: "
        print(title, img_name)
        biopsy = skimage.io.MultiImage(data_dir + '\\' + img_name + '.tiff')
        img = biopsy[-1]
        biopsy_mask = skimage.io.MultiImage(mask_dir + '\\' + img_name.strip() + '_mask.tiff')
        mask = biopsy_mask[-1]
        print(img.shape, mask.shape)

        _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220, 255, cv2.THRESH_BINARY)
        img[thresh == 255] = 0

        ht = img.shape[0] // tile
        wt = img.shape[1] // tile
        hr = tile - img.shape[0] % tile
        wr = tile - img.shape[1] % tile
        img = cv2.copyMakeBorder(img, 0, hr, 0, wr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, hr, 0, wr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
   
        for h in range(1, ht + 1):
            img = cv2.line(img, (0, h * tile), (img.shape[1] + wr, h * tile), (255, 0, 0), 5)
        for w in range(1, wt + 1):
            img = cv2.line(img, (w * tile, 0), (w * tile, img.shape[0] + hr), (255, 0, 0), 5)

        overlay = np.zeros(img.shape, dtype=np.uint8)
        result = 0
        for col in range(wt + 1):
            jdx = col * tile
            for row in range(ht + 1):
                kdx = row * tile
                t_img = cv2.cvtColor(img[kdx: kdx + tile, jdx: jdx + tile, :], cv2.COLOR_BGR2GRAY)
                t_mask = mask[kdx: kdx + tile, jdx: jdx + tile, 0]
                # hist, bins = np.histogram(t_mask, bins=6, range=(0, 5))
                mmax = np.amax(t_mask)
                if flag:
                    result = mmax
                    if mmax > 2:
                        result = 2
                else:
                    if mmax == 1 or mmax == 2:
                        result = 1
                    if mmax > 2:
                        result = 2

                if cv2.countNonZero(t_img) > 0.15 * tile * tile:
                    overlay[kdx: kdx + tile, jdx: jdx + tile, 0] = overlay_mask[result][0]
                    overlay[kdx: kdx + tile, jdx: jdx + tile, 1] = overlay_mask[result][1]
                    overlay[kdx: kdx + tile, jdx: jdx + tile, 2] = overlay_mask[result][2]

        img_mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
        pixel = [0, 0, 0]
        for rdx, row in enumerate(mask[:, :, 0]):
            for vdx, val in enumerate(row):
                if flag:
                    if val > 2:
                        val = 2
                    pixel = overlay_mask[val]
                else:
                    if val > 5: 
                        val = 5
                    pixel = radboud_mask[val]
                img_mask[rdx, vdx, 0] = pixel[0]
                img_mask[rdx, vdx, 1] = pixel[1]
                img_mask[rdx, vdx, 2] = pixel[2]

        # print(np.histogram(img_contours, bins=51))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.imshow(img_mask, alpha=0.4)
        ax.imshow(overlay, alpha=0.3)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_title(title + img_name)
        plt.show()


if __name__ == "__main__":
    main()