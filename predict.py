import torch
import cv2
import numpy as np
import random as rv
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import skimage.io

from utils.genData import ImageDataSet
from models.resnet50 import ResNet50

# Location of the training images
data_dir = 'D:\\******************************\\train_images'
mask_dir = 'D:\\******************************\\\train_label_masks'
checkpoint_file = 'C:\\******************************\\checkpoint.pt'
# mask color blackouts by site
overlay_mask = {0: [0, 0, 0], 1: [0, 0, 0], 2: [255, 0, 0]}
radboud_mask = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 255], 3: [255, 255, 0], 4: [255, 128, 0], 5: [255, 0, 0]}
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

        # Train/validate model
    model = ResNet50(batch_size=1, num_chan=3, classes=3, dropout=0.1)
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda:' + str(torch.cuda.current_device())
    model.eval()

    # try to load saved model weights from file
    try:
        # Load the previously saved model state
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        # print("MODEL STATE: \n", checkpoint)
    except (ImportError, IOError) as ex:
        print("NO EXISTING MODEL STATES IN FILE -- ", ex)

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

        tile_results = np.zeros((ht + 1, wt + 1))
        tile_predictions = np.zeros((ht + 1, wt + 1))
        overlay_actual = np.zeros(img.shape, dtype=np.uint8)
        overlay_predicted = np.zeros(img.shape, dtype=np.uint8)
        result = 0
        predicted = 0
        ctr = 0
        total = 0
        for col in range(wt + 1):
            jdx = col * tile
            for row in range(ht + 1):
                kdx = row * tile
                img_tile = img[kdx: kdx + tile, jdx: jdx + tile, :]
                gray_tile = cv2.cvtColor(img_tile, cv2.COLOR_BGR2GRAY)
                # determine tile actual result
                mask_tile = mask[kdx: kdx + tile, jdx: jdx + tile, 0]
                mmax = np.amax(mask_tile)
                if flag:
                    result = mmax
                    if mmax > 2:
                        result = 2
                else:
                    if mmax == 1 or mmax == 2:
                        result = 1
                    if mmax > 2:
                        result = 2

                # determine tile prediction from model
                mean = np.mean(img_tile)
                stddev = np.std(img_tile)
                img_tile = (img_tile - mean) / (stddev + 1.0e-8)
                img_tile = np.swapaxes(img_tile, 0, 2) 
                img_tile = np.swapaxes(img_tile, 1, 2)
                x_img_tile = torch.unsqueeze(torch.from_numpy(img_tile).to(dtype=torch.float32, device=device), 0)
                model.zero_grad()
                outputs = model(x_img_tile)
                predicted = torch.argmax(torch.sigmoid(outputs), dim=1).cpu().numpy()[0]
                # print(torch.sigmoid(outputs), predicted)
                
                tile_results[row, col] = result
                tile_predictions[row, col] = predicted

                if cv2.countNonZero(gray_tile) > 0.15 * tile * tile:
                    overlay_actual[kdx: kdx + tile, jdx: jdx + tile, 0] = overlay_mask[result][0]
                    overlay_actual[kdx: kdx + tile, jdx: jdx + tile, 1] = overlay_mask[result][1]
                    overlay_actual[kdx: kdx + tile, jdx: jdx + tile, 2] = overlay_mask[result][2]

                    overlay_predicted[kdx: kdx + tile, jdx: jdx + tile, 0] = overlay_mask[predicted][0]
                    overlay_predicted[kdx: kdx + tile, jdx: jdx + tile, 1] = overlay_mask[predicted][1]
                    overlay_predicted[kdx: kdx + tile, jdx: jdx + tile, 2] = overlay_mask[predicted][2]
                    total += 1
                    if result == predicted:
                        ctr += 1

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

        fig = plt.figure(figsize=(16, 8))
        ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
        ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)

        ax1.imshow(img)
        ax1.imshow(img_mask, alpha=0.4)
        ax1.imshow(overlay_actual, alpha=0.3)
        ax2.imshow(img)
        ax2.imshow(img_mask, alpha=0.4)
        ax2.imshow(overlay_predicted, alpha=0.3)

        ax1.xaxis.set_ticks([])
        ax1.yaxis.set_ticks([])
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        ax2.xaxis.set_ticks([])
        ax2.yaxis.set_ticks([])
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        acc = ctr / total
        print('acc: ', acc)
        ax1.set_title('ACTUAL: {} {}'.format(title, img_name))
        ax2.set_title('PREDICTED: {} {} with acc={:.3f}'.format(title, img_name, acc))

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
