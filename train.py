import time
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from utils.genData import ImageDataSet
from models.resnet50 import ResNet50


data_dir = 'D:\\******************************\\train_images'
mask_dir = 'D:\\******************************\\train_label_masks'
checkpoint_file = 'C:\\******************************\\checkpoint.pt'

# training parameters
EPOCHS = 30
TILE_SIZE = 224
BATCH_SIZE = 48
PARAMS = {'batch_size': BATCH_SIZE, 'shuffle': True, 'drop_last': True}

# Plot Model History
def plot(epochs, history):
    xdata = list(range(1, epochs + 1))
    plt.plot(xdata, history['accuracy'], label='Train Acc')
    plt.plot(xdata, history['val_accuracy'], label='Val Acc')
    plt.plot(xdata, history['loss'], label='Train Loss')
    plt.plot(xdata, history['val_loss'], label='Val Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.show()


def main():
    # read in Panda tiles
    img_tiles = []
    with open("\\******************************\\panda_tiles.txt", 'r') as rfile:
        img_tiles = [tile for tile in list(rfile.readlines())]

    print("Number of image tiles:", len(img_tiles))

    # split training data for test and validation
    np.random.seed(42)
    train_tiles, val_tiles = train_test_split(img_tiles, test_size=0.1, random_state=42)

    # initialize timer before training model
    startTime = time.time()
    # Train/validate model
    model = ResNet50(batch_size=BATCH_SIZE, num_chan=3, classes=3, dropout=0.1)
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda:' + str(torch.cuda.current_device())

    # criterion = nn.SmoothL1Loss()
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.AdamW(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    print(model)

    # try to load saved model weights from file
    epochs_run = 0
    try:
        # Load the previously saved model state
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_run = checkpoint['epoch']
        loss = checkpoint['loss']
        # print("MODEL STATE: \n", checkpoint)
    except (ImportError, IOError) as ex:
        print("NO EXISTING MODEL STATES IN FILE -- ", ex)

    # define Data Generators

    training_dataset = ImageDataSet(train_tiles, data_dir, mask_dir, batch_size=BATCH_SIZE, tile_size=TILE_SIZE, num_class=3)
    validation_dataset = ImageDataSet(val_tiles, data_dir, mask_dir, batch_size=BATCH_SIZE, tile_size=TILE_SIZE, num_class=3)
    train_loader = DataLoader(training_dataset, **PARAMS)
    test_loader = DataLoader(validation_dataset, **PARAMS)

    # start model with Data Generators
    total_step = len(train_loader)
    for epoch in range(epochs_run, EPOCHS):
        model.train()
        history = {}
        # Train the model
        loss_list = []
        acc_list = []
        # print("EPOCH: ", epoch + 1)
        for i, (tiles, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            print('.', end="")
            outputs = model(tiles)
            # print(tiles.shape, tiles.dtype, tiles.device, targets.shape, targets.dtype, targets.device, outputs.shape, outputs.dtype, outputs.device)
            loss = criterion(outputs, targets)
            # print("LOSS:", loss)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            # determine accuracy for image only pixels where pixel is healthy or has gleason score
            # print(outputs.shape)
            predicted = torch.argmax(torch.sigmoid(outputs), dim=1)
            labels = torch.argmax(targets, dim=1)
            # print(labels.shape, predicted.shape)
            correct = (predicted == labels).sum().item()
            acc = correct / labels.shape[0]
            acc_list.append(acc)

            # print interim results to console (every 100 batches in each epoch)
            if i == 0:
                continue
            # if (i + 1) % 200 == 0:                                    # batch size of 48
            if (i + 1) % 20000 == 0 or (i + 1) == total_step:           # batch size of 1
                print('\n')
                print("-------------------------------------------------------------------------------------------------------")
                print("Epoch [{}/{}], Step [{}/{}], Avg Loss: {:.4f}, Avg Acc: {:.4f}"
                      .format(epoch + 1, EPOCHS, i + 1, total_step, sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)))
                print("-------------------------------------------------------------------------------------------------------")
                # print('\n')

        history['epoch'] = epoch + 1
        history['loss'] = sum(loss_list) / len(loss_list)
        history['acc'] = sum(acc_list) / len(acc_list)

        # Test the model
        model.eval()
        with torch.no_grad():
            loss_list = []
            acc_list = []

            for tiles, targets in test_loader:
                print('-', end="")
                outputs = model(tiles)
                loss = criterion(outputs, targets)
                loss_list.append(loss.item())
                # determine accuracy for image only pixels where pixel is healthy or has gleason score

                predicted = torch.argmax(torch.sigmoid(outputs), dim=1)
                labels = torch.argmax(targets, dim=1)
                # print(labels.shape, predicted.shape)
                correct = (predicted == labels).sum().item()
                acc = correct / labels.shape[0]
                acc_list.append(acc)

            print("\n")
            print('>>>----------------------------------------------------------------------------------------------<<<')
            print("TEST -- Avg Loss: {:.4f}, Avg Acc: {:.4f}".format(sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)))
            print('>>>----------------------------------------------------------------------------------------------<<<')

            history['test_loss'] = sum(loss_list) / len(loss_list)
            history['test_acc'] = sum(acc_list) / len(acc_list)

        # save model, optimizer, etc., at end of each epoch (checkpoint)
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss},
                   checkpoint_file)
        with open("history.json", "a") as jsonfile:
            if epoch > 0: 
                jsonfile.write('\n')
            json.dump(history, jsonfile)

    # output history file and execution time
    print('TRAINING AND VALIDATION COMPLETE... elapsed time: {} sec'.format(time.time() - startTime))


if __name__ == "__main__":
    main()
