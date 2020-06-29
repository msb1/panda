import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

epoch = []
loss = []
test_loss = []
acc = []
test_acc = []

with open('history.json', 'r') as f:
    for line in f:
        record = json.loads(line)
        print(record)
        epoch.append(record['epoch'])
        loss.append(record['loss'])
        test_loss.append(record['test_loss'])

        acc.append(record['acc'])
        test_acc.append(record['test_acc'])

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

ax1.set_xlabel('epoch')
ax1.set_xticks(epoch)
ax1.set_ylabel('loss')
l11, = ax1.plot(epoch, loss)
l12, = ax1.plot(epoch, test_loss)
ax1.legend((l11, l12), ('train', 'val'), loc='upper left', shadow=True)

ax2.set_xlabel('epoch')
ax2.set_xticks(epoch)
ax2.set_ylabel('accuracy')
l21, = ax2.plot(epoch, acc)
l22, = ax2.plot(epoch, test_acc)
ax2.legend((l21, l22), ('train', 'val'), loc='upper left', shadow=True)

plt.show()