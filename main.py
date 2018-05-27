import numpy as np
import time
from yolo_net import Yolov2Net
import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from yolo_dataloaders import VOCDataset
import sys
import os
import local

path_jpeg = os.path.join(local.VOC_path, 'JPEGImages')
path_annotations = os.path.join(local.VOC_path, 'Annotations')

print('Images in', path_jpeg)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

voc_dataset = VOCDataset(path_jpg=path_jpeg,
                         path_xml=path_annotations,
                         grid_size=local.grid_size,
                         output_size=local.scaled_img_size)

n_images = len(voc_dataset)
indices = list(range(n_images))

split_train = int(np.floor(local.train_ratio * n_images))
split_valid = n_images - split_train
train_idx, valid_idx = indices[:split_train], indices[split_train:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(voc_dataset,
                          batch_size=local.batch_size,
                          sampler=train_sampler,
                          num_workers=local.num_workers)

valid_loader = DataLoader(voc_dataset,
                          batch_size=local.batch_size,
                          sampler=valid_sampler,
                          num_workers=local.num_workers)


model = Yolov2Net(local.grid_size, voc_dataset.n_bnd_boxes, voc_dataset.n_classes).to(device)

#print(model)

loss_function_CEL = torch.nn.CrossEntropyLoss()
loss_function_BCEL = torch.nn.BCELoss()
loss_function_MSEL = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# TODO: clean this all up
# Create tensor sizes required for building the loss function
n_b = 5 + voc_dataset.n_classes
n_ij = voc_dataset.n_bnd_boxes * n_b
obj_pred_idx = []
class_pred_idx = []
idx = 0
for i in range(local.grid_size):
    for j in range(local.grid_size):
        for b in range(voc_dataset.n_bnd_boxes):
            obj_pred_idx.append(idx)
            for l in range(idx + 5, idx + n_b):
                class_pred_idx.append(l)
            idx += n_b

print('\nrunning on: {}\n'.format(device))

loss_log = []

start_time = time.time()
total_batches = (local.num_epochs+1) * local.batch_quit
total_time_estimate = None
time_remaining = None

for epoch in range(local.num_epochs):

    model.train()

    epoch_loss = 0

    for i_batch, sample_batch in enumerate(train_loader):

        batch_start = time.time()

        if local.batch_quit and i_batch+1 > local.batch_quit:
            break

        print('[%d, %3d]' %
              (epoch + 1, i_batch + 1), end='')

        model.zero_grad()

        x = torch.autograd.Variable(sample_batch['image'], requires_grad=True).to(device)
        #print('input size  =', x.size())
        print(' =>', end='')
        sys.stdout.flush()
        pred = model(x)
        #print('output size =', pred.size())

        target = torch.autograd.Variable(sample_batch['grid_tensor'], requires_grad=False)
        target = target.to(device)

        batch_n_img = target.size()[0]

        # For all images in batch:
        #   Defines if an object appears in cell '(i,j)', attributed to bounding box 'b', attributed to class 'c'
        appears = np.zeros((local.batch_size,
                            voc_dataset.grid_size, voc_dataset.grid_size,
                            voc_dataset.n_bnd_boxes, 1))

        objects_detected = []  # list of (cell(i,j), bnd_box) of objects detected in image

        for img in range(batch_n_img):
            #yolo_utils.visualize_with_boxes(sample_batch, voc_dataset, show_ix=img, orig_file=True)
            #print('\n', '#'*30)
            for i in range(voc_dataset.grid_size):
                for j in range(voc_dataset.grid_size):
                    for b in range(voc_dataset.n_bnd_boxes):
                        class_start_idx = b * (voc_dataset.n_classes + 5)
                        if target[img, i, j, class_start_idx] > 0:
                            #print(target[img, i, j, class_start_idx:(b+1)*(voc_dataset.n_classes+5)])
                            class_end_idx = (b+1)*(voc_dataset.n_classes+5)
                            class_id = np.argmax(target[img, i, j,
                                                 class_start_idx+5: class_end_idx].numpy())
                            coords = target[img, i, j, class_start_idx+1: class_start_idx+5].numpy()
                            objects_detected.append(((i,j), b))
                            #print(' cell={}, bb={}, class_id={}, coords={}'.format((i, j), b, class_id, coords))

        target = target.view(pred.size())
        #print('target / pred size =', target.size(), pred.size())

        loss = 0
        cell_tensor_len = voc_dataset.n_bnd_boxes * (5 + voc_dataset.n_classes)
        for im in range(target.size()[0]):
            for i in range(voc_dataset.grid_size):
                for j in range(voc_dataset.grid_size):
                    grid_id = i * voc_dataset.grid_size + j
                    for b in range(voc_dataset.n_bnd_boxes):
                        class_start_idx = grid_id * cell_tensor_len + (b) * (voc_dataset.n_classes + 5)
                        class_end_idx = grid_id * cell_tensor_len + (b + 1) * (voc_dataset.n_classes + 5)

                        # Objectness loss
                        if target[im, class_start_idx + 0] == 1:
                            weight = 1
                        else:
                            weight = 0.5

                        loss += weight * loss_function_BCEL(pred[im, class_start_idx + 0], target[im, class_start_idx + 0])

                        # Coordinates loss: (t_x, t_y) \in [0,1], BCEL
                        #loss += loss_function_BCEL(pred[im, class_start_idx + 1], target[im, class_start_idx + 1])
                        #loss += loss_function_BCEL(pred[im, class_start_idx + 2], target[im, class_start_idx + 2])
                        # Coordinates loss: (t_w, t_h) unconstrained, MSEL
                        #loss += loss_function_MSEL(pred[im, class_start_idx + 3], target[im, class_start_idx + 3])
                        #loss += loss_function_MSEL(pred[im, class_start_idx + 4], target[im, class_start_idx + 4])

                        # Class losses
                        # TODO: class losses

        loss /= local.batch_size



        print(' loss: %.3f' % loss.item(), end='')
        sys.stdout.flush()

        print(' <=', end='')
        sys.stdout.flush()
        loss.backward()

        batch_end = time.time()
        batch_time = batch_end - batch_start
        time_sofar = batch_end - start_time
        if total_time_estimate:
            w = .1
            total_time_estimate = (1-w) * total_time_estimate + w * (batch_time * total_batches)
        else:
            total_time_estimate = batch_time * total_batches

        time_remaining = max(0, (total_time_estimate - time_sofar) / 60)

        print(' %.0f s' % batch_time, '(%.2f m est remaining)' % time_remaining)
        sys.stdout.flush()

        optimizer.step()

        epoch_loss += loss.item()

        if local.batch_quit == 0:
            if i_batch == 0:
                loss_log.append(epoch_loss)
                epoch_loss = 0

        if i_batch+1 == local.batch_quit:
            loss_log.append(epoch_loss / local.batch_quit)
            epoch_loss = 0

        best_model_weights = copy.deepcopy(model.state_dict())

        #exit()


def output_predict_vec():

    for i in range(voc_dataset.grid_size):
        for j in range(voc_dataset.grid_size):
            grid_id = i * voc_dataset.grid_size + j
            for b in range(voc_dataset.n_bnd_boxes):
                class_start_idx = grid_id * cell_tensor_len + (b) * (voc_dataset.n_classes + 5)
                class_end_idx = grid_id * cell_tensor_len + (b + 1) * (voc_dataset.n_classes + 5)

                print((i, j),
                      pred[0, class_start_idx + 0].detach().numpy(),
                      target[0, class_start_idx + 0].detach().numpy(), end='')
                if target[0, class_start_idx + 0] > 0:
                    print(' *****')
                else:
                    print()

    for im in range(min(target.size()[0], 5)):
        for i in range(pred.size()[1]):
            if target[im, i].detach().numpy() > 0:
                print(im, i, pred[im, i].detach().numpy(), target[im, i].detach().numpy(), end='')
                if target[im, i].detach().numpy() == 1:
                    print(' <{}'.format('-'*50))
                else:
                    print()

def display_loss_history():
    import matplotlib.pyplot as plt
    plt.plot(loss_log)
    plt.show()

output_predict_vec()
display_loss_history()