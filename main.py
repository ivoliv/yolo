import numpy as np
import time
from yolo_net import Yolov2Net, tinyYolov2Net
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

voc_dataset = VOCDataset(path_jpg=path_jpeg,
                         path_xml=path_annotations,
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


if os.path.isfile('model_state.pt'):
    print('Found model_state.pt file, loading state...')
    model = torch.load('model_state.pt').to(device)
else:
    model = tinyYolov2Net(voc_dataset.grid_size, voc_dataset.n_bnd_boxes, voc_dataset.n_classes).to(device)

best_model_weights = copy.deepcopy(model.state_dict())

#print(model)

loss_function_CEL = torch.nn.CrossEntropyLoss()
loss_function_BCEL = torch.nn.BCELoss()
loss_function_MSEL = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=local.lr, momentum=0.9)

#optimizer = torch.optim.Adam(model.parameters(), lr=local.lr, amsgrad=True)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)



print('\nrunning on: {}\n'.format(device))

loss_log = []

start_time = time.time()
total_batches = (local.num_epochs+1) * local.batch_quit
total_time_estimate = None
time_remaining = None

best_loss = None

for epoch in range(local.num_epochs):

    model.train()

    epoch_loss = 0

    model.load_state_dict(best_model_weights)

    for i_batch, sample_batch in enumerate(train_loader):

        batch_start = time.time()

        if local.batch_quit and i_batch+1 > local.batch_quit:
            break

        print('[%d, %3d]' %
              (epoch + 1, i_batch + 1), end='')

        x = torch.autograd.Variable(sample_batch['image'], requires_grad=True).to(device)
        #print('input size  =', x.size())

        model.zero_grad()

        print(' =>', end='')
        sys.stdout.flush()
        pred = model(x)
        #print('output size =', pred.size())

        target = torch.autograd.Variable(sample_batch['grid_tensor'], requires_grad=False)
        target = target.to(device)

        batch_n_img = target.size()[0]

        objects_detected = []  # list of (cell(i,j), bnd_box) of objects detected in image

        target = target.view(pred.size())
        # print('target / pred size =', target.size(), pred.size())

        loss = 0
        coord_weight = 5
        cell_tensor_len = voc_dataset.n_bnd_boxes * (5 + voc_dataset.n_classes)
        for im in range(target.size()[0]):
            for i in range(voc_dataset.grid_size):
                for j in range(voc_dataset.grid_size):
                    grid_id = i * voc_dataset.grid_size + j
                    for b in range(voc_dataset.n_bnd_boxes):
                        class_start_idx = grid_id * cell_tensor_len + b * (voc_dataset.n_classes + 5)
                        class_end_idx = grid_id * cell_tensor_len + (b + 1) * (voc_dataset.n_classes + 5)

                        if target[im, class_start_idx] == 1:
                            obj_weight = 1

                            # Class losses
                            # TODO: class CEL: Decide if to keep this:
                            #loss += loss_function_BCEL(pred[im, class_start_idx + 5: class_end_idx],
                            #                           target[im, class_start_idx + 5: class_end_idx])
                            _, targ_class = torch.max(target[im, class_start_idx + 5: class_end_idx].view(-1), 0)
                            #print(pred[im, class_start_idx + 5: class_end_idx].view(1,-1).shape)
                            #print(targ_class.view(1).shape)
                            loss += loss_function_CEL(pred[im, class_start_idx + 5: class_end_idx].view(1,-1),
                                                      targ_class.view(1))

                            # Coordinates loss: (t_x, t_y) \in [0,1], BCEL
                            loss += coord_weight * loss_function_BCEL(pred[im, class_start_idx + 1],
                                                                      target[im, class_start_idx + 1])
                            loss += coord_weight * loss_function_BCEL(pred[im, class_start_idx + 2],
                                                                      target[im, class_start_idx + 2])

                            # Coordinates loss: (t_w, t_h) unconstrained, MSEL
                            loss += coord_weight * loss_function_MSEL(pred[im, class_start_idx + 3],
                                                                      target[im, class_start_idx + 3])
                            loss += coord_weight * loss_function_MSEL(pred[im, class_start_idx + 4],
                                                                      target[im, class_start_idx + 4])
                        else:
                            obj_weight = 0.5

                        # Objectness loss
                        loss += obj_weight * loss_function_BCEL(pred[im, class_start_idx + 0],
                                                                target[im, class_start_idx + 0])

        loss /= local.batch_size



        print(' loss: %.3f' % loss.item(), end='')
        sys.stdout.flush()

        if not best_loss or loss.item() < best_loss:
            best_loss = loss.item()
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model, 'model_state.pt')
            print('*', end='')
        else:
            print(' ', end='')

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

        #exit()

model.load_state_dict(best_model_weights)
torch.save(model, 'model_state.pt')

def output_predict_vec():

    for i in range(voc_dataset.grid_size):
        for j in range(voc_dataset.grid_size):
            grid_id = i * voc_dataset.grid_size + j
            print()
            for b in range(voc_dataset.n_bnd_boxes):
                class_start_idx = grid_id * cell_tensor_len + (b) * (voc_dataset.n_classes + 5)
                class_end_idx = grid_id * cell_tensor_len + (b + 1) * (voc_dataset.n_classes + 5)

                _, pred_class = torch.max(pred[0, class_start_idx + 5: class_end_idx].view(-1), 0)
                _, targ_class = torch.max(target[0, class_start_idx + 5: class_end_idx].view(-1), 0)

                print((i, j),
                      pred[0, class_start_idx + 0].detach().cpu().numpy(),
                      target[0, class_start_idx + 0].detach().cpu().numpy(), end='')
                if target[0, class_start_idx + 0] > 0:
                    print(' ***** target class: {}, pred class: {}'
                          .format(targ_class.cpu().numpy(), pred_class.cpu().numpy()))
                else:
                    print()

    for im in range(min(target.size()[0], 5)):
        for i in range(pred.size()[1]):
            if abs(target[im, i].detach().cpu().numpy()) > 0.001:
                print(im, i, pred[im, i].detach().cpu().numpy(), target[im, i].detach().cpu().numpy(), end='')
                if target[im, i].detach().cpu().numpy() == 1:
                    print(' <{}'.format('-'*50))
                else:
                    print()

def display_loss_history():
    if local.display:
        import matplotlib.pyplot as plt
        plt.plot(loss_log)
        plt.show()

    print(loss_log)

output_predict_vec()
display_loss_history()
