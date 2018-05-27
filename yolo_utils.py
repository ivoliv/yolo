import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import numpy as np


def different_center_IOU(boxA, boxB):
    """
    Determine intersection over union between two boxes 'A' and 'B'
    :param boxA: (center x, center y, width, height)
    :param boxB: (center x, center y, width, height)
    :return: IOU
    """
    # ll: lower left corner
    # ru: upper right orner

    boxA_ll = (boxA[0] - boxA[2] / 2, boxA[1] - boxA[3] / 2)
    boxA_ur = (boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2)

    boxB_ll = (boxB[0] - boxB[2] / 2, boxB[1] - boxB[3] / 2)
    boxB_ur = (boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2)

    x_ll = max(boxA_ll[0], boxB_ll[0])
    y_ll = max(boxA_ll[1], boxB_ll[1])
    x_ur = min(boxA_ur[0], boxB_ur[0])
    y_ur = min(boxA_ur[1], boxB_ur[1])

    # intersection rectangle
    interArea = (x_ur - x_ll) * (y_ur - y_ll)

    # union area
    boxAArea = (boxA_ur[0] - boxA_ll[0]) * (boxA_ur[1] - boxA_ll[1])
    boxBArea = (boxB_ur[0] - boxB_ll[0]) * (boxB_ur[1] - boxB_ll[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def voc_metadata(voc_dataset):
    cell_size = voc_dataset.cell_size
    grid_size = voc_dataset.grid_size
    n_bnd_boxes = voc_dataset.n_bnd_boxes
    n_classes = voc_dataset.n_classes

    return cell_size, grid_size, n_bnd_boxes, n_classes


def get_boxes(sample_batch, voc_dataset,
              show_ix, orig_file, orig_size):

    batch_size = sample_batch['grid_tensor'].size()[0]-1
    assert(show_ix <= batch_size)
    cell_size, grid_size, n_bnd_boxes, n_classes = voc_metadata(voc_dataset)

    image = sample_batch['image'][show_ix]
    grid_tensor = sample_batch['grid_tensor'][show_ix]

    image = transforms.ToPILImage()(image)

    transf_size_x = image.size[0]
    transf_size_y = image.size[1]

    if orig_size:
        org_size_x = int(sample_batch['orig_size'][show_ix][0])
        org_size_y = int(sample_batch['orig_size'][show_ix][1])
        image = transforms.Resize((org_size_y, org_size_x))(image)  # note this takes (h,w) not (w,h)!
        factor_x = org_size_x / transf_size_x
        factor_y = org_size_y / transf_size_y
        transf_size_x = image.size[0]
        transf_size_y = image.size[1]
    else:
        factor_x = 1
        factor_y = 1

    if orig_file:
        path_jpeg = voc_dataset.path_jpg
        image = Image.open(os.path.join(path_jpeg, sample_batch['name'][show_ix] + '.jpg'))

    n_boxes = 0
    boxes = []
    for i in range(grid_size):
        cell_x_min = cell_size * i
        for j in range(grid_size):
            cell_y_min = cell_size * j
            next_offset = 0
            for b in range(n_bnd_boxes):
                offset = next_offset
                next_offset = (b + 1) * (5 + n_classes)
                if grid_tensor[i, j, offset] > 0.1:  # = 1 is ground truth object in anchor box
                    t_x = grid_tensor[i, j, offset + 1]
                    t_y = grid_tensor[i, j, offset + 2]
                    t_w = grid_tensor[i, j, offset + 3]
                    t_h = grid_tensor[i, j, offset + 4]
                    b_x = t_x + i
                    b_y = t_y + j
                    obj_center_x = b_x * cell_size
                    obj_center_y = b_y * cell_size
                    obj_w = voc_dataset.anchor_priors[b][0] * np.exp(t_w)
                    obj_h = voc_dataset.anchor_priors[b][1] * np.exp(t_h)
                    val, k = torch.max(grid_tensor[i, j, offset+5:next_offset], 0)
                    name = voc_dataset.classes[k]
                    color = voc_dataset.class_colors[k]
                    n_boxes += 1
                    boxes.append((name, color, obj_center_x, obj_center_y, obj_w, obj_h))

    return boxes, factor_x, factor_y, image, transf_size_x, transf_size_y


def draw_boxes(boxes, grid_size, factor_x, factor_y, image,
               padx, pady):

    #print(' boxes =', boxes)
    for b in boxes:
        name = b[0]
        color = b[1]
        obj_center_x = int(b[2])
        obj_center_y = int(b[3])
        obj_w = int(b[4] * grid_size)
        obj_h = int(b[5] * grid_size)

        xmin = int((obj_center_x - obj_w/2) * factor_x)
        xmax = int((obj_center_x + obj_w/2) * factor_x)
        ymin = int((obj_center_y - obj_h/2) * factor_y)
        ymax = int((obj_center_y + obj_h/2) * factor_y)

        draw = ImageDraw.Draw(image)
        draw.rectangle([xmin, ymin, xmax, ymax],
                       fill=None, outline=color)
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 10)

        fw, fh = font.getsize(name)
        draw.rectangle([xmin, ymin, xmin+fw+2*padx, ymin+fh+2*pady], fill=color)
        draw.text([xmin+padx, ymin+pady], name, font=font, fill='black')

    return draw, font


def visualize_with_boxes(sample_batch, voc_dataset,
                         show_ix=0, orig_file=True, orig_size=True):

    if orig_file == True:
        if orig_size == False:
            print('NOTE: visualize_with_boxes called with orig_file=True, orig_size=False')
            print('NOTE:   adjusting to orig_size = True')
            orig_size = True

    padx = 2
    pady = 1

    boxes, factor_x, factor_y, image, transf_size_x, transf_size_y \
        = get_boxes(sample_batch, voc_dataset, show_ix, orig_file, orig_size)

    draw, font = draw_boxes(boxes, voc_dataset.grid_size, factor_x, factor_y, image,
                            padx, pady)

    img_name = sample_batch['name'][show_ix]
    fw, fh = font.getsize(img_name)
    draw.rectangle([0, transf_size_y-fh-pady, fw+2*padx, transf_size_y], fill='black')
    draw.text([padx, transf_size_y-fh-pady], img_name, font=font, fill='white')

    image.show()


def show_some():
    for i_batch, sample_batch in enumerate(train_loader):

        if i_batch == 2:
            break

        print(i_batch, sample_batch['name'],
              sample_batch['image'].size(),
              sample_batch['orig_size'],
              sample_batch['grid_tensor'].size()
              )

        yolo_utils.visualize_with_boxes(sample_batch, voc_dataset, show_ix=0, orig_file=True)