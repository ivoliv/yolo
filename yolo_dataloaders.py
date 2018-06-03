import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import random
import yolo_utils
import numpy as np

class VOCDataset(Dataset):
    """
    Dataset for VOC inputs
    """
    def __init__(self, path_jpg, path_xml, output_size=448,
                 grid_size=7):
        """
        Set up to read in VOC 2012 format image and annotation data. Also read in all
        annotation (xml) data - image (jpg) data is ready in __getitem__().

        :param path_jpg: contains all jpg image files (e.g. 2007_000027.jpg)
        :param path_xml: contains all xml annotation files (e.g. 2007_000027.xml)
        :param transform: transformations to the image data and annotation coords
        :param include_classes: subset of classes to include (None to include all)
        """
        try:
            assert(os.path.isdir(path_jpg))
            assert(os.path.isdir(path_xml))
            self.path_jpg = path_jpg
            self.path_xml = path_xml
            self.output_size = output_size
            self.grid_size = grid_size
            self.cell_size = output_size // grid_size

            all_xml_files = os.listdir(self.path_xml)
            print('Images in {}, total number in set: {}'.format(
                path_jpg,
                len(all_xml_files))
            )

            self.image_names = []
            for xml_file in all_xml_files:
                self.image_names.append(xml_file[:-4])

            # TODO: automate this part
            self.anchor_priors = {0: (1.08, 1.19),
                                  1: (3.42, 4.41),
                                  2: (6.63, 11.38),
                                  3: (9.42, 5.11),
                                  4: (16.62, 10.52)}
            # TEMP:
            #self.anchor_priors = {0: (16.62, 10.52)}
            self.n_bnd_boxes = len(self.anchor_priors)

            # TODO: automate this part
            self.classes = [
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
            ]

            self.classes_idx = {x: self.classes.index(x) for x in self.classes}
            # randomly generate colors for display of bounding boxes
            self.class_colors = [(random.randint(100, 255),
                                  random.randint(100, 255),
                                  random.randint(100, 255)) for k in self.classes_idx]

            self.n_classes = len(self.classes)

            #print('classes:', self.classes)
            #print('classes_idx:', self.classes_idx)

        except OSError:
            print('ERROR: VOCDataset initialization failed.')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        name = self.image_names[item]
        img_file = os.path.join(self.path_jpg, name + '.jpg')
        xml_file = os.path.join(self.path_xml, name + '.xml')

        image = Image.open(img_file)
        orig_size_x = image.size[0]
        orig_size_y = image.size[1]

        boxes = self.MakeDict(xml_file)
        #print(boxes)

        if self.output_size:
            image_trans = transforms.Resize((self.output_size, self.output_size))(image)
            boxes_trans = self.ResizeBoxes(orig_size_x, orig_size_y, boxes)
        else:
            image_trans = image
            boxes_trans = boxes

        image_trans = transforms.ToTensor()(image_trans)
        image_trans = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image_trans)

        #print(boxes_trans)

        #  - define number of bounding boxes per class B
        #  - define 5 parameters within cell: (x, y), (w, h), Prob
        #
        #  from https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L39:
        #     l.outputs = h*w*n*(classes + 4 + 1);
        #
        #  - grid tensor: cells x cells x (n_bnd_boxes * (n_classes + 5)):
        #          "5" : [p, bx, by, bw, bh]
        #                 p  = probability of an object for the cell/bounding (anchor) box
        #                 bx = bounding box x location in box as fraction of box dimension
        #                 by = bounding box y location " " " "
        #                 bw = bounding box width
        #                 bh = bounding box height

        grid_tensor = torch.zeros(self.grid_size, self.grid_size, self.n_bnd_boxes*(self.n_classes + 5))

        #print('#'*30, name)
        #sys.stdout.flush()

        cell_id = 0
        for i in range(self.grid_size):
            cell_x_min = self.cell_size * i
            cell_x_max = self.cell_size * (i+1)
            for j in range(self.grid_size):
                cell_y_min = self.cell_size * j
                cell_y_max = self.cell_size * (j+1)
                cell_avail_priors = self.anchor_priors.copy()

                for obj in boxes_trans.items():

                    obj_center_x = obj[1][1][0]
                    obj_center_y = obj[1][1][1]

                    if obj_center_x in range(cell_x_min, cell_x_max) \
                            and obj_center_y in range(cell_y_min, cell_y_max):

                        obj_name = obj[1][0]
                        #print('   DETECTED:', obj_name, 'in cell', cell_id, (i, j))
                        #sys.stdout.flush()

                        # Scale coordinates
                        obj_w = obj[1][2][0] / self.grid_size
                        obj_h = obj[1][2][1] / self.grid_size

                        max_IOU_idx = None
                        max_IOU = 0
                        boxA = (obj_center_x, obj_center_y, obj_w, obj_h)
                        for p in cell_avail_priors:
                            boxB = (obj_center_x, obj_center_y,
                                    self.anchor_priors[p][0], self.anchor_priors[p][1])
                            IOU = yolo_utils.different_center_IOU(boxA, boxB)
                            if IOU > max_IOU:
                                max_IOU_idx = p
                                max_IOU = IOU
                            #print('prior {} IOU = {}'.format(p, IOU))
                            #sys.stdout.flush()

                        #print('>> max_IOU_idx =', max_IOU_idx)
                        #print(cell_avail_priors)
                        # TODO: shouldn't have to check this here because of check at bottom of loop
                        if len(cell_avail_priors) > 0:
                            try:
                                selected_prior = cell_avail_priors.pop(max_IOU_idx)
                            except:
                                print('\nException caused in pop! cell_avail_priors = ', cell_avail_priors)
                                break
                        else:
                            break

                        offset = max_IOU_idx * (5 + self.n_classes)

                        # Final scaling (extract DNN output)
                        t_x = (obj_center_x - cell_x_min) / (cell_x_max - cell_x_min)
                        t_y = (obj_center_y - cell_y_min) / (cell_y_max - cell_y_min)
                        t_w = np.log(obj_w / selected_prior[0])
                        t_h = np.log(obj_h / selected_prior[1])

                        grid_tensor[i, j, offset + 0] = 1  # there is an object
                        grid_tensor[i, j, offset + 1] = t_x
                        grid_tensor[i, j, offset + 2] = t_y
                        grid_tensor[i, j, offset + 3] = t_w
                        grid_tensor[i, j, offset + 4] = t_h
                        grid_tensor[i, j, offset + 5 +
                                    self.classes_idx[obj_name]] = 1

                        if len(cell_avail_priors) == 0:
                            break  # used all available priors for this cell - move on

                cell_id += 1

        #self.PrintNonzeroGridTensor(grid_tensor)

        sample = {'name': name, 'image': image_trans,
                  'orig_size': torch.tensor([orig_size_x, orig_size_y]),  # transpose required
                  'grid_tensor': grid_tensor}

        return sample

    def IOU(self, boxA, boxB):
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

    def PrintNonzeroGridTensor(self, grid_tensor):
        """
        Print only nonzero cells of the grid tensor
        """
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid_tensor[i, j, 0] == 1:
                    print((i, j), grid_tensor[i, j])


    def MakeDict(self, xml_file):
        """
        Make dictionary of all classes in the image file

        :param img_name: image file name (jpg and xml)
        :return: dictionary of {object_id: (center_x, center_y, width, height)}
        """
        boxes = {}
        root = ET.parse(xml_file).getroot()
        obj_idx = 0
        for child in root.findall('object'):
            class_name = child.find('name').text
            bndbox = {}
            for child2 in child.findall('bndbox'):
                for child3 in child2:
                    bndbox[child3.tag] = int(round(float(child3.text)))
            center = ((bndbox['xmin'] + bndbox['xmax']) // 2,
                      (bndbox['ymin'] + bndbox['ymax']) // 2)
            dims = ((bndbox['xmax'] - bndbox['xmin']),
                    (bndbox['ymax'] - bndbox['ymin']))
            boxes[obj_idx] = (class_name, center, dims)
            obj_idx += 1
        return boxes

    def ResizeBoxes(self, orig_size_x, orig_size_y, boxes):
        """
        Apply resize transform to boxes in the image
        """

        boxes_trans = {}
        for b in boxes.items():
            box_new_center_x = int(b[1][1][0] * self.output_size / orig_size_x)
            box_new_center_y = int(b[1][1][1] * self.output_size / orig_size_y)
            box_new_dims_x = int(b[1][2][0] * self.output_size / orig_size_x)
            box_new_dims_y = int(b[1][2][1] * self.output_size / orig_size_y)
            boxes_trans[b[0]] = (b[1][0],
                                 (box_new_center_x, box_new_center_y),
                                 (box_new_dims_x, box_new_dims_y))

        return boxes_trans

