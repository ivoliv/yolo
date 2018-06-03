import torch
import torch.nn as nn

class ConvBatchLeaky(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding):
        super(ConvBatchLeaky, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        self.bn_layer = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        x = self.activation(x)

        return x

class Yolov2Net(nn.Module):
    def __init__(self, grid_size, n_bnd_boxes, n_classes):
        super(Yolov2Net, self).__init__()
        self.grid_size = grid_size
        self.n_bnd_boxes = n_bnd_boxes
        self.n_classes = n_classes
        self.cell_tensor_len = n_bnd_boxes * (5 + n_classes)

        self.layers1 = nn.Sequential(
            ConvBatchLeaky(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers2 = nn.Sequential(
            ConvBatchLeaky(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers3 = nn.Sequential(
            ConvBatchLeaky(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(128, 64, kernel_size=1, stride=1, padding=0),
            ConvBatchLeaky(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers4 = nn.Sequential(
            ConvBatchLeaky(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(256, 128, kernel_size=1, stride=1, padding=0),
            ConvBatchLeaky(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers5 = nn.Sequential(
            ConvBatchLeaky(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBatchLeaky(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBatchLeaky(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers6 = nn.Sequential(
            ConvBatchLeaky(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBatchLeaky(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBatchLeaky(512, 1024, kernel_size=3, stride=1, padding=1),
        )

        self.layers7 = nn.Sequential(
            ConvBatchLeaky(1024, 1000, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=grid_size),
        )

        self.classification = nn.Sequential(
            nn.Linear(1000,
                      grid_size * grid_size * (n_bnd_boxes * (n_classes + 5))),
            #nn.ReLU()  <== I don't think I'm supposed to use a relu here!
        )

    def forward(self, x):
        print('size 0      =', x.size())
        x = self.layers1(x)
        print('size 1      =', x.size())
        x = self.layers2(x)
        print('size 2      =', x.size())
        x = self.layers3(x)
        print('size 3      =', x.size())
        x = self.layers4(x)
        print('size 4      =', x.size())
        x = self.layers5(x)
        print('size 5      =', x.size())
        x = self.layers6(x)
        print('size 6      =', x.size())
        x = self.layers7(x)
        print('size 7      =', x.size())
        x = x.view(x.size()[0], -1)
        print('size 8      =', x.size())
        x = self.classification(x)
        print('size 9      =', x.size())

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid_id = i * self.grid_size + j
                for b in range(self.n_bnd_boxes):
                    class_start_idx = grid_id*self.cell_tensor_len + (b    ) * (self.n_classes + 5)
                    class_end_idx   = grid_id*self.cell_tensor_len + (b + 1) * (self.n_classes + 5)
                    #print((i, j, b), grid_id, class_start_idx, class_end_idx)

                    # Apply sigmoid to: objectness, tx, ty - not to t_w, t_h
                    x[:, class_start_idx: class_start_idx+3] = torch.sigmoid(x[:, class_start_idx: class_start_idx+3])
                    # Apply sigmoid to all classes
                    #TODO: class CEL: Decide if to keep this:
                    #x[:, class_start_idx+5: class_end_idx] = torch.sigmoid(x[:, class_start_idx+5: class_end_idx])

        return x




class tinyYolov2Net(nn.Module):
    def __init__(self, grid_size, n_bnd_boxes, n_classes):
        super(tinyYolov2Net, self).__init__()
        self.grid_size = grid_size
        self.n_bnd_boxes = n_bnd_boxes
        self.n_classes = n_classes
        self.cell_tensor_len = n_bnd_boxes * (5 + n_classes)

        self.layers1 = nn.Sequential(
            ConvBatchLeaky(3, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers2 = nn.Sequential(
            ConvBatchLeaky(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers3 = nn.Sequential(
            ConvBatchLeaky(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers4 = nn.Sequential(
            ConvBatchLeaky(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers5 = nn.Sequential(
            ConvBatchLeaky(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers6 = nn.Sequential(
            ConvBatchLeaky(256, 512, kernel_size=3, stride=1, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.layers7 = nn.Sequential(
            ConvBatchLeaky(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(1024, 1024, kernel_size=3, stride=1, padding=1),
            ConvBatchLeaky(1024, 225, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=grid_size),
        )

        self.classification = nn.Sequential(
            nn.Linear(225,
                      grid_size * grid_size * (n_bnd_boxes * (n_classes + 5))),
            #nn.ReLU()  <== I don't think I'm supposed to use a relu here!
        )

    def forward(self, x):
        #print('size 0      =', x.size())
        x = self.layers1(x)
        #print('size 1      =', x.size())
        x = self.layers2(x)
        #print('size 2      =', x.size())
        x = self.layers3(x)
        #print('size 3      =', x.size())
        x = self.layers4(x)
        #print('size 4      =', x.size())
        x = self.layers5(x)
        #print('size 5      =', x.size())
        x = self.layers6(x)
        #print('size 6      =', x.size())
        x = self.layers7(x)
        #print('size 7      =', x.size())
        x = x.view(x.size()[0], -1)
        #print('size 8      =', x.size())
        x = self.classification(x)
        #print('size 9      =', x.size())

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid_id = i * self.grid_size + j
                for b in range(self.n_bnd_boxes):
                    class_start_idx = grid_id*self.cell_tensor_len + (b    ) * (self.n_classes + 5)
                    class_end_idx   = grid_id*self.cell_tensor_len + (b + 1) * (self.n_classes + 5)
                    #print((i, j, b), grid_id, class_start_idx, class_end_idx)

                    # Apply sigmoid to: objectness, tx, ty - not to t_w, t_h
                    x[:, class_start_idx: class_start_idx+3] = torch.sigmoid(x[:, class_start_idx: class_start_idx+3])
                    # Apply sigmoid to all classes
                    #TODO: class CEL: Decide if to keep this:
                    #x[:, class_start_idx+5: class_end_idx] = torch.sigmoid(x[:, class_start_idx+5: class_end_idx])

        return x


