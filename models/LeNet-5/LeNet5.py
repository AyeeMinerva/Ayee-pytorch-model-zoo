import torch.nn as nn
import torch.nn.functional as F

"""
Image: 28*28*1
"""

class LeNet5(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet5,self).__init__()
        #######输入#######
        #image: 28*28*1
        
        #######卷积层部分#######
        # 卷积层 1: 1个输入通道 (灰度图像), 6个输出通道, 5x5 卷积核
        # 28*28*1 -> 28*28*6
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        
        # 平均池化层 1: 2x2 池化窗口, 步长为 2
        # 28*28*6 -> 14*14*6
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
        
        # 卷积层 2: 6个输入通道, 16个输出通道, 5x5 卷积核
        # 14*14*6 -> 10*10*16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 平均池化层 2: 2x2 池化窗口, 步长为 2
        # 10*10*16 -> 5*5*16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        
        #######全连接层部分#######
        # 全连接层 1: 16*5*5 -> 120 ：将卷积层 2 的输出展平
        # 5*5*16 -> 400 -> 120
        self.fc1 = nn.Linear(in_features=16*5*5,out_features=120)
        
        # 全连接层 2: 120个输入，84个输出
        # 120 -> 84
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        
        # 全连接层 3 (输出层): 84个输入，num_classes个输出
        # 84 -> num_classes
        self.fc3 = nn.Linear(in_features=84,out_features=num_classes)
        
    def forward(self,x):
        # 卷积
        x=self.conv1(x)
        x=F.sigmoid(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        x=self.pool2(x)
        
        x=x.view(-1,16*5*5) #展平
        
        # 全连接
        x=self.fc1(x)
        x=F.sigmoid(x)
        x=self.fc2(x)
        x=F.sigmoid(x)
        x=self.fc3(x)
        return x
    
if __name__ == '__main__':
    model = LeNet5()
    print(model)