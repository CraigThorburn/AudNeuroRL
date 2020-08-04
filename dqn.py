import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, inputs,  outputs):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(inputs, 30)
        self.sm1 = nn.Softmax()
        self.lin2 = nn.Linear(30, 30)
        self.sm2 =  nn.Softmax()
        self.lin3 = nn.Linear(30,30)
        self.sm3 =  nn.Softmax()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        #def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #    return (size - (kernel_size - 1) - 1) // stride  + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #linear_input_size = convw * convh * 32
        self.head = nn.Linear(inputs, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))