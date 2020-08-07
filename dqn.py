import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, inputs,  outputs):

        super(DQN, self).__init__()
        self.lin1 = nn.Linear(inputs, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lstm = nn.LSTM(30,30, 1)
        self.lin3 = nn.Linear(30,outputs)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        #def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #    return (size - (kernel_size - 1) - 1) // stride  + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #linear_input_size = convw * convh * 32
        #self.head = nn.Linear(inputs, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, hidden):


        #x = torch.from_numpy(x).float()
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        # This means that we are looking at a batch
        if x.dim() !=1:
            x = x.reshape(1, x.size()[0], 30)
            h0 = hidden[0].reshape(1, hidden[0].size()[0], 30)
            c0 = hidden[1].reshape(1, hidden[1].size()[0], 30)
        else:
            x = x.reshape(1,1,30)
            h0 = hidden[0]
            c0 = hidden[1]

        x, hidden = self.lstm(x, (h0, c0))
        x = F.relu(self.lin3(x))
        return x, hidden # self.head(x.view(x.size(0), -1))