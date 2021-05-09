import torch 
import torch.nn.functional as F
import torch.nn as nn


class AuxNet(torch.nn.Module):
    def __init__(self, input_dim, output, output_aux) :
        super(AuxNet, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(input_dim, output))
        self.fc_aux = torch.nn.Sequential(torch.nn.Linear(input_dim, output_aux))

    def forward(self, x):
        y = self.fc(x)
        y_aux = self.fc_aux(x)
        return y, y_aux



# class AuxNet(torch.nn.Module):
#     def __init__(self, original_model, num_classes, num_classes_aux):
#         super(AuxNet, self).__init__()
#         # Everything except the last linear layer
#         self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

#         self.classifier1 = torch.nn.Linear(1024, num_classes)

#         self.classifier2 = torch.nn.Linear(1024, num_classes_aux)

#         # Freeze those weights
#         for p in self.features.parameters():
#             p.requires_grad = True


#     def forward(self, x):
#         f = self.features(x) 
   
#         f = F.adaptive_avg_pool2d(f, (1, 1))
#         f = torch.flatten(f, 1)

#         y1 = self.classifier1(f)
#         y2 = self.classifier2(f)
#         return y1, y2



# class Net(nn.Module):
#     def __init__():
#         super().__init__()
#         self.linear = Linear(300, 200)
#         self.obj1 = Linear(200, 5)
#         self.obj2 = Linear(200, 1)

#     def forward(inputs):
#         out = self.linear(inputs)
#         out_obj1 = self.obj1(out)
#         out_obj2 = self.obj2(out)
#         return out_obj1, out_obj2





