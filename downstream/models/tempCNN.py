import os
import torch
import torch.nn as nn
import torch.utils.data



class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=10, kernel_size=7, hidden_dims=128, dropout=0.18203942949809093):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        
        # Global average pooling goes here
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # require NxTxD
        #x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)

        # global average pooling
        x = self.pool(x)
        
        return x


    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)




class TemporalCNN(nn.Module):
    def __init__(self, hidden_dim=128, expander_dim=256, num_classes=10):
        super().__init__()
         # Define the layers of the TempCNN encoder
        self.encoder_s1 = TempCNN(input_dim=2, kernel_size=7, hidden_dims=hidden_dim, dropout=0.5)
        self.encoder_s2 = TempCNN(input_dim=10, kernel_size=7, hidden_dims=hidden_dim, dropout=0.5)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, expander_dim),
            nn.ReLU(),
            nn.Linear(expander_dim, expander_dim),
            nn.ReLU()
        )
        # Create the final classifier layer
        self.classifier = nn.Linear(expander_dim, num_classes)

    # Pass the inputs through the TempCNN encoders
    def forward(self, s1, s2):
        repr_s1 = self.encoder_s1(s1)
        repr_s2 = self.encoder_s2(s2)

        # Flatten the representations
        repr_s1 = repr_s1.view(repr_s1.size(0), -1)
        repr_s2 = repr_s2.view(repr_s2.size(0), -1)

        # Concatenate the representations from both encoders
        combined_repr = torch.cat((repr_s1, repr_s2), dim=1)

        mlp_output = self.mlp(combined_repr)
        output = self.classifier(mlp_output)

        return output


# Create the model  
if __name__ == '__main__':
    model = TemporalCNN()

