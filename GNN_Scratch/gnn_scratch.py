import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A 

        # A_hat = A + I
        self.A_hat = self.A + torch.eye(self.A.size(0))

        # Create diagonal matrix
        self.ones = torch.ones(input_dim, input_dim)
        self.D = torch.matmul(self.A.float(), self.ones.float())

        # Extract diagonal elements
        self.D = torch.diag(self.D)

         # Create a new tensor with the diagonal elements and zeros elsewhere
        self.D = torch.diag_embed(self.D)
        
        # Create D^{-1/2}
        self.D_neg_sqrt = torch.diag_embed(torch.diag(torch.pow(self.D, -0.5)))
        
        # Initialise the weight matrix as a parameter
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, X: torch.Tensor):
        # D^-1/2 * (A_hat * D^-1/2)
        support_1 = torch.matmul(self.D_neg_sqrt, torch.matmul(self.A_hat, self.D_neg_sqrt))
        
        # (D^-1/2 * A_hat * D^-1/2) * (X * W)
        support_2 = torch.matmul(support_1, torch.matmul(X, self.W))
        
        # ReLU(D^-1/2 * A_hat * D^-1/2 * X * W)
        H = F.relu(support_2)

        return H
    
if __name__ == "__main__":

    # Example Usage
    input_dim = 3  # Assuming the input dimension is 3
    output_dim = 2  # Assuming the output dimension is 2

    # Example adjacency matrix
    A = torch.tensor([[1., 0., 0.],
                      [0., 1., 1.],
                      [0., 1., 1.]])  

    # Create the GCN Layer
    gcn_layer = GCNLayer(input_dim, output_dim, A)

    # Example input feature matrix
    X = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

    # Forward pass
    output = gcn_layer(X)
    
    print(output)
    # tensor([[ 3.8823,  4.0625],
    #     [10.5007, 10.0951],
    #     [12.3979, 11.8572]], grad_fn=<ReluBackward0>)

