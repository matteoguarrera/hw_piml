import torch

class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)

def rotate_3d_coordinates(coordinates, x_degrees = 30, y_degrees = 30, z_degrees = 30):
    """
    Rotates a set of 3D coordinates around the x, y, and z axes by specified angles.
    
    Args:
    coordinates (torch.Tensor): An N x 3 array of 3D coordinates.
    x_degrees (float): The rotation angle around the x-axis in degrees.
    y_degrees (float): The rotation angle around the y-axis in degrees.
    z_degrees (float): The rotation angle around the z-axis in degrees.
    
    Returns:
    torch.Tensor: The rotated coordinates: An N x 3 array of 3D coordinates.
    """
    device = coordinates.device
    with torch.no_grad():    
        # Convert degrees to radians
        x_radians = torch.deg2rad(torch.Tensor([x_degrees]))
        y_radians = torch.deg2rad(torch.Tensor([y_degrees]))
        z_radians = torch.deg2rad(torch.Tensor([z_degrees]))

        
        # Rotation matrices for each axis
        Rx = torch.tensor([[1, 0, 0],
                    [0, torch.cos(x_radians), -torch.sin(x_radians)],
                    [0, torch.sin(x_radians), torch.cos(x_radians)]], device=device)

        Ry = torch.tensor([[torch.cos(y_radians), 0, torch.sin(y_radians)],
                        [0, 1, 0],
                        [-torch.sin(y_radians), 0, torch.cos(y_radians)]], device=device)

        Rz = torch.tensor([[torch.cos(z_radians), -torch.sin(z_radians), 0],
                        [torch.sin(z_radians), torch.cos(z_radians), 0],
                        [0, 0, 1]], device=device)

        # Combined rotation matrix
        R = torch.mm(Rz, torch.mm(Ry, Rx)).to(device)
        # Apply the rotation
        return torch.mm(coordinates, R.T)
