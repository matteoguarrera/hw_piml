import numpy as np

def rotate_3d_coordinates(coordinates, x_degrees, y_degrees, z_degrees):
    """
    Rotates a set of 3D coordinates around the x, y, and z axes by specified angles.
    
    Args:
    coordinates (np.array): An N x 3 array of 3D coordinates.
    x_degrees (float): The rotation angle around the x-axis in degrees.
    y_degrees (float): The rotation angle around the y-axis in degrees.
    z_degrees (float): The rotation angle around the z-axis in degrees.
    
    Returns:
    np.array: The rotated coordinates: An N x 3 array of 3D coordinates.
    """
    #TODO: write this function
    # Rotation matrix for the x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_degrees), -np.sin(x_degrees)],
        [0, np.sin(x_degrees), np.cos(x_degrees)]
    ])

    # Rotation matrix for the y-axis
    Ry = np.array([
        [np.cos(y_degrees), 0, np.sin(y_degrees)],
        [0, 1, 0],
        [-np.sin(y_degrees), 0, np.cos(y_degrees)]
    ])

    # Rotation matrix for the z-axis
    Rz = np.array([
        [np.cos(z_degrees), -np.sin(z_degrees), 0],
        [np.sin(z_degrees), np.cos(z_degrees), 0],
        [0, 0, 1]
    ])

    R = Rx @ Ry @ Rz  # convention used everywhere.

    return np.dot(coordinates, R)
