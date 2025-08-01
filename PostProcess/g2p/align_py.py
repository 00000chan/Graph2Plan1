import numpy as np
# We will likely need shapely for more complex geometric operations later
# from shapely.geometry import Polygon

def get_entrance_space_py(door_seg, door_ori, threshold):
    """
    Python port of get_entrance_space.m
    Creates a bounding box for the entrance area.

    Args:
        door_seg (np.ndarray): Shape (2, 2), coordinates of the two door points.
        door_ori (int): Orientation of the door (0: right, 1: down, 2: left, 3: up).
        threshold (float): How far to extend the entrance box.

    Returns:
        np.ndarray: Shape (4,), the entrance bounding box [x0, y0, x1, y1].
    """
    # Create a bounding box [x0, y0, x1, y1] from the two points
    x_coords = door_seg[:, 0]
    y_coords = door_seg[:, 1]
    door_box = np.array([
        np.min(x_coords),
        np.min(y_coords),
        np.max(x_coords),
        np.max(y_coords)
    ])

    if door_ori == 0:  # Facing right, extend in +y direction
        door_box[3] = door_box[3] + threshold
    elif door_ori == 1:  # Facing down, extend in -x direction
        door_box[0] = door_box[0] - threshold
    elif door_ori == 2:  # Facing left, extend in -y direction
        door_box[1] = door_box[1] - threshold
    elif door_ori == 3:  # Facing up, extend in +x direction
        door_box[2] = door_box[2] + threshold

    return door_box


def align_fp_py(boundary, r_box, r_type, r_edge, fp_layout, threshold, draw_result=False):
    """
    Python port of the main align_fp.m script.
    This function will orchestrate the alignment process.
    """
    print("Running Python-native alignment...")

    # The MATLAB script first calls get_entrance_space. We do the same.
    # Note: boundary format is (x, y, dir, isNew). Door is the first two points.
    door_seg = boundary[:2, :2]
    door_ori = boundary[0, 2]
    entrance_box = get_entrance_space_py(door_seg, door_ori, threshold)

    print("Calculated entrance box:", entrance_box)

    # --- TODO: Port the rest of the align_fp.m pipeline ---
    # 1. align_with_boundary_py
    # 2. align_neighbor_py
    # 3. regularize_fp_py
    # 4. get_room_boundary_py

    # For now, return placeholder values
    num_rooms = r_box.shape[0]
    new_box = r_box.copy() # Return original boxes for now
    order = np.arange(num_rooms)
    r_boundary = np.array([b for b in new_box], dtype=object) # Placeholder

    print("Placeholder: Returning un-aligned boxes.")
    return new_box, order, r_boundary
