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


def find_close_seg_py(box, boundary):
    """
    Python port of find_close_seg.m
    Finds the closest horizontal and vertical boundary segments for a given box.

    Args:
        box (np.ndarray): Shape (4,), the room box [x0, y0, x1, y1].
        boundary (np.ndarray): Shape (N, 4), the building boundary segments.

    Returns:
        tuple: (closedSeg, distSeg)
            - closedSeg (np.ndarray): Shape (4,). The coords of the closest wall for each of the 4 box sides.
            - distSeg (np.ndarray): Shape (4,). The distances to the closest wall for each of the 4 box sides.
    """
    is_new = boundary[:, 3]
    boundary = boundary[is_new == 0] # Filter for main boundary walls

    # Create boundary segments [x_start, y_start, x_end, y_end, orientation]
    b_seg = np.hstack([boundary[:, :2], np.roll(boundary, -1, axis=0)[:, :2], boundary[:, 2:3]])

    v_seg = b_seg[b_seg[:, 4] % 2 == 1]
    h_seg = b_seg[b_seg[:, 4] % 2 == 0]

    # Ensure consistent direction for segments
    v_seg[v_seg[:, 4] == 3, [1, 3]] = v_seg[v_seg[:, 4] == 3, [3, 1]] # Sort by y
    h_seg[h_seg[:, 4] == 2, [0, 2]] = h_seg[h_seg[:, 4] == 2, [2, 0]] # Sort by x

    # Sort segments by their primary axis
    v_seg = v_seg[np.argsort(v_seg[:, 0])]
    h_seg = h_seg[np.argsort(h_seg[:, 2])]

    closed_seg = np.ones(4) * 256.0
    dist_seg = np.ones(4) * 256.0

    # Check vertical segments (for left and right alignment)
    for seg in v_seg:
        v_dist = 0
        if seg[3] <= box[1]: # seg is below box
            v_dist = box[1] - seg[3]
        elif seg[1] >= box[3]: # seg is above box
            v_dist = seg[1] - box[3]

        h_dist_left = box[0] - seg[0]
        h_dist_right = box[2] - seg[0]

        dist1 = np.linalg.norm([h_dist_left, v_dist])
        dist3 = np.linalg.norm([h_dist_right, v_dist])

        if dist1 < dist_seg[0] and h_dist_left >= 0:
            dist_seg[0] = dist1
            closed_seg[0] = seg[0]
        if dist3 < dist_seg[2] and h_dist_right <= 0:
            dist_seg[2] = dist3
            closed_seg[2] = seg[0]

    # Check horizontal segments (for top and bottom alignment)
    for seg in h_seg:
        h_dist = 0
        if seg[2] <= box[0]: # seg is left of box
            h_dist = box[0] - seg[2]
        elif seg[0] >= box[2]: # seg is right of box
            h_dist = seg[0] - box[2]

        v_dist_top = box[1] - seg[1]
        v_dist_bottom = box[3] - seg[1]

        dist2 = np.linalg.norm([v_dist_top, h_dist])
        dist4 = np.linalg.norm([v_dist_bottom, h_dist])

        if dist2 < dist_seg[1] and v_dist_top >= 0:
            dist_seg[1] = dist2
            closed_seg[1] = seg[1]
        if dist4 < dist_seg[3] and v_dist_bottom <= 0:
            dist_seg[3] = dist4
            closed_seg[3] = seg[1]

    return closed_seg, dist_seg


def shrink_box_py(room_box, entrance_box):
    """
    Python port of shrink_box.m.
    Shrinks a room box if it overlaps with the entrance area.
    This implementation uses shapely for robust geometric operations.

    Args:
        room_box (np.ndarray): The bounding box of the room [x0, y0, x1, y1].
        entrance_box (np.ndarray): The bounding box of the entrance area.

    Returns:
        np.ndarray: The new, shrunken bounding box for the room.
    """
    from shapely.geometry import Polygon, box

    room_poly = box(room_box[0], room_box[1], room_box[2], room_box[3])
    entrance_poly = box(entrance_box[0], entrance_box[1], entrance_box[2], entrance_box[3])

    # The core logic is to subtract the entrance area from the room
    shrunken_poly = room_poly.difference(entrance_poly)

    # Return the bounding box of the resulting (potentially multi-part) polygon
    return np.array(shrunken_poly.bounds)


def align_with_boundary_py(box, boundary, threshold, r_type):
    """
    Python port of align_with_boundary.m
    """
    from shapely.geometry import box as shapely_box

    temp_box = box.copy()
    num_boxes = box.shape[0]
    updated = np.zeros_like(box, dtype=bool)

    # 1. Find and snap to closest boundary segments
    closed_segs = np.zeros_like(box)
    dist_segs = np.zeros_like(box)
    for i in range(num_boxes):
        closed_segs[i], dist_segs[i] = find_close_seg_py(box[i], boundary)

    mask = dist_segs <= threshold
    box[mask] = closed_segs[mask]
    updated[mask] = True

    # 2. Check for and resolve overlaps with the entrance area
    entrance_box = get_entrance_space_py(boundary[:2, :2], boundary[0, 2], threshold)
    entrance_poly = shapely_box(entrance_box[0], entrance_box[1], entrance_box[2], entrance_box[3])

    for i in range(num_boxes):
        # LivingRoom (0) and Entrance (10) are exempt from shrinking
        if r_type[i] not in [0, 10]:
            room_poly = shapely_box(box[i, 0], box[i, 1], box[i, 2], box[i, 3])
            if entrance_poly.intersects(room_poly):
                box[i, :] = shrink_box_py(box[i, :], entrance_box)
                # Check which specific sides of the box were changed
                updated[i, box[i, :] != temp_box[i, :]] = True
                updated[i, box[i, :] == temp_box[i, :]] = False

    return box, updated


def align_fp_py(boundary, r_box, r_type, r_edge, fp_layout, threshold, draw_result=False):
    """
    Python port of the main align_fp.m script.
    This function will orchestrate the alignment process.
    """
    print("Running Python-native alignment...")

    # 1. Align with boundary
    new_box, updated = align_with_boundary_py(r_box.copy(), boundary, threshold, r_type)
    print("Step 1: align_with_boundary_py complete.")

    # --- TODO: Port the rest of the align_fp.m pipeline ---
    # 2. align_neighbor_py
    # 3. regularize_fp_py
    # 4. get_room_boundary_py

    # For now, return placeholder values
    num_rooms = new_box.shape[0]
    order = np.arange(num_rooms)
    r_boundary = np.array([b for b in new_box], dtype=object) # Placeholder

    print("Placeholder: Returning boundary-aligned boxes.")
    return new_box, order, r_boundary
