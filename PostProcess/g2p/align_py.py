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


def align_adjacent_room3_py(box, temp_box, updated, type, threshold):
    """
    Python port of align_adjacent_room3.m.
    Aligns two adjacent rooms based on their relationship type.
    """
    new_box = box.copy()

    # This nested function mimics the behavior of the MATLAB `align` sub-function.
    def align(idx1, idx2, align_threshold, attach=False):
        # idx1 and idx2 are tuples like (box_index, side_index), e.g., (0, 0) for box 0, left side
        # In MATLAB, this was 1-based, here it's 0-based.
        box1_idx, side1_idx = idx1
        box2_idx, side2_idx = idx2

        # Check if the distance is within the threshold
        if abs(temp_box[box1_idx, side1_idx] - temp_box[box2_idx, side2_idx]) <= align_threshold:
            # If one box was updated and the other wasn't, snap the non-updated to the updated.
            if updated[box1_idx, side1_idx] and not updated[box2_idx, side2_idx]:
                new_box[box2_idx, side2_idx] = new_box[box1_idx, side1_idx]
            elif updated[box2_idx, side2_idx] and not updated[box1_idx, side1_idx]:
                new_box[box1_idx, side1_idx] = new_box[box2_idx, side2_idx]
            # If neither has been updated, move both halfway.
            elif not updated[box1_idx, side1_idx] and not updated[box2_idx, side2_idx]:
                if attach:
                    new_box[box2_idx, side2_idx] = new_box[box1_idx, side1_idx]
                else:
                    y = (new_box[box1_idx, side1_idx] + new_box[box2_idx, side2_idx]) / 2.0
                    new_box[box1_idx, side1_idx] = y
                    new_box[box2_idx, side2_idx] = y
            return True
        return False

    # Relationship type mapping:
    # 0:left-above, 1:left-below, 2:left-of, 3:above, 4:inside,
    # 5:surrounding, 6:below, 7:right-of, 8:right-above, 9:right-below

    # Note: MATLAB's logic for diagonal cases (0,1,8,9) is complex and appears to have
    # further sub-logic in alignV/alignH. For now, we replicate the main cases.

    if type == 2: # left-of (box1 is left of box2)
        # align box1's right edge (2) with box2's left edge (0)
        align((0, 2), (1, 0), threshold)
        # align tops and bottoms within a smaller threshold
        align((0, 1), (1, 1), threshold / 2)
        align((0, 3), (1, 3), threshold / 2)
    elif type == 3: # above (box1 is above box2)
        # align box1's bottom edge (3) with box2's top edge (1)
        align((0, 3), (1, 1), threshold)
        # align lefts and rights
        align((0, 0), (1, 0), threshold / 2)
        align((0, 2), (1, 2), threshold / 2)
    elif type == 4: # inside (box1 is inside box2)
        align((1,0), (0,0), threshold, attach=True)
        align((1,1), (0,1), threshold, attach=True)
        align((1,2), (0,2), threshold, attach=True)
        align((1,3), (0,3), threshold, attach=True)
    elif type == 6: # below (box1 is below box2)
        # align box1's top edge (1) with box2's bottom edge (3)
        align((0, 1), (1, 3), threshold)
        align((0, 0), (1, 0), threshold / 2)
        align((0, 2), (1, 2), threshold / 2)
    elif type == 7: # right-of (box1 is right of box2)
        # align box1's left edge (0) with box2's right edge (2)
        align((0, 0), (1, 2), threshold)
        align((0, 1), (1, 1), threshold / 2)
        align((0, 3), (1, 3), threshold / 2)

    # The diagonal cases (0, 1, 8, 9) and surrounding (5) are more complex
    # and are omitted in this simplified first-pass port. They can be added later.

    return new_box


def align_neighbor_py(box, r_edge, updated, threshold):
    """
    Python port of align_neighbor.m
    """
    temp_box = box.copy()
    checked = np.zeros(r_edge.shape[0], dtype=bool)

    # This helper calculates how many sides of a room pair have been updated
    def get_updated_count(upd, edge_list):
        counts = np.zeros(edge_list.shape[0])
        for k, edge in enumerate(edge_list):
            indices = edge[:2] # In Python, indices are already 0-based
            counts[k] = np.sum(upd[indices, :])
        return counts

    updated_count = get_updated_count(updated, r_edge)

    for i in range(r_edge.shape[0]):
        # Find the unchecked edge with the most updated sides
        unchecked_indices = np.where(~checked)[0]
        if len(unchecked_indices) == 0:
            break

        best_local_idx = np.argmax(updated_count[unchecked_indices])
        edge_idx = unchecked_indices[best_local_idx]
        checked[edge_idx] = True

        current_edge = r_edge[edge_idx]
        room_indices = current_edge[:2]
        rel_type = current_edge[2]

        # Call the alignment function for the pair of rooms
        b = align_adjacent_room3_py(
            box[room_indices, :],
            temp_box[room_indices, :],
            updated[room_indices, :],
            rel_type,
            threshold + 6 # MATLAB code adds 6 to the threshold here
        )

        # Update the main box array with the aligned results
        box[room_indices, :] = b

        # Recalculate updated counts for the next iteration
        updated_count = get_updated_count(updated, r_edge)

    return box, updated


import networkx as nx

def find_room_order_py(adj_matrix):
    """
    Python port of find_room_order.m using networkx.
    Performs a topological sort on the room overlap graph to find a valid drawing order.
    """
    try:
        # Create a directed graph from the adjacency matrix
        graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        # Perform topological sort
        # This will raise an error if the graph has a cycle
        order = list(nx.topological_sort(graph))
        return np.array(order)

    except nx.NetworkXUnfeasible:
        # This block handles cases where a cycle is detected (which shouldn't happen with area-based ordering)
        # but mimics the MATLAB code's fallback of just returning a partial or incorrect order.
        print("Warning: Cycle detected in room overlap graph. Resulting order may be incorrect.")
        # Fallback similar to MATLAB's simple cycle breaking
        graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        order = []
        while graph.number_of_nodes() > 0:
            in_degrees = dict(graph.in_degree())
            # Find nodes with in-degree 0
            zero_in_degree_nodes = [node for node, degree in in_degrees.items() if degree == 0]

            # If no such node, break cycle by picking node with lowest in-degree > 0
            if not zero_in_degree_nodes:
                min_degree = min(d for d in in_degrees.values() if d > 0)
                zero_in_degree_nodes = [node for node, degree in in_degrees.items() if degree == min_degree]

            # Add these nodes to the order and remove from graph
            node_to_remove = zero_in_degree_nodes[0]
            order.append(node_to_remove)
            graph.remove_node(node_to_remove)
        return np.array(order)


def regularize_fp_py(box, boundary, r_type):
    """
    Python port of regularize_fp.m
    Crops rooms to the boundary, determines drawing order, and fills gaps.
    """
    from shapely.geometry import Polygon, box as shapely_box, MultiPolygon
    from shapely.ops import unary_union

    # 1. Crop each room box to the building boundary
    is_new = boundary[:, 3]
    poly_boundary = Polygon(boundary[is_new == 0, :2])

    for i in range(box.shape[0]):
        poly_room = shapely_box(box[i, 0], box[i, 1], box[i, 2], box[i, 3])
        intersection = poly_boundary.intersection(poly_room)
        if not intersection.is_empty:
            box[i, :] = np.array(intersection.bounds)
        else:
            print(f"Warning: Room {i} is completely outside the building boundary.")

    # 2. Determine drawing order based on overlaps
    num_rooms = box.shape[0]
    order_m = np.zeros((num_rooms, num_rooms), dtype=bool)
    room_polys = [shapely_box(b[0], b[1], b[2], b[3]) for b in box]

    for i in range(num_rooms):
        for j in range(i + 1, num_rooms):
            if room_polys[i].intersects(room_polys[j]):
                if room_polys[i].area <= room_polys[j].area:
                    order_m[i, j] = True
                else:
                    order_m[j, i] = True

    order = find_room_order_py(order_m)
    # MATLAB version reverses the order, let's replicate that
    order = order[::-1]

    # 3. Fill gaps
    living_idx = np.where(r_type == 0)[0]
    if len(living_idx) == 0: return box, order # No living room, can't fill gaps
    living_idx = living_idx[0]

    # Subtract all non-living room areas from the main boundary
    other_rooms_union = unary_union([room_polys[i] for i in range(num_rooms) if i != living_idx])
    gaps = poly_boundary.difference(other_rooms_union)

    if isinstance(gaps, Polygon):
        # Only one gap, assume it's the living room
        box[living_idx, :] = np.array(gaps.bounds)
    elif isinstance(gaps, MultiPolygon):
        living_poly = room_polys[living_idx]

        # Find the gap that has the biggest overlap with the original living room
        overlap_areas = np.array([gap.intersection(living_poly).area for gap in gaps.geoms])
        living_gap_idx = np.argmax(overlap_areas)

        for k, gap_poly in enumerate(gaps.geoms):
            if k == living_gap_idx:
                box[living_idx, :] = np.array(gap_poly.bounds)
            else:
                # For other gaps, find the closest room and expand it
                gap_center = np.array(gap_poly.centroid.coords)
                box_centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in box])
                distances = np.linalg.norm(box_centers - gap_center, axis=1)
                closest_room_idx = np.argmin(distances)

                # Expand the closest room to include the gap
                expanded_poly = unary_union([room_polys[closest_room_idx], gap_poly])
                box[closest_room_idx, :] = np.array(expanded_poly.bounds)
                room_polys[closest_room_idx] = expanded_poly # Update for next iteration

    return box, order


def align_fp_py(boundary, r_box, r_type, r_edge, fp_layout, threshold, draw_result=False):
    """
    Python port of the main align_fp.m script.
    This function will orchestrate the alignment process.
    """
    print("Running Python-native alignment...")

    # The MATLAB code re-orders edges to process living room last. We replicate this.
    living_idx = np.where(r_type == 0)[0]
    if len(living_idx) > 0:
        living_idx = living_idx[0]
        is_living_edge = (r_edge[:, 0] == living_idx) | (r_edge[:, 1] == living_idx)
        r_edge = np.vstack([r_edge[~is_living_edge], r_edge[is_living_edge]])

    # 1. Align with boundary
    new_box, updated = align_with_boundary_py(r_box.copy(), boundary, threshold, r_type)
    print("Step 1: align_with_boundary_py complete.")

    # 2. Align with neighbors
    new_box, updated = align_neighbor_py(new_box, r_edge, updated, threshold)
    print("Step 2: align_neighbor_py complete.")

    # 3. Regularize floorplan
    new_box, order = regularize_fp_py(new_box, boundary, r_type)
    print("Step 3: regularize_fp_py complete.")

    # --- TODO: Port the rest of the align_fp.m pipeline ---
    # 4. get_room_boundary_py

    # For now, return placeholder values
    num_rooms = new_box.shape[0]
    r_boundary = np.array([b for b in new_box], dtype=object) # Placeholder

    print("Placeholder: Returning regularized boxes.")
    return new_box, order, r_boundary
