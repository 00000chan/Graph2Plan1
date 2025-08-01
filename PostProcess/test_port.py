import scipy.io as sio
import numpy as np
import os

from app import App
from g2p.plot import plot_fp
from g2p.align_py import align_fp_py # UNCOMMENTED

def main():
    # --- Setup ---
    print("Setting up test harness...")
    model_path = '../Interface/model/model.pth'
    device = 'cpu'  # Use CPU to avoid GPU dependencies for this test
    dataset_path = '../Network/data/data_test.mat'
    output_dir = './port_test_output'
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data ---
    print("Loading test data...")
    app = App(model_path, device)
    dataset = sio.loadmat(dataset_path, squeeze_me=True, struct_as_record=False)['data']
    # Use a specific data point for consistent testing
    data_test = dataset[0]

    # --- Run Original MATLAB Implementation ---
    print("Running original MATLAB implementation to get ground truth...")

    # 1. Get initial network prediction
    data_from_net = app.forward(data_test, network_data=True)

    # 2. Run MATLAB alignment
    # We clone the data object to keep the pre-alignment state
    data_aligned_matlab = data_from_net.clone()
    data_aligned_matlab = app.align(data_aligned_matlab)

    # 3. Save the ground truth results
    ground_truth_path = os.path.join(output_dir, 'ground_truth_matlab.npz')
    print(f"Saving MATLAB ground truth to {ground_truth_path}")
    np.savez(
        ground_truth_path,
        newBox=data_aligned_matlab.newBox,
        order=data_aligned_matlab.order,
        rBoundary=data_aligned_matlab.rBoundary
    )

    # --- Run Python Implementation ---
    print("\n--- Running Python implementation ---")

    # Note: The original code uses a different threshold for alignment
    # We use the same one for a fair comparison.
    from g2p.align import REFINE_ThRESHOLD

    boxes_py, order_py, rboundary_py = align_fp_py(
        data_from_net.boundary,
        data_from_net.boxes,
        data_from_net.rType,
        data_from_net.rEdge,
        data_from_net.layout,
        threshold=REFINE_ThRESHOLD,
    )

    python_output_path = os.path.join(output_dir, 'output_python.npz')
    print(f"Saving Python output to {python_output_path}")
    np.savez(
        python_output_path,
        newBox=boxes_py,
        order=order_py,
        rBoundary=rboundary_py
    )

    # --- Comparison ---
    # print("\n--- Comparison (to be enabled later) ---")
    # ground_truth = np.load(ground_truth_path, allow_pickle=True)
    # print("Comparing box alignment...")
    # assert np.allclose(ground_truth['newBox'], boxes_py)
    # print("Comparing room order...")
    # assert np.array_equal(ground_truth['order'], order_py)

    print("\nTest harness setup complete. Ground truth saved.")


if __name__ == '__main__':
    main()
