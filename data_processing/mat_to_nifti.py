import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nib
from skimage.draw import polygon
from typing import Tuple, Any

# DEFAULT CONFIGURATIONS AND CONSTANTS
DEFAULT_INPUT_DIR = "input/mats_originais" 
DEFAULT_OUTPUT_DIR = "output/niftis_extraidos"

# Labels for anatomical segmentations
LABEL_ENDO_LV = 1  # Left Ventricle Endocardium (Red)
LABEL_EPI_LV = 2   # Left Ventricle Epicardium (Green)
LABEL_ENDO_RV = 3  # Right Ventricle Endocardium (Magenta)
LABEL_SCAR = 1     # Scar / Fibrosis

def extract_image(mat_data: Any, patient_name: str, output_dir: str) -> Tuple[int, int]:
    """
    Searches for the raw image inside the .MAT data, saves it as NIfTI,
    and returns the slice shape (X, Y) to be used in segmentations.
    """
    raw_image = None
    slice_shape = (256, 256) # Default size if image is not found
    
    # Check where the image is stored in the .MAT structure
    if 'im' in mat_data and hasattr(mat_data['im'], 'shape') and mat_data['im'].size > 0:
        raw_image = mat_data['im']
    elif 'setstruct' in mat_data and hasattr(mat_data['setstruct'], 'IM') and mat_data['setstruct'].IM.size > 0:
        raw_image = mat_data['setstruct'].IM

    if raw_image is not None:
        slice_shape = raw_image.shape[:2]
        print(f"   -> Image found. Real shape: {raw_image.shape}")
        
        output_filepath = os.path.join(output_dir, f"{patient_name}_img.nii.gz")
        nifti_img = nib.Nifti1Image(raw_image, np.eye(4))
        nib.save(nifti_img, output_filepath)
    else:
        print("   -> Image not found. Generating segmentations only.")
        
    return slice_shape

def get_coordinates(setstruct: Any, attr_name: str, z: int, t: int) -> Any:
    """
    Helper function to extract X or Y coordinates from a structure,
    handling the differences between 3D (z) and 4D (z, t) data.
    """
    if not hasattr(setstruct, attr_name): 
        return None
        
    val = getattr(setstruct, attr_name)
    
    if len(val.shape) == 2 and t == 0: 
        return val[:, z]
    if len(val.shape) == 3: 
        return val[:, z, t]
        
    return None

def draw_mask_polygon(mask: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, 
                      label: int, slice_shape: Tuple[int, int], mask_idx: tuple) -> None:
    """
    Draws the polygon on the mask if there are enough valid coordinates.
    """
    if x_coords is not None and y_coords is not None:
        valid_points = ~np.isnan(x_coords)
        if np.sum(valid_points) > 3:
            # Subtract 1 to convert from MATLAB format (1-based) to Python format (0-based)
            cx = np.round(x_coords[valid_points]).astype(int) - 1
            cy = np.round(y_coords[valid_points]).astype(int) - 1
            
            rr, cc = polygon(cx, cy, shape=slice_shape)
            mask[mask_idx][rr, cc] = label

def extract_segmentations(mat_data: Any, patient_name: str, slice_shape: Tuple[int, int], output_dir: str) -> None:
    """
    Extracts anatomical and scar segmentations, draws the polygons
    in empty matrices, and saves them as NIfTI files.
    """
    if 'setstruct' not in mat_data:
        return
        
    setstruct = mat_data['setstruct']
    
    if hasattr(setstruct, 'EndoX'):
        coords_shape = setstruct.EndoX.shape
        num_slices = coords_shape[1] if len(coords_shape) > 1 else 1
        num_frames = coords_shape[2] if len(coords_shape) > 2 else 1
        
        print(f"   -> Detected {num_slices} Slices (Z) and {num_frames} Frames (T).")
        
        # Define the array shape depending on whether it is 3D or 4D
        if num_frames > 1:
            mask_shape = (slice_shape[0], slice_shape[1], num_slices, num_frames)
        else:
            mask_shape = (slice_shape[0], slice_shape[1], num_slices)
        
        anat_seg = np.zeros(mask_shape, dtype=np.uint8)
        scar_seg = np.zeros(mask_shape, dtype=np.uint8)

        # Loop through Time (T) and Slices (Z) dimensions
        for t in range(num_frames):
            for z in range(num_slices):
                mask_idx = (slice(None), slice(None), z, t) if num_frames > 1 else (slice(None), slice(None), z)

                # A. LV Epicardium
                epi_x = get_coordinates(setstruct, 'EpiX', z, t)
                epi_y = get_coordinates(setstruct, 'EpiY', z, t)
                draw_mask_polygon(anat_seg, epi_x, epi_y, LABEL_EPI_LV, slice_shape, mask_idx)

                # B. LV Endocardium
                endo_x = get_coordinates(setstruct, 'EndoX', z, t)
                endo_y = get_coordinates(setstruct, 'EndoY', z, t)
                draw_mask_polygon(anat_seg, endo_x, endo_y, LABEL_ENDO_LV, slice_shape, mask_idx)

                # C. RV Endocardium
                rv_x = get_coordinates(setstruct, 'RVEndoX', z, t)
                rv_y = get_coordinates(setstruct, 'RVEndoY', z, t)
                draw_mask_polygon(anat_seg, rv_x, rv_y, LABEL_ENDO_RV, slice_shape, mask_idx)

        # D. Extract Scar / Fibrosis (Roi)
        if hasattr(setstruct, 'Roi'):
            rois = setstruct.Roi
            if not isinstance(rois, np.ndarray): 
                rois = [rois]
            
            for roi in rois:
                try:
                    # Convert from 1-based (MATLAB) to 0-based (Python)
                    z_roi = int(roi.Z) - 1 if hasattr(roi, 'Z') else 0
                    t_roi = int(roi.T) - 1 if hasattr(roi, 'T') else 0
                    
                    if 0 <= z_roi < num_slices and 0 <= t_roi < num_frames:
                        cx = np.round(roi.X).astype(int) - 1
                        cy = np.round(roi.Y).astype(int) - 1
                        rr, cc = polygon(cx, cy, shape=slice_shape)
                        
                        idx_scar = (slice(None), slice(None), z_roi, t_roi) if num_frames > 1 else (slice(None), slice(None), z_roi)
                        scar_seg[idx_scar][rr, cc] = LABEL_SCAR
                except Exception:
                    pass

        # Save the processed files
        nib.save(nib.Nifti1Image(anat_seg, np.eye(4)), os.path.join(output_dir, f"{patient_name}_seg_anat.nii.gz"))
        nib.save(nib.Nifti1Image(scar_seg, np.eye(4)), os.path.join(output_dir, f"{patient_name}_seg_scar.nii.gz"))
        print("   -> Segmentations saved! Anatomy and scar correctly positioned.")

def process_mat_files(input_dir: str, output_dir: str) -> None:
    """
    Main function that orchestrates reading the folder and processing each file.
    """
    print("=" * 51)
    print("Converting .MAT (4D, Multi-Class and Scar)")
    print("=" * 51)

    os.makedirs(output_dir, exist_ok=True)
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))

    for mat_filepath in mat_files:
        patient_name = os.path.basename(mat_filepath).replace('.mat', '')
        print(f"\nAnalyzing: {patient_name}")

        mat_data = sio.loadmat(mat_filepath, squeeze_me=True, struct_as_record=False)
        
        # 1. Extract the image and find the slice shape
        slice_shape = extract_image(mat_data, patient_name, output_dir)
        
        # 2. Use the slice shape to create and extract the segmentations
        extract_segmentations(mat_data, patient_name, slice_shape, output_dir)

    print("\n" + "=" * 88)
    print("Process finished successfully!")
    print("=" * 88)

if __name__ == "__main__":
    process_mat_files(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)