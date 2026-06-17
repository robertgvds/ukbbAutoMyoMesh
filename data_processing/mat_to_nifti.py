"""
MATLAB to NIfTI Converter for Cardiac Segmentation.

This script extracts image volumes and segmentation masks from MATLAB (.mat) files.
It applies geometric corrections (rotation and flipping) to fix MATLAB's native 
orientation and injects the Affine matrix from a reference NIfTI image to ensure 
correct spatial alignment, which is required by Convolutional Neural Networks (CNNs).

Usage:
    python mat_to_nifti.py -i <input_dir> -o <output_dir> -r <reference_nifti>
"""

import os
import glob
import argparse
import scipy.io as sio
import numpy as np
import nibabel as nib
from skimage.draw import polygon
from typing import Tuple, Any, Optional

# =====================================================================
# DEFAULT CONFIGURATIONS & CONSTANTS
# =====================================================================
DEFAULT_INPUT_DIR = "input/mats_originais" 
DEFAULT_OUTPUT_DIR = "input/niftis_extraidos"

# Path to a reference image to inherit real world coordinates (Affine)
DEFAULT_REFERENCE_NIFTI = "input/patient/patient.nii.gz"

# Anatomical Segmentation Labels
LABEL_ENDO_LV = 1  # Left Ventricle Endocardium
LABEL_EPI_LV = 2   # Left Ventricle Epicardium
LABEL_ENDO_RV = 3  # Right Ventricle Endocardium
LABEL_SCAR = 1     # Scar / Fibrosis

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract and spatially correct .mat files to NIfTI.")
    parser.add_argument("-i", "--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="Path to the directory containing the original .mat files.")
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Path to the directory where the .nii.gz files will be saved.")
    parser.add_argument("-r", "--reference", type=str, default=DEFAULT_REFERENCE_NIFTI,
                        help="Path to a valid NIfTI file used to inherit the Affine matrix and Header.")
    return parser.parse_args()

def fix_matlab_orientation(array_3d_or_4d: np.ndarray) -> np.ndarray:
    """
    Corrects the inverted orientation from MATLAB by rotating and flipping 
    both the image and the masks exactly the same way before saving.
    
    Args:
        array_3d_or_4d (np.ndarray): The raw 3D or 4D array extracted from MATLAB.
        
    Returns:
        np.ndarray: The geometrically corrected array.
    """
    # Rotate 90 degrees in the axial plane (axes 0 and 1)
    fixed_array = np.rot90(array_3d_or_4d, k=-1, axes=(0, 1))
    
    # Vertical flip to match the target reference orientation
    fixed_array = np.flip(fixed_array, axis=1)
    
    return fixed_array

def get_coordinates(setstruct: Any, attr_name: str, z: int, t: int) -> Optional[np.ndarray]:
    """
    Extracts X or Y coordinates from a MATLAB annotation structure.
    
    Args:
        setstruct (Any): The 'setstruct' object loaded from the .mat file.
        attr_name (str): The name of the attribute to extract (e.g., 'EndoX', 'EpiY').
        z (int): Slice index.
        t (int): Time frame index.
        
    Returns:
        Optional[np.ndarray]: An array of coordinates, or None if not found.
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
    Draws a filled polygon on the segmentation mask based on the provided coordinates.
    
    Args:
        mask (np.ndarray): The mask array where the polygon will be drawn.
        x_coords (np.ndarray): X-axis coordinates.
        y_coords (np.ndarray): Y-axis coordinates.
        label (int): The anatomical class value to fill.
        slice_shape (Tuple[int, int]): The 2D dimensions of the slice (X, Y).
        mask_idx (tuple): The specific spatial/temporal slice index.
    """
    if x_coords is not None and y_coords is not None:
        valid_points = ~np.isnan(x_coords)
        if np.sum(valid_points) > 3:  # At least 3 points are required to form a polygon
            cx = np.round(x_coords[valid_points]).astype(int) - 1
            cy = np.round(y_coords[valid_points]).astype(int) - 1
            rr, cc = polygon(cx, cy, shape=slice_shape)
            mask[mask_idx][rr, cc] = label

def process_mat_files(input_dir: str, output_dir: str, reference_nifti_path: str) -> None:
    """
    Orchestrates the reading of the input directory and processes each .mat file.
    
    Args:
        input_dir (str): Directory containing the source .mat files.
        output_dir (str): Directory where the NIfTI files will be saved.
        reference_nifti_path (str): Path to the NIfTI file used for spatial reference.
    """
    os.makedirs(output_dir, exist_ok=True)
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))
    
    if not mat_files:
        print(f"Warning: No .mat files found in {input_dir}")
        return

    # Load the Affine matrix from the reference image
    if os.path.exists(reference_nifti_path):
        ref_nifti = nib.load(reference_nifti_path)
        ref_affine = ref_nifti.affine
        ref_header = ref_nifti.header
        print(f"Using Affine matrix from reference: {reference_nifti_path}")
    else:
        print("WARNING: Reference image not found. Using generic fallback (np.eye).")
        ref_affine = np.eye(4)
        ref_header = nib.Nifti1Header()

    for mat_filepath in mat_files:
        patient_name = os.path.basename(mat_filepath).replace('.mat', '')
        print(f"\nExtracting patient: {patient_name}")
        
        try:
            mat_data = sio.loadmat(mat_filepath, squeeze_me=True, struct_as_record=False)
            
            # 1. Find the original raw image (before rotation)
            raw_image = None
            if 'im' in mat_data and hasattr(mat_data['im'], 'shape') and mat_data['im'].size > 0:
                raw_image = mat_data['im']
            elif 'setstruct' in mat_data and hasattr(mat_data['setstruct'], 'IM') and mat_data['setstruct'].IM.size > 0:
                raw_image = mat_data['setstruct'].IM

            if raw_image is None:
                print("Error: RAW Image not found in this .mat file. Skipping patient.")
                continue

            original_slice_shape = raw_image.shape[:2]
            
            # 2. Generate masks using original coordinates and dimensions
            anat_seg = np.zeros_like(raw_image, dtype=np.uint8)
            scar_seg = np.zeros_like(raw_image, dtype=np.uint8)
            
            if 'setstruct' in mat_data:
                setstruct = mat_data['setstruct']
                if hasattr(setstruct, 'EndoX'):
                    coords_shape = setstruct.EndoX.shape
                    num_slices = coords_shape[1] if len(coords_shape) > 1 else 1
                    num_frames = coords_shape[2] if len(coords_shape) > 2 else 1

                    for t in range(num_frames):
                        for z in range(num_slices):
                            mask_idx = (slice(None), slice(None), z, t) if num_frames > 1 else (slice(None), slice(None), z)
                            
                            draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'EpiX', z, t), get_coordinates(setstruct, 'EpiY', z, t), LABEL_EPI_LV, original_slice_shape, mask_idx)
                            draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'EndoX', z, t), get_coordinates(setstruct, 'EndoY', z, t), LABEL_ENDO_LV, original_slice_shape, mask_idx)
                            draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'RVEndoX', z, t), get_coordinates(setstruct, 'RVEndoY', z, t), LABEL_ENDO_RV, original_slice_shape, mask_idx)

                    if hasattr(setstruct, 'Roi'):
                        rois = setstruct.Roi
                        if not isinstance(rois, np.ndarray): 
                            rois = [rois]
                        for roi in rois:
                            try:
                                z_roi = int(roi.Z) - 1 if hasattr(roi, 'Z') else 0
                                t_roi = int(roi.T) - 1 if hasattr(roi, 'T') else 0
                                if 0 <= z_roi < num_slices and 0 <= t_roi < num_frames:
                                    cx = np.round(roi.X).astype(int) - 1
                                    cy = np.round(roi.Y).astype(int) - 1
                                    rr, cc = polygon(cx, cy, shape=original_slice_shape)
                                    idx_scar = (slice(None), slice(None), z_roi, t_roi) if num_frames > 1 else (slice(None), slice(None), z_roi)
                                    scar_seg[idx_scar][rr, cc] = LABEL_SCAR
                            except Exception: 
                                pass

            # 3. Apply geometric transformations to image and masks
            raw_image = fix_matlab_orientation(raw_image)
            anat_seg = fix_matlab_orientation(anat_seg)
            scar_seg = fix_matlab_orientation(scar_seg)

            # 4. Save NIfTI files using the reference Affine matrix
            nib.save(nib.Nifti1Image(raw_image, affine=ref_affine, header=ref_header), os.path.join(output_dir, f"{patient_name}_img.nii.gz"))
            nib.save(nib.Nifti1Image(anat_seg, affine=ref_affine, header=ref_header), os.path.join(output_dir, f"{patient_name}_seg_anat.nii.gz"))
            nib.save(nib.Nifti1Image(scar_seg, affine=ref_affine, header=ref_header), os.path.join(output_dir, f"{patient_name}_seg_scar.nii.gz"))
            
            print("   -> Success: Image and segmentations aligned and saved.")
            
        except Exception as e:
            print(f"   -> Critical failure processing {patient_name}. Error: {str(e)}")

    print("\nExtraction and conversion completed.")

if __name__ == "__main__":
    args = parse_arguments()
    process_mat_files(args.input_dir, args.output_dir, args.reference)