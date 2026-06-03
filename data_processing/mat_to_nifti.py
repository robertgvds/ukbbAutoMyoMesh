import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nib
from skimage.draw import polygon
from typing import Tuple, Any

# =====================================================================
# ⚙️ CONFIGURAÇÕES
# =====================================================================
DEFAULT_INPUT_DIR = "input/mats_originais" 
DEFAULT_OUTPUT_DIR = "input/niftis_extraidos"

# ⚠️ CAMINHO DA IMAGEM "BOA" DO UKBB (Para roubarmos as coordenadas reais)
REFERENCE_GOOD_NIFTI = "input/patient/patient.nii.gz"

LABEL_ENDO_LV = 1
LABEL_EPI_LV = 2
LABEL_ENDO_RV = 3
LABEL_SCAR = 1

def fix_matlab_orientation(array_3d_or_4d: np.ndarray) -> np.ndarray:
    """
    Corrige a orientação invertida do MATLAB rotacionando tudo (Imagem e Máscaras)
    exatamente da mesma forma, no último instante antes de salvar.
    """
    # Rotaciona 90 graus. Mantenha o k=-1 que funcionou para a sua imagem!
    fixed_array = np.rot90(array_3d_or_4d, k=-1, axes=(0, 1))
    
    # flip vertical
    fixed_array = np.flip(fixed_array, axis=1)
    
    return fixed_array

def get_coordinates(setstruct: Any, attr_name: str, z: int, t: int) -> Any:
    if not hasattr(setstruct, attr_name): return None
    val = getattr(setstruct, attr_name)
    if len(val.shape) == 2 and t == 0: return val[:, z]
    if len(val.shape) == 3: return val[:, z, t]
    return None

def draw_mask_polygon(mask: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, 
                      label: int, slice_shape: Tuple[int, int], mask_idx: tuple) -> None:
    if x_coords is not None and y_coords is not None:
        valid_points = ~np.isnan(x_coords)
        if np.sum(valid_points) > 3:
            cx = np.round(x_coords[valid_points]).astype(int) - 1
            cy = np.round(y_coords[valid_points]).astype(int) - 1
            rr, cc = polygon(cx, cy, shape=slice_shape)
            mask[mask_idx][rr, cc] = label

def process_mat_files(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))

    # Carrega a Affine da imagem boa de referência uma única vez
    if os.path.exists(REFERENCE_GOOD_NIFTI):
        ref_nifti = nib.load(REFERENCE_GOOD_NIFTI)
        ref_affine = ref_nifti.affine
        ref_header = ref_nifti.header
        print(f"✅ Usando Matriz Affine da referência: {REFERENCE_GOOD_NIFTI}")
    else:
        print("⚠️ AVISO: Imagem de referência não encontrada! Usando fallback (np.eye).")
        ref_affine = np.eye(4)
        ref_header = nib.Nifti1Header()

    for mat_filepath in mat_files:
        patient_name = os.path.basename(mat_filepath).replace('.mat', '')
        print(f"\nExtraindo paciente: {patient_name}")
        mat_data = sio.loadmat(mat_filepath, squeeze_me=True, struct_as_record=False)
        
        # 1. ENCONTRA A IMAGEM RAW ORIGINAL (Sem rotacionar ainda!)
        raw_image = None
        if 'im' in mat_data and hasattr(mat_data['im'], 'shape') and mat_data['im'].size > 0:
            raw_image = mat_data['im']
        elif 'setstruct' in mat_data and hasattr(mat_data['setstruct'], 'IM') and mat_data['setstruct'].IM.size > 0:
            raw_image = mat_data['setstruct'].IM

        if raw_image is None:
            print("❌ Imagem RAW não encontrada neste .mat. Pulando paciente.")
            continue

        original_slice_shape = raw_image.shape[:2]
        
        # 2. GERA AS MÁSCARAS USANDO AS COORDENADAS E DIMENSÕES ORIGINAIS
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
                        
                        # Desenha as máscaras
                        draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'EpiX', z, t), get_coordinates(setstruct, 'EpiY', z, t), LABEL_EPI_LV, original_slice_shape, mask_idx)
                        draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'EndoX', z, t), get_coordinates(setstruct, 'EndoY', z, t), LABEL_ENDO_LV, original_slice_shape, mask_idx)
                        draw_mask_polygon(anat_seg, get_coordinates(setstruct, 'RVEndoX', z, t), get_coordinates(setstruct, 'RVEndoY', z, t), LABEL_ENDO_RV, original_slice_shape, mask_idx)

                # Desenha as fibroses (Scar)
                if hasattr(setstruct, 'Roi'):
                    rois = setstruct.Roi
                    if not isinstance(rois, np.ndarray): rois = [rois]
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
                        except Exception: pass

        # 3. O GRANDE FINAL: GIRA TUDO DE UMA VEZ SÓ
        raw_image = fix_matlab_orientation(raw_image)
        anat_seg = fix_matlab_orientation(anat_seg)
        scar_seg = fix_matlab_orientation(scar_seg)

        # 4. SALVA TUDO COM A MATRIZ AFFINE DA IMAGEM REFERÊNCIA (UKBB)
        nib.save(nib.Nifti1Image(raw_image, affine=ref_affine, header=ref_header), os.path.join(output_dir, f"{patient_name}_img.nii.gz"))
        nib.save(nib.Nifti1Image(anat_seg, affine=ref_affine, header=ref_header), os.path.join(output_dir, f"{patient_name}_seg_anat.nii.gz"))
        nib.save(nib.Nifti1Image(scar_seg, affine=ref_affine, header=ref_header), os.path.join(output_dir, f"{patient_name}_seg_scar.nii.gz"))
        
        print(f"   -> Concluído! Imagem e Segmentações rotacionadas e alinhadas.")

if __name__ == "__main__":
    process_mat_files(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)