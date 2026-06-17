import os
import argparse
import numpy as np
import nibabel as nib

def calculate_dice(pred: np.ndarray, target: np.ndarray, roi_label: int) -> float:
    """Calcula o Dice Similarity Coefficient para uma etiqueta (label) específica."""
    pred_roi = (pred == roi_label)
    target_roi = (target == roi_label)
    
    intersection = np.logical_and(pred_roi, target_roi).sum()
    total = pred_roi.sum() + target_roi.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_dir", required=True, help="Diretório de teste do Fold")
    parser.add_argument("-l", "--label", type=int, default=2, help="Valor do pixel correspondente à ROI (ex: 2 para Miocárdio, 1 para Scar)")
    args = parser.parse_args()

    test_dir = args.test_dir
    target_label = args.label
    
    pacientes = [p for p in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, p))]
    scores = []
    
    print("-" * 50)
    print(f"Resultados de Avaliação da ROI (Label {target_label})")
    print("-" * 50)

    for p in pacientes:
        path_p = os.path.join(test_dir, p)
        path_gt = os.path.join(path_p, "label_sa_ED.nii.gz")
        path_pred = os.path.join(path_p, "seg_sa_ED.nii.gz") # Ficheiro gerado pelo deploy_network
        
        if os.path.exists(path_gt) and os.path.exists(path_pred):
            gt_data = nib.load(path_gt).get_fdata()
            pred_data = nib.load(path_pred).get_fdata()
            
            dice = calculate_dice(pred_data, gt_data, target_label)
            scores.append(dice)
            print(f"Paciente {p}: Dice Score = {dice:.4f}")
        else:
            print(f"Paciente {p}: Ficheiros de segmentação não encontrados.")
            
    if scores:
        media = np.mean(scores)
        print("-" * 50)
        print(f"MÉDIA DO FOLD: {media:.4f}")
        print("-" * 50)

if __name__ == '__main__':
    main()