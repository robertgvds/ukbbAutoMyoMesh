import os
import glob
import argparse
import nibabel as nib
import nibabel.processing
from datetime import datetime

# =====================================================================
# ⚙️ CONFIGURAÇÕES PADRÃO (Para uso direto no código)
# =====================================================================
# A imagem original que tem a Matriz Affine e o Header corretos
DEFAULT_REFERENCE_PATH = "input/patient/patient.nii.gz" 

# A pasta onde estão TODOS os NIfTIs que perderam a orientação
DEFAULT_TARGET_DIR = "output/niftis_extraidos" 

# A pasta onde os NIfTIs consertados serão salvos
DEFAULT_OUTPUT_DIR = "output/niftis_corrigidos" 


def transfer_metadata_to_file(reference_nifti: nib.Nifti1Image, target_filepath: str, output_dir: str) -> None:
    filename = os.path.basename(target_filepath)
    target_nifti = nib.load(target_filepath)
    
    # EM VEZ DE COPIAR O HEADER, NÓS ROTACIONAMOS E REINTERPOLAMOS OS PIXELS
    print(f"  -> Reinterpolando pixels 3D para {filename}...")
    resampled_nifti = nibabel.processing.resample_from_to(target_nifti, reference_nifti)
    
    output_filepath = os.path.join(output_dir, filename)
    nib.save(resampled_nifti, output_filepath)
    
    print(f"  -> ✅ Corrigido fisicamente: {filename}")

def process_directory(reference_filepath: str, target_dir: str, output_dir: str) -> None:
    """
    Varre um diretório inteiro e aplica os metadados da imagem de referência 
    a todos os arquivos NIfTI encontrados.
    """
    print("=" * 70)
    print("🔄 Iniciando Clonagem de Metadados em Lote (Batch Metadata Transfer)")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Referência (Doador): {os.path.basename(reference_filepath)}")
    print(f"📂 Diretório Alvo (Receptores): {target_dir}")
    print("=" * 70)

    # Verifica se a imagem de referência existe
    if not os.path.exists(reference_filepath):
        print(f"❌ Erro: Imagem de referência não encontrada em: {reference_filepath}")
        return

    # Garante que a pasta de destino exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Busca por arquivos .nii e .nii.gz na pasta alvo
    nifti_files = glob.glob(os.path.join(target_dir, "*.nii*"))
    
    if not nifti_files:
        print(f"⚠️ Nenhum arquivo NIfTI encontrado na pasta: {target_dir}")
        return

    print(f"\nCarregando metadados e processando {len(nifti_files)} arquivos...\n")
    
    # Carregamos a imagem de referência apenas UMA VEZ para economizar memória e tempo
    reference_nifti = nib.load(reference_filepath)

    for filepath in nifti_files:
        transfer_metadata_to_file(reference_nifti, filepath, output_dir)

    print("\n" + "=" * 70)
    print(f"🎯 Processo Finalizado! Todos os arquivos foram salvos em: {output_dir}")
    print("=" * 70)

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Affine Cloner para NIfTI")
    parser.add_argument("-r", "--reference", required=False, help="Caminho para o NIfTI original (Doador da Affine)")
    parser.add_argument("-d", "--dir", required=False, help="Pasta contendo os NIfTIs com a Affine quebrada")
    parser.add_argument("-o", "--output", required=False, help="Pasta para salvar as imagens corrigidas")
    
    args = parser.parse_args()
    
    final_reference = args.reference if args.reference else DEFAULT_REFERENCE_PATH
    final_target_dir = args.dir if args.dir else DEFAULT_TARGET_DIR
    final_output_dir = args.output if args.output else DEFAULT_OUTPUT_DIR
    
    process_directory(final_reference, final_target_dir, final_output_dir)

if __name__ == "__main__":
    main()