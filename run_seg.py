
import os
import subprocess
import argparse
import shutil
from datetime import datetime

# =====================================================================
# ⚙️ CONFIGURAÇÕES PADRÃO (Para uso direto no código)
# =====================================================================
DEFAULT_INPUT_FILEPATH = "/home/robert/dev/research/DL-Cardiac-Segmentation/AutoMyoMesh/input/2/sa_corrigido.nii.gz"
DEFAULT_PROCESS_SEQ = False 

# Caminhos relativos padrão do projeto
RELATIVE_MODEL_PATH = "ukbb_model/trained_model/FCN_sa"
RELATIVE_DEPLOY_SCRIPT = "ukbb_model/common/deploy_network.py"

def execute_segmentation(seq_name: str, data_dir: str, process_seq: bool, project_root: str) -> None:
    """
    Executa a rede de segmentação e move os resultados para a pasta do paciente.
    """
    print("=" * 88)
    print(f"🧠 Deploying the segmentation network for {seq_name}...")
    print("=" * 88)
    
    absolute_deploy_script = os.path.join(project_root, RELATIVE_DEPLOY_SCRIPT)
    absolute_model_path = os.path.join(project_root, RELATIVE_MODEL_PATH)
    
    deploy_command = [
        "python3", absolute_deploy_script,
        "--seq_name", seq_name,
        "--data_dir", data_dir,
        "--model_path", absolute_model_path
    ]
    
    if process_seq:
        deploy_command.append("--process_seq")
        print("Modo CINE (4D Sequence) Ativado!")
    
    environment_vars = os.environ.copy()
    environment_vars["CUDA_VISIBLE_DEVICES"] = "0"
    environment_vars["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    if "PYTHONPATH" in environment_vars:
        environment_vars["PYTHONPATH"] = f"{project_root}:{environment_vars['PYTHONPATH']}"
    else:
        environment_vars["PYTHONPATH"] = project_root

    try:
        # cwd=project_root garante que a rede ache seus arquivos internos (find_contours)
        subprocess.run(
            deploy_command, 
            env=environment_vars, 
            check=True, 
            cwd=project_root 
        )
        print("\n✅ Processamento da rede finalizado!")
            
    except subprocess.CalledProcessError as error:
        print(f"\n❌ Erro crítico ao executar a segmentação! Código: {error.returncode}")

def run_pipeline(input_filepath: str, process_seq: bool) -> None:
    """
    Inicia o pipeline convertendo caminhos para o formato absoluto.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    absolute_input_path = os.path.abspath(input_filepath)
    
    filename = os.path.basename(absolute_input_path)          
    seq_name = filename.replace('.nii.gz', '').replace('.nii', '')
    patient_dir = os.path.dirname(absolute_input_path)
    
    print("=" * 88)
    print("🚀 Starting the pipeline for patient data processing.")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Target Dir: {patient_dir}")
    print(f"🎞️ Sequence: {seq_name}")
    print("=" * 88)
    
    execute_segmentation(seq_name, patient_dir, process_seq, project_root)

def main() -> None:
    """
    Função principal que gerencia os argumentos.
    """
    parser = argparse.ArgumentParser(description="Pipeline Wrapper - AutoMyoMesh")
    parser.add_argument("-i", "--input", required=False, help="Caminho completo para a imagem")
    parser.add_argument("--seq", action="store_true", help="Processa a sequência de tempo inteira (Imagens 4D)")
    
    args = parser.parse_args()
    
    final_input_path = args.input if args.input else DEFAULT_INPUT_FILEPATH
    final_process_seq = args.seq if args.seq else DEFAULT_PROCESS_SEQ
    
    run_pipeline(final_input_path, final_process_seq)

if __name__ == '__main__':
    main()