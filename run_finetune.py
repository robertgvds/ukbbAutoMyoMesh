import os
import subprocess
import argparse
from datetime import datetime

# =====================================================================
# ⚙️ CONFIGURAÇÕES PADRÃO (Para teste rápido no PC)
# =====================================================================
DEFAULT_DATASET_DIR = "/home/robert/dev/research/DL-Cardiac-Segmentation/dataset/segmented_dataset"
DEFAULT_OUTPUT_DIR = "saida_teste"
DEFAULT_ITERATIONS = 2  # Apenas 2 iterações por padrão para teste (Smoke Test)

# Caminhos relativos padrão do projeto
RELATIVE_TRAIN_SCRIPT = "ukbb_model/common/train_network.py"
RELATIVE_MODEL_PATH = "ukbb_model/trained_model/FCN_sa"

def apply_compatibility_patches(project_root: str) -> None:
    """
    Substitui imports antigos do TensorFlow e corrige os nomes das pastas
    ('ukbb_cardiac' para 'ukbb_model') direto no código, EVITANDO atalhos no SO.
    """
    files_to_patch = [
        "ukbb_model/common/train_network.py",
        "ukbb_model/common/network.py",
        "ukbb_model/common/image_utils.py"
    ]
    
    tf1_import = "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()\n"
    
    for rel_path in files_to_patch:
        abs_path = os.path.join(project_root, rel_path)
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            changed = False
            
            # 1. Força a compatibilidade com TF1 (Se ainda não tiver sido feito)
            if "import tensorflow as tf" in content and "compat.v1" not in content:
                content = content.replace("import tensorflow as tf", tf1_import)
                changed = True
                
            # 2. Arruma o nome da pasta antiga para a sua atual (Evita o ln -sfn)
            if "ukbb_cardiac" in content:
                content = content.replace("ukbb_cardiac", "ukbb_model")
                changed = True
                
            # 3. Evita o erro de conflito de log do Kaggle
            if "DEFINE_string('log_dir'" in content:
                content = content.replace("DEFINE_string('log_dir'", "DEFINE_string('log_directory'")
                changed = True

            # 4. CORREÇÃO DO NIBABEL: Atualiza a função de leitura de imagem extinta
            if ".get_data()" in content:
                content = content.replace(".get_data()", ".get_fdata()")
                changed = True
                
            if changed:
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"🔧 Patch de compatibilidade aplicado em: {rel_path}")

def execute_training(dataset_dir: str, output_dir: str, iterations: int, project_root: str) -> None:
    """
    Executa a rede de Fine-Tuning do modelo AutoMyoMesh.
    """
    print("=" * 88)
    print("🧠 Deploying the fine-tuning network...")
    print("=" * 88)
    
    absolute_train_script = os.path.join(project_root, RELATIVE_TRAIN_SCRIPT)
    absolute_model_path = os.path.join(project_root, RELATIVE_MODEL_PATH)
    
    train_command = [
        "python", absolute_train_script, # Usando 'python' genérico (funciona no Win/Linux/Kaggle)
        "--dataset_dir", dataset_dir,
        "--checkpoint_dir", output_dir,
        "--model_path", absolute_model_path,
        "--train_iteration", str(iterations),
        "--train_batch_size", "1"
    ]
    
    environment_vars = os.environ.copy()
    environment_vars["CUDA_VISIBLE_DEVICES"] = "0"
    environment_vars["TF_CPP_MIN_LOG_LEVEL"] = "2"
    environment_vars["TF_USE_LEGACY_KERAS"] = "1" # A mágica que resolve o erro do Keras 3!
    
    if "PYTHONPATH" in environment_vars:
        environment_vars["PYTHONPATH"] = f"{project_root}:{environment_vars['PYTHONPATH']}"
    else:
        environment_vars["PYTHONPATH"] = project_root

    try:
        subprocess.run(
            train_command, 
            env=environment_vars, 
            check=True, 
            cwd=project_root 
        )
        print("\n✅ Treinamento da rede finalizado com sucesso!")
            
    except subprocess.CalledProcessError as error:
        print(f"\n❌ Erro crítico ao executar o treinamento! Código: {error.returncode}")

def run_pipeline(dataset_filepath: str, output_filepath: str, iterations: int) -> None:
    """
    Inicia o pipeline convertendo caminhos para o formato absoluto.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    absolute_dataset_path = os.path.abspath(dataset_filepath)
    absolute_output_path = os.path.abspath(output_filepath)
    
    print("=" * 88)
    print("🚀 Starting the pipeline for Fine-Tuning AutoMyoMesh.")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Dataset Dir: {absolute_dataset_path}")
    print(f"💾 Output Dir:  {absolute_output_path}")
    print(f"🔄 Iterations:  {iterations}")
    print("=" * 88)
    
    # 1. Aplica correções nos arquivos antes de rodar
    apply_compatibility_patches(project_root)
    
    # 2. Chama o treinamento real
    execute_training(absolute_dataset_path, absolute_output_path, iterations, project_root)

def main() -> None:
    """
    Função principal que gerencia os argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(description="Pipeline Wrapper - Fine-Tuning AutoMyoMesh")
    parser.add_argument("-d", "--dataset", required=False, help="Caminho completo para a pasta do dataset")
    parser.add_argument("-o", "--output", required=False, help="Caminho para a pasta onde salvará os pesos")
    parser.add_argument("-i", "--iters", type=int, required=False, help="Quantidade de iterações de treino")
    
    args = parser.parse_args()
    
    final_dataset_path = args.dataset if args.dataset else DEFAULT_DATASET_DIR
    final_output_path = args.output if args.output else DEFAULT_OUTPUT_DIR
    final_iterations = args.iters if args.iters else DEFAULT_ITERATIONS
    
    # Garante que a pasta de saída exista
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    
    run_pipeline(final_dataset_path, final_output_path, final_iterations)

if __name__ == '__main__':
    main()