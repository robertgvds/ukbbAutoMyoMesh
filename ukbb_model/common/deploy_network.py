# Copyright 2017, Wenjia Bai. All Rights Reserved.
# (Licença Apache 2.0 omitida para brevidade no comentário, mas válida)
# ============================================================================

import os
import time
import math
import numpy as np # type: ignore
import nibabel as nib # type: ignore
import tensorflow.compat.v1 as tf # type: ignore
from scipy.ndimage import rotate # type: ignore
from skimage.measure import find_contours # type: ignore
import matplotlib.pyplot as plt # type: ignore
import cv2 as cv # type: ignore
from scipy.io import savemat  # type: ignore

from ukbb_model.common.image_utils import rescale_intensity # type: ignore

# ============================================================================
# ⚙️ PARÂMETROS DE DEPLOYMENT (Flags)
# ============================================================================
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('seq_name', 'sa', 'Nome da sequência.')
tf.app.flags.DEFINE_string('data_dir', '/vol/bitbucket/wbai/own_work/ukbb_cardiac_demo', 'Caminho para o diretório de dados do paciente.')
tf.app.flags.DEFINE_string('model_path', '', 'Caminho para o modelo treinado.')
tf.app.flags.DEFINE_boolean('process_seq', False, 'Processar uma sequência temporal de imagens (4D).')
tf.app.flags.DEFINE_boolean('save_seg', True, 'Salvar as segmentações geradas.')
tf.app.flags.DEFINE_boolean('seg4', False, 'Segmentar as 4 câmaras na visualização de eixo longo (LA).')
tf.app.flags.DEFINE_string('output_dir', 'output', 'Diretório onde os resultados serão salvos.')

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # ====================================================================
        # 1. CARREGAR O MODELO
        # ====================================================================
        saver = tf.train.import_meta_graph(f"{FLAGS.model_path}.meta")
        saver.restore(sess, FLAGS.model_path)

        print('Start deployment on the data set ...')
        start_time = time.time()

        processed_list = []
        table_time = []
        
        # Garante que o diretório de saída exista antes de começar a salvar as coisas
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        
        image_name = os.path.join(FLAGS.data_dir, f"{FLAGS.seq_name}.nii.gz")
        
        if not os.path.exists(image_name):
            print(f'  Directory {FLAGS.data_dir} does not contain an image with file name {os.path.basename(image_name)}. Skip.')
            exit(1)

        print(f'  Reading {image_name} ...')
        nim = nib.load(image_name)
        image = nim.get_fdata()

        # ====================================================================
        # 2. PROCESSAMENTO: MODO SEQUÊNCIA (4D - CINE)
        # ====================================================================
        if FLAGS.process_seq:
            X, Y, Z, T = image.shape
            orig_image = image

            print('  Segmenting full sequence ...')
            start_seg_time = time.time()

            # Rescala de intensidade
            image = rescale_intensity(image, (1, 99))
            pred = np.zeros(image.shape)

            # Ajuste de Padding para a rede (múltiplos de 16)
            X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
            x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
            x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
            image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

            # Processa cada frame no tempo
            for t in range(T):
                image_fr = image[:, :, :, t]
                image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                image_fr = np.expand_dims(image_fr, axis=-1)
                
                prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'], feed_dict={'image:0': image_fr, 'training:0': False})
                
                pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                pred[:, :, :, t] = pred_fr

            seg_time = time.time() - start_seg_time
            print(f'  Segmentation time = {seg_time:3f}s')
            table_time.append(seg_time)
            processed_list.append(FLAGS.seq_name)

            # Identificação de Fim de Diástole (ED) e Sístole (ES)
            k = {'ED': 0}
            if FLAGS.seq_name == 'sa' or (FLAGS.seq_name == 'la_4ch' and FLAGS.seg4):
                k['ES'] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))
            else:
                k['ES'] = np.argmax(np.sum(pred == 1, axis=(0, 1, 2)))
            print(f"  ED frame = {k['ED']}, ES frame = {k['ES']}")

            # Salvar Segmentações (Modo 4D)
            if FLAGS.save_seg:
                print('  Saving segmentation ...')
                nim2 = nib.Nifti1Image(pred, nim.affine)
                nim2.header['pixdim'] = nim.header['pixdim']
                
                prefix = 'seg4_' if (FLAGS.seq_name == 'la_4ch' and FLAGS.seg4) else 'seg_'
                seg_name = os.path.join(FLAGS.output_dir, f"{prefix}{FLAGS.seq_name}.nii.gz")
                nib.save(nim2, seg_name)

                # Salvar imagens originais e predições isoladas para ED e ES
                for fr in ['ED', 'ES']:
                    # Imagem original isolada
                    orig_fr_name = os.path.join(FLAGS.output_dir, f"{FLAGS.seq_name}_{fr}.nii.gz")
                    nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine), orig_fr_name)
                    
                    # Segmentação isolada
                    seg_fr_name = os.path.join(FLAGS.output_dir, f"{prefix}{FLAGS.seq_name}_{fr}.nii.gz")
                    nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine), seg_fr_name)

        # ====================================================================
        # 3. PROCESSAMENTO: MODO FRAME ÚNICO (3D)
        # ====================================================================
        else:
            X, Y = image.shape[:2]
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)

            print('  Segmenting image ...')
            start_seg_time = time.time()

            image = rescale_intensity(image, (1, 99))

            X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
            x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
            x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
            image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

            image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
            image = np.expand_dims(image, axis=-1)

            prob, pred = sess.run(['prob:0', 'pred:0'], feed_dict={'image:0': image, 'training:0': False})

            pred = np.transpose(pred, axes=(1, 2, 0))
            pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

            seg_time = time.time() - start_seg_time
            print(f'  Segmentation time = {seg_time:3f}s')
            table_time.append(seg_time)
            processed_list.append(FLAGS.seq_name)

            # Salvar Segmentação (Modo 3D)
            if FLAGS.save_seg:
                print('  Saving segmentation ...')
                nim2 = nib.Nifti1Image(pred, nim.affine)
                nim2.header['pixdim'] = nim.header['pixdim']
                
                if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                    seg_name = os.path.join(FLAGS.output_dir, f"seg4_{FLAGS.seq_name}.nii.gz")
                else:
                    seg_name = os.path.join(FLAGS.output_dir, f"{FLAGS.seq_name}_seg.nii.gz")
                    
                nib.save(nim2, seg_name)

                # Extração de contornos e salvamento do .mat na pasta de saída
                print('Finding contours ...')
                mat_path = os.path.join(FLAGS.output_dir, f"{FLAGS.seq_name}.mat")
                os.system(f'python3 common/find_contours.py {mat_path} {seg_name}')

        # ====================================================================
        # 4. FINALIZAÇÃO E ESTATÍSTICAS
        # ====================================================================
        metric = 'sequence' if FLAGS.process_seq else 'frame'
        print(f'Average segmentation time = {np.mean(table_time):.3f}s per {metric}')
        
        process_time = time.time() - start_time
        print(f'Including image I/O, CUDA resource allocation, it took {process_time:.3f}s '
              f'for processing {len(processed_list)} subjects '
              f'({process_time / len(processed_list):.3f}s per subject).')