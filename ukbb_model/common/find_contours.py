import os
import sys
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.io import loadmat, savemat # type: ignore
from scipy.interpolate import interp1d, splprep, splev # type: ignore
from nibabel import load # type: ignore
from cv2 import dilate, erode # type: ignore
from skimage.measure import find_contours # type: ignore
from shapely.geometry import Polygon # type: ignore

def generate_closed_curve(contours, x, y, frame, saida):
    """
    Atualiza os arrays x e y com exatamente 80 pontos da maior curva fechada informada.
    """
    if len(contours) == 1:
        contour = contours[0]
        # retorna apenas o contorno válido
        saida[frame] = contour
        if len(contour) > 2:
            # new_contours = resample_closed_curve(contour, num_pontos)
            new_contours = smooth_closed_curve(contour, 15, 80)
            for ind, point in enumerate(new_contours):
                x[ind, 0, frame] = point[1]
                y[ind, 0, frame] = point[0]
            # Fechando a curva
            x[-1, 0, frame] = x[0, 0, frame]
            y[-1, 0, frame] = y[0, 0, frame]

    elif len(contours) > 1:
        maior_area = 0.0
        id_maior = 0
        for i, contour in enumerate(contours):
            area = calculate_area_closed_curve(contour)
            if area > maior_area:
                maior_area = area
                id_maior = i
        generate_closed_curve([contours[id_maior]], x, y, frame, saida)

def resample_closed_curve(points, num_points):
    points = np.asarray(points)
    
    # Close the curve if needed
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Compute cumulative distances (arc lengths)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # Total length of the curve
    total_length = arc_lengths[-1]
    
    # Create uniform arc lengths
    target_lengths = np.linspace(0, total_length, num_points, endpoint=False)
    
    # Interpolate in x and y separately
    interp_x = interp1d(arc_lengths, points[:, 0], kind='cubic')
    interp_y = interp1d(arc_lengths, points[:, 1], kind='cubic')
    
    resampled_x = interp_x(target_lengths)
    resampled_y = interp_y(target_lengths)
    
    return np.stack([resampled_x, resampled_y], axis=1)

def smooth_closed_curve(pontos: list, fator_suavizacao: float = 1.0, num_pontos_resultado: int = 80) -> np.ndarray:
    """
    Suaviza uma curva 2D fechada usando interpolação por splines.

    Args:
        pontos (list): Uma lista de tuplas ou listas com as coordenadas (x, y) dos pontos da curva.
                       Ex: [(x1, y1), (x2, y2), ...]
        fator_suavizacao (float): O fator de suavização (parâmetro 's' do splprep).
                                  s=0: a curva passará por todos os pontos.
                                  s>0: a curva será mais suave. O padrão é 1.0.
        num_pontos_resultado (int): O número de pontos que a curva suavizada final terá. O padrão é 100.

    Returns:
        np.ndarray: Um array NumPy de formato (num_pontos_resultado, 2) com os pontos (x, y) da curva suavizada.
                    Retorna um array vazio se a entrada for insuficiente.
    """
    # Validação da entrada: splprep precisa de pelo menos k+1 pontos, onde k=3 (grau cúbico)
    if len(pontos) < 4:
        # Retorna um array vazio ou os pontos originais, dependendo do que for mais útil para o seu caso.
        # Um array vazio é mais seguro para evitar erros inesperados.
        return np.array([])

    # 1. Descompactar a lista de pontos em coordenadas x e y separadas
    x_coords, y_coords = zip(*pontos)

    # 2. Calcular o modelo do spline (tck)
    #    per=True é essencial para tratar a curva como fechada. k=3 é para splines cúbicos.
    tck, u = splprep([x_coords, y_coords], s=fator_suavizacao, k=3, per=True)

    # 3. Gerar os pontos da nova curva suave
    u_new = np.linspace(u.min(), u.max(), num_pontos_resultado)
    x_spline, y_spline = splev(u_new, tck, der=0)

    # 4. Empacotar os resultados em um único array NumPy de formato (N, 2) e retorná-lo
    pontos_suavizados = np.vstack((x_spline, y_spline)).T
    return pontos_suavizados

def calculate_area_closed_curve(pontos):
    """
    Calcula a área de uma curva fechada (polígono) usando a Fórmula de Shoelace.

    Args:
        pontos: Uma lista de tuplas, onde cada tupla (x, y) representa um ponto
                no plano cartesiano que compõe a curva fechada.
                Os pontos devem estar em ordem sequencial (horário ou anti-horário).

    Returns:
        A área da curva fechada (polígono) como um float.
        Retorna 0.0 se a lista de pontos tiver menos de 3 pontos (não forma um polígono).
    """
    num_pontos = len(pontos)
    if num_pontos < 3:
        return 0.0

    soma_primeiro_termo = 0
    soma_segundo_termo = 0

    for i in range(num_pontos):
        x1, y1 = pontos[i]
        x2, y2 = pontos[(i + 1) % num_pontos]  # O operador % garante que o último ponto se conecta ao primeiro

        soma_primeiro_termo += x1 * y2
        soma_segundo_termo += y1 * x2

    area = 0.5 * abs(soma_primeiro_termo - soma_segundo_termo)
    return area

def check_contained_curve(curva_interna_pontos, curva_externa_pontos):
    """
    Verifica se uma curva fechada está completamente contida dentro da área de outra curva fechada.

    Args:
        curva_interna_pontos (list of tuples): Uma lista de tuplas (x, y) representando os pontos
                                               da curva que se deseja verificar se está contida.
                                               A ordem dos pontos deve formar uma curva fechada.
        curva_externa_pontos (list of tuples): Uma lista de tuplas (x, y) representando os pontos
                                               da curva que representa a área externa.
                                               A ordem dos pontos deve formar uma curva fechada.

    Returns:
        bool: True se a curva interna estiver completamente contida na curva externa, False caso contrário.
    """
    if len(curva_interna_pontos) < 3 or len(curva_externa_pontos) < 3:
        return False

    # Cria objetos Polygon a partir dos pontos.
    # Shapely automaticamente fecha o polígono se o último ponto não for igual ao primeiro.
    poligono_externo = Polygon(curva_externa_pontos)
    poligono_interno = Polygon(curva_interna_pontos)

    # Verifica se o polígono interno está contido dentro do polígono externo
    if (poligono_externo.contains(poligono_interno)):
        return True
    else:
        return False

# Parâmetros de entrada: patient.mat patient_seg.nii.gz
if len(sys.argv) < 2:
    sys.exit(1)
mat_path = sys.argv[1]
seg_path = sys.argv[2]

file_name = os.path.splitext(os.path.basename(mat_path))[0]
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_dir = os.path.join(root_dir, 'output')

try:
    mat = loadmat(mat_path)
    print(f'  Reading {mat_path} ...')
except FileNotFoundError:
    print('  Error: .MAT file not found')
    sys.exit(1)

if not os.path.exists(seg_path):
    print('   Directory {0} does not contain an image with file '
            'name {1}. Skip.'.format(mat_path, os.path.basename(seg_path)))
    sys.exit(1)
print('  Reading {} ...'.format(seg_path))

nim = load(seg_path)
image = nim.get_fdata()
X, Y, Z = image.shape
endox = np.full((80,1,Z), np.nan)
endoy = np.full((80,1,Z), np.nan)
rvendox = np.full((80,1,Z), np.nan)
rvendoy = np.full((80,1,Z), np.nan)
epix = np.full((80,1,Z), np.nan)
epiy = np.full((80,1,Z), np.nan)
rvepix = np.full((80,1,Z), np.nan)
rvepiy = np.full((80,1,Z), np.nan)
nan = np.full((80,1,1), np.nan)
endo = [[] for _ in range(Z)]
epi = [[] for _ in range(Z)]
rvendo = [[] for _ in range(Z)]
rvepi = [[] for _ in range(Z)]
mask_rvepi = [[] for _ in range(Z)]

""" Extraindo contorno das fatias segmentadas """
for frame in range(Z):
    # Invertendo ordem de armazenamento
    id = Z - 1 - frame
    slice_2d = image[:, :, frame]

    """ Endo """
    mask = (slice_2d == 1)
    contours = find_contours(mask, level=0.5) # Retorno: um array numpy [N, (y, x)]
    generate_closed_curve(contours, endox, endoy, id, endo)
        
    """ RVEndo """
    mask = (slice_2d == 3)
    contours = find_contours(mask, level=0.5)
    generate_closed_curve(contours, rvendox, rvendoy, id, rvendo)

    """ RVEpi """
    # Usando técnica de dilatação para criar um padding no RVEndo simulando seu epicárdio
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = dilate(mask.astype(np.uint8), kernel, iterations=1)
    # Subtraindo interseção entre a máscara 2
    mask = mask & ~(slice_2d == 2)
    # Adicionando padding na imagem original
    slice_2d += mask
    # Salvando máscaras do RVEpi
    mask_rvepi[id] = slice_2d
    # Extraindo os contornos
    contours = find_contours(slice_2d, level=0.1)
    generate_closed_curve(contours, rvepix, rvepiy, id, rvepi)

    """ Epi """
    mask = (slice_2d == 2)
    contours = find_contours(mask, level=0.5)
    generate_closed_curve(contours, epix, epiy, id, epi)

""" Verificando se o Endo e o RVEndo estão contidos no RVEpi """
for frame in range(int(Z/2)):
    if ((check_contained_curve(endo[frame], rvepi[frame]) == False) or (check_contained_curve(rvendo[frame], rvepi[frame]) == False)):
        ind = frame
        while ind >= 0:
            endox[:,0,ind] = endoy[:,0,ind] = rvendox[:,0,ind] = rvendoy[:,0,ind] = rvepix[:,0,ind] = rvepiy[:,0,ind] = epix[:,0,ind] = epiy[:,0,ind] = nan[:,0,0]
            ind -= 1
    if frame == -1: 
        break
while frame < Z:
    if ((check_contained_curve(endo[frame], rvepi[frame]) == False) and (len(endo[frame]) != 0)) or ((check_contained_curve(rvendo[frame], rvepi[frame]) == False) and (len(rvendo[frame]) != 0)):
        while frame < Z:
            endox[:,0,frame] = endoy[:,0,frame] = rvendox[:,0,frame] = rvendoy[:,0,frame] = rvepix[:,0,frame] = rvepiy[:,0,frame] = epix[:,0,frame] = epiy[:,0,frame] = nan[:,0,0]
            frame += 1
    frame += 1

""" Verificando o tamanho da área do epicárdio nas fatias iniciais """
frame = int(Z / 2)
while frame >= 0:
    tolerancia = 400
    if len(rvepi[frame]) != 0:
        if (calculate_area_closed_curve(rvepi[frame]) > calculate_area_closed_curve(rvepi[frame-1]) + tolerancia):
            frame -= 1
            while (frame >= 0):
                endox[:,0,frame] = endoy[:,0,frame] = rvendox[:,0,frame] = rvendoy[:,0,frame] = rvepix[:,0,frame] = rvepiy[:,0,frame] = epix[:,0,frame] = epiy[:,0,frame] = nan[:,0,0]
                frame -= 1
    frame -= 1

""" Verificando o tamanho da área do rvendo nas fatias iniciais """
frame = int(Z / 2)
while frame >= 0:
    tolerancia = 50
    if len(rvendo[frame]) != 0:
        if (calculate_area_closed_curve(rvendo[frame]) > calculate_area_closed_curve(rvendo[frame-1]) + tolerancia):
            frame -= 1
            while (frame >= 0):
                endox[:,0,frame] = endoy[:,0,frame] = rvendox[:,0,frame] = rvendoy[:,0,frame] = rvepix[:,0,frame] = rvepiy[:,0,frame] = epix[:,0,frame] = epiy[:,0,frame] = nan[:,0,0]
                frame -= 1
    frame -= 1

""" Verificando o tamanho da área do endo nas fatias iniciais """
frame = int(Z / 2)
while frame >= 0:
    tolerancia = 200
    if len(endo[frame]) != 0:
        if (calculate_area_closed_curve(endo[frame]) > calculate_area_closed_curve(endo[frame-1]) + tolerancia):
            frame -= 1
            while (frame >= 0):
                endox[:,0,frame] = endoy[:,0,frame] = rvendox[:,0,frame] = rvendoy[:,0,frame] = rvepix[:,0,frame] = rvepiy[:,0,frame] = epix[:,0,frame] = epiy[:,0,frame] = nan[:,0,0]
                frame -= 1
    frame -= 1

""" Usando a técnica de erosão para criar uma fatia final menor que a anterior artificialmente """
frame = Z - 1
while frame >= 0:
    if not np.isnan(rvepix[0,0,frame]):
        # Aplicando técnica de erosão
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = erode(mask_rvepi[frame].astype(np.uint8), kernel, iterations=2)
        # Extraindo contornos da segmentação completa com adição do padding
        contours = find_contours(mask, level=0.1)
        # Salvando os contornos em um array numpy
        if np.isnan(rvepix[0,0,Z-1]):
            generate_closed_curve(contours, rvepix, rvepiy, frame+1, rvepi)
        else:
            contours = find_contours(mask_rvepi[frame], level=0.1)
            endox = np.concatenate((endox, nan), axis=2)
            endoy = np.concatenate((endoy, nan), axis=2)
            rvendox = np.concatenate((rvendox, nan), axis=2)
            rvendoy = np.concatenate((rvendoy, nan), axis=2)
            epix = np.concatenate((epix, nan), axis=2)
            epiy = np.concatenate((epiy, nan), axis=2)
            new_rvepix = np.full((80,1,1), np.nan)
            new_rvepiy = np.full((80,1,1), np.nan)
            rvepi.append([])
            generate_closed_curve(contours, new_rvepix, new_rvepiy, 0, rvepi)
            rvepix = np.concatenate((rvepix, new_rvepix), axis=2)
            rvepiy = np.concatenate((rvepiy, new_rvepiy), axis=2)
            Z += 1
        break
    frame -= 1

# Salvando os contornos
for frame in range(Z):
    fig, ax = plt.subplots(figsize=(Y / 25, X / 25))
    ax.plot(endoy[:,0,frame], endox[:,0,frame], linewidth=0.5, color='black')
    ax.plot(epiy[:,0,frame], epix[:,0,frame], linewidth=0.5, color='green')
    ax.plot(rvendoy[:,0,frame], rvendox[:,0,frame], linewidth=0.5, color='red')
    ax.plot(rvepiy[:,0,frame], rvepix[:,0,frame], linewidth=0.5, color='blue')
    os.makedirs(f'{output_dir}/contours', exist_ok=True)
    contour_image_path = os.path.join(f'{output_dir}/contours', f'frame_{frame}.png')
    plt.axis('off')
    plt.savefig(contour_image_path, pad_inches=0)
    plt.close(fig)

# Reescrevendo arquivo .MAT
if 'setstruct' in mat: 
    mat['setstruct']['EndoX'][0][0] = endox
    mat['setstruct']['EndoY'][0][0] = endoy
    mat['setstruct']['RVEndoX'][0][0] = rvendox
    mat['setstruct']['RVEndoY'][0][0] = rvendoy
    mat['setstruct']['EpiX'][0][0] = epix
    mat['setstruct']['EpiY'][0][0] = epiy
    mat['setstruct']['RVEpiX'][0][0] = rvepix
    mat['setstruct']['RVEpiY'][0][0] = rvepiy
    savemat(f'{output_dir}/{file_name}_seg.mat', mat)
    print('  Saving new .MAT file ...')
else:
    print('Error: "setstruct" not found in .MAT file')

print ('Find Contours done.')