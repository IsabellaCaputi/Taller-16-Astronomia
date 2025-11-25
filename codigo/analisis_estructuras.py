"""
Análisis de Estructura a Gran Escala del Universo
Estudiantes: [NOMBRES]
"""

import time
import numpy as np
from astropy.table import Table, Column
import matplotlib.pyplot as plt
import os

# PARÁMETROS
N_VECINOS = 20
DISTANCIA_AGRUPACION_CLUSTER = 50.0  # millones años luz
DISTANCIA_AGRUPACION_VOID = 80.0     # millones años luz
DISTANCIA_MAXIMA_FILAMENTO = 300.0   # millones años luz
PASO_MUESTREO_FILAMENTO = 10.0       # millones años luz

INPUT_FILE = 'galaxy_cartesian_coordinates.ecsv'

# Salidas
OUT_CATALOGO_CLUSTERS = 'catalogo_clusters.ecsv'
OUT_CATALOGO_VOIDS = 'catalogo_voids.ecsv'
OUT_TRAZADO_FILAMENTOS = 'trazado_filamentos.ecsv'
OUT_ESTADISTICAS = 'estadisticas.txt'
PDF_DISTRIBUCION = 'distribucion_galaxias.pdf'
PDF_CLASIFICADAS = 'estructuras_clasificadas.pdf'
PDF_CLUSTERS_VOIDS = 'clusters_y_voids.pdf'
PDF_RED_COMPLETA = 'red_cosmica_completa.pdf'

def leer_datos():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {INPUT_FILE}")
    t = Table.read(INPUT_FILE, format='ascii.ecsv')
    # Esperamos columnas TARGETID, X, Y, Z
    coords = np.vstack((t['X'], t['Y'], t['Z'])).T.astype(float)
    ids = np.array(t['TARGETID'])
    return ids, coords, t

def calcular_densidad_local(coords, n_vecinos=N_VECINOS):
    """
    Para cada galaxia, encuentra los n_vecinos más cercanos (excluyendo la propia)
    y calcula una estimación de densidad local basada en volumen esférico
    definido por la distancia al n-ésimo vecino: rho = N / (4/3 pi r_n^3)
    """
    N = coords.shape[0]
    densidades = np.zeros(N, dtype=float)

    # Para eficiencia, procesar por bloques (evita almacenar NxN)
    for i in range(N):
        # distancias al resto
        diff = coords - coords[i]
        d2 = np.sum(diff * diff, axis=1)
        # incluir la misma galaxia: distancia 0 => estará en primera posición
        # tomar n_vecinos+1 para obtener n_vecinos excluyendo a sí misma
        k = min(n_vecinos + 1, N)
        idx = np.argpartition(d2, k)[:k]
        # excluir la propia (d2 == 0)
        if i in idx:
            idx = idx[idx != i]
        # si menos vecinos disponibles ajustar
        if idx.size == 0:
            r_n = 1e-6
        else:
            # distancia al vecino n
            # si hay menos vecinos que n_vecinos, tomar el mayor disponible
            selected = np.sqrt(d2[idx])
            r_n = np.max(selected)  # radio que contiene los n vecinos
            if r_n == 0:
                r_n = 1e-6
        volumen = (4.0 / 3.0) * np.pi * (r_n ** 3)
        dens = n_vecinos / volumen
        densidades[i] = dens
    return densidades

def clasificar_galaxias(densidades):
    """
    Clasificar según umbrales:
    CUMULO: densidad > 2 * densidad_mediana
    FILAMENTO: 0.5 * densidad_mediana <= dens <= 2 * densidad_mediana
    VACIO: dens < 0.5 * densidad_mediana
    """
    med = np.median(densidades)
    umbral_alto = 2.0 * med
    umbral_bajo = 0.5 * med
    labels = np.empty(densidades.shape, dtype='U10')
    labels[densidades > umbral_alto] = 'CUMULO'
    mid_mask = (densidades >= umbral_bajo) & (densidades <= umbral_alto)
    labels[mid_mask] = 'FILAMENTO'
    labels[densidades < umbral_bajo] = 'VACIO'
    return labels, med, umbral_bajo, umbral_alto

def agrupar_por_distancia(indices, coords_subset, linking_length):
    """
    Agrupamiento simple tipo 'friends-of-friends' sobre un subconjunto de índices.
    devuelve lista de grupos (cada grupo es lista de índices en la referencia original)
    indices: array de índices globales
    coords_subset: coords[indices]
    """
    if len(indices) == 0:
        return []
    idx_unassigned = set(range(len(indices)))
    groups = []
    coords_local = coords_subset
    while idx_unassigned:
        seed_local = idx_unassigned.pop()
        to_visit = [seed_local]
        group_local = [seed_local]
        while to_visit:
            cur = to_visit.pop()
            # buscar vecinos dentro de linking_length en el subconjunto
            diff = coords_local - coords_local[cur]
            d2 = np.sum(diff * diff, axis=1)
            neigh_local = np.where(d2 <= linking_length ** 2)[0]
            for nl in neigh_local:
                if nl in idx_unassigned:
                    idx_unassigned.remove(nl)
                    to_visit.append(nl)
                    group_local.append(nl)
        # convertir locales a globales
        group_global = [int(indices[g]) for g in group_local]
        groups.append(group_global)
    return groups

def identificar_clusters(coords, labels):
    """
    Tomar galaxias etiquetadas como 'CUMULO' y agruparlas con linking length DISTANCIA_AGRUPACION_CLUSTER.
    Para cada grupo calcular centro de masa (promedio), radio (max distancia), y conteo.
    """
    indices_cluster = np.where(labels == 'CUMULO')[0]
    coords_cluster = coords[indices_cluster]
    groups = agrupar_por_distancia(indices_cluster, coords_cluster, DISTANCIA_AGRUPACION_CLUSTER)
    catalog = []
    for i, group in enumerate(groups, start=1):
        pts = coords[np.array(group)]
        centro = np.mean(pts, axis=0)
        radios = np.sqrt(np.sum((pts - centro) ** 2, axis=1))
        radio = float(np.max(radios)) if radios.size > 0 else 0.0
        n_gal = len(group)
        catalog.append({'CLUSTER_ID': i,
                        'X_CENTRO': float(centro[0]),
                        'Y_CENTRO': float(centro[1]),
                        'Z_CENTRO': float(centro[2]),
                        'RADIO': float(radio),
                        'N_GALAXIAS': int(n_gal),
                        'MEMBERS': group})
    return catalog

def identificar_voids(coords, labels):
    """
    Identificar 'vacíos' agrupando galaxias etiquetadas 'VACIO' con linking length DISTANCIA_AGRUPACION_VOID.
    Interpreta vacíos como agrupaciones de galaxias escasas (aproximación práctica).
    """
    indices_void = np.where(labels == 'VACIO')[0]
    coords_void = coords[indices_void]
    groups = agrupar_por_distancia(indices_void, coords_void, DISTANCIA_AGRUPACION_VOID)
    catalog = []
    for i, group in enumerate(groups, start=1):
        pts = coords[np.array(group)]
        centro = np.mean(pts, axis=0)
        # radio efectivo: máxima distancia desde el centro a las galaxias (aprox.)
        radios = np.sqrt(np.sum((pts - centro) ** 2, axis=1))
        radio = float(np.max(radios)) if radios.size > 0 else 0.0
        catalog.append({'VOID_ID': i,
                        'X_CENTRO': float(centro[0]),
                        'Y_CENTRO': float(centro[1]),
                        'Z_CENTRO': float(centro[2]),
                        'RADIO': float(radio),
                        'MEMBERS': group})
    return catalog

def trazar_filamentos(clusters):
    """
    Conectar pares de cúmulos cuya separación sea < DISTANCIA_MAXIMA_FILAMENTO.
    Para cada par, samplear puntos a lo largo del segmento con paso PASO_MUESTREO_FILAMENTO.
    """
    filamentos = []
    if len(clusters) < 2:
        return filamentos
    centers = np.array([[c['X_CENTRO'], c['Y_CENTRO'], c['Z_CENTRO']] for c in clusters])
    n_clusters = len(clusters)
    fil_id = 0
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            c1 = centers[i]
            c2 = centers[j]
            dist = np.linalg.norm(c2 - c1)
            if dist <= DISTANCIA_MAXIMA_FILAMENTO:
                fil_id += 1
                # número de puntos (incluye extremos)
                if dist == 0:
                    n_points = 1
                else:
                    n_points = int(np.ceil(dist / PASO_MUESTREO_FILAMENTO)) + 1
                tvals = np.linspace(0, 1, n_points)
                pts = np.outer(1 - tvals, c1) + np.outer(tvals, c2)
                for p in pts:
                    filamentos.append({'FILAMENTO_ID': fil_id,
                                       'X': float(p[0]),
                                       'Y': float(p[1]),
                                       'Z': float(p[2]),
                                       'CLUSTER_ORIGEN': clusters[i]['CLUSTER_ID'],
                                       'CLUSTER_DESTINO': clusters[j]['CLUSTER_ID']})
    return filamentos

def generar_graficas(coords, labels, clusters, voids, filamentos):
    """Crear los 4 PDFs requeridos con 3 subplots cada uno: XY, XZ, YZ"""
    proyecciones = [('X','Y',0,1), ('X','Z',0,2), ('Y','Z',1,2)]
    # Preparar arrays fácilmente indexables
    x = coords[:,0]; y = coords[:,1]; z = coords[:,2]

    # 1) distribucion_galaxias.pdf - todas galaxias grises
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    for ax, (_,_,i1,i2) in zip(axs, proyecciones):
        ax.scatter(coords[:,i1], coords[:,i2], s=1)
        ax.set_xlabel(['X','Y','Z'][i1] + ' (Mly)')
        ax.set_ylabel(['X','Y','Z'][i2] + ' (Mly)')
    fig.suptitle('Distribución de galaxias (proyecciones 2D)')
    plt.tight_layout()
    fig.savefig(PDF_DISTRIBUCION)
    plt.close(fig)

    # 2) estructuras_clasificadas.pdf (verde: VACIO, azul: FILAMENTO, rojo: CUMULO)
    color_map = {'VACIO': 'green', 'FILAMENTO': 'blue', 'CUMULO': 'red'}
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    for ax, (_,_,i1,i2) in zip(axs, proyecciones):
        for lab in ['VACIO','FILAMENTO','CUMULO']:
            mask = (labels == lab)
            ax.scatter(coords[mask,i1], coords[mask,i2], s=2, label=lab, color=color_map[lab])
        ax.set_xlabel(['X','Y','Z'][i1] + ' (Mly)')
        ax.set_ylabel(['X','Y','Z'][i2] + ' (Mly)')
        ax.legend(markerscale=4)
    fig.suptitle('Estructuras clasificadas (VACIO verde, FILAMENTO azul, CUMULO rojo)')
    plt.tight_layout()
    fig.savefig(PDF_CLASIFICADAS)
    plt.close(fig)

    # 3) clusters_y_voids.pdf - marcar centros y radios (con leyenda)
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    for ax, (_,_,i1,i2) in zip(axs, proyecciones):
        ax.scatter(coords[:,i1], coords[:,i2], s=1, color='gray')

        # Dibujado de clusters y voids (como antes)
        for c in clusters:
            cx, cy, cz = c['X_CENTRO'], c['Y_CENTRO'], c['Z_CENTRO']
            arr = [cx, cy, cz]
            ax.plot(arr[i1], arr[i2], 'o', color='red', markersize=4)
            circle = plt.Circle((arr[i1], arr[i2]), c['RADIO'],
                                color='red', fill=False, linestyle='--', linewidth=0.8)
            ax.add_patch(circle)

        for v in voids:
            vx, vy, vz = v['X_CENTRO'], v['Y_CENTRO'], v['Z_CENTRO']
            arr = [vx, vy, vz]
            ax.plot(arr[i1], arr[i2], 'o', color='green', markersize=4)
            circle = plt.Circle((arr[i1], arr[i2]), v['RADIO'],
                                color='green', fill=False, linestyle='--', linewidth=0.8)
            ax.add_patch(circle)

        # Crear handles proxy para la leyenda (evita entradas duplicadas)
        galaxy_handle = plt.Line2D([], [], marker='.', linestyle='None', color='gray', markersize=6, label='Galaxias')
        cluster_handle = plt.Line2D([], [], marker='o', linestyle='None', color='red', markersize=7, label='Cúmulos (centro)')
        cluster_radius_handle = plt.Line2D([], [], color='red', linestyle='--', label='Radio cúmulo')
        void_handle = plt.Line2D([], [], marker='o', linestyle='None', color='green', markersize=7, label='Vacíos (centro)')
        void_radius_handle = plt.Line2D([], [], color='green', linestyle='--', label='Radio vacío')

        ax.set_xlabel(['X','Y','Z'][i1] + ' (Mly)')
        ax.set_ylabel(['X','Y','Z'][i2] + ' (Mly)')

        # Añadir leyenda en una posición que no tape la información
        ax.legend(handles=[galaxy_handle, cluster_handle, cluster_radius_handle, void_handle, void_radius_handle],
                  loc='best', fontsize='small', framealpha=0.9)
    fig.suptitle('Centros y radios de cúmulos (rojo) y vacíos (verde)')
    plt.tight_layout()
    fig.savefig(PDF_CLUSTERS_VOIDS)
    plt.close(fig)


        # 4) red_cosmica_completa.pdf - galaxias grises, filamentos azules, clusters rojos, voids verdes (con leyenda)
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    for ax, (_,_,i1,i2) in zip(axs, proyecciones):
        ax.scatter(coords[:,i1], coords[:,i2], s=1, color='lightgray')

        # filamentos como líneas azules (especificar color para consistencia)
        if filamentos:
            fil_table = {}
            for f in filamentos:
                fid = f['FILAMENTO_ID']
                fil_table.setdefault(fid, []).append((f['X'], f['Y'], f['Z']))
            for fid, pts in fil_table.items():
                pts = np.array(pts)
                ax.plot(pts[:,i1], pts[:,i2], linewidth=0.8, color='blue')

        # clusters (centros)
        for c in clusters:
            arr = [c['X_CENTRO'], c['Y_CENTRO'], c['Z_CENTRO']]
            ax.plot(arr[i1], arr[i2], 'o', color='red', markersize=4)

        # voids (centros)
        for v in voids:
            arr = [v['X_CENTRO'], v['Y_CENTRO'], v['Z_CENTRO']]
            ax.plot(arr[i1], arr[i2], 'o', color='green', markersize=4)

        # Handles proxy para la leyenda
        galaxy_handle = plt.Line2D([], [], marker='.', linestyle='None', color='lightgray', markersize=6, label='Galaxias')
        filament_handle = plt.Line2D([], [], color='blue', linewidth=1.5, label='Filamentos')
        cluster_handle = plt.Line2D([], [], marker='o', linestyle='None', color='red', markersize=7, label='Cúmulos (centro)')
        void_handle = plt.Line2D([], [], marker='o', linestyle='None', color='green', markersize=7, label='Vacíos (centro)')

        ax.set_xlabel(['X','Y','Z'][i1] + ' (Mly)')
        ax.set_ylabel(['X','Y','Z'][i2] + ' (Mly)')

        ax.legend(handles=[galaxy_handle, filament_handle, cluster_handle, void_handle],
                  loc='best', fontsize='small', framealpha=0.9)
    fig.suptitle('Red cósmica completa: galaxias, filamentos, cúmulos y vacíos')
    plt.tight_layout()
    fig.savefig(PDF_RED_COMPLETA)
    plt.close(fig)


def escribir_catalogos(clusters, voids, filamentos, ids, labels, densidades, dens_mediana):
    """Guardar los catálogos ECSV y estadisticas"""
    # Catalogo clusters
    if clusters:
        cols = {
            'CLUSTER_ID': [c['CLUSTER_ID'] for c in clusters],
            'X_CENTRO': [c['X_CENTRO'] for c in clusters],
            'Y_CENTRO': [c['Y_CENTRO'] for c in clusters],
            'Z_CENTRO': [c['Z_CENTRO'] for c in clusters],
            'RADIO': [c['RADIO'] for c in clusters],
            'N_GALAXIAS': [c['N_GALAXIAS'] for c in clusters],
        }
        t = Table(cols)
        t.write(OUT_CATALOGO_CLUSTERS, format='ascii.ecsv', overwrite=True)
    else:
        # escribir tabla vacía con columnas correctas
        t = Table(names=('CLUSTER_ID','X_CENTRO','Y_CENTRO','Z_CENTRO','RADIO','N_GALAXIAS'),
                  dtype=('i4','f8','f8','f8','f8','i4'))
        t.write(OUT_CATALOGO_CLUSTERS, format='ascii.ecsv', overwrite=True)

    # Catalogo voids
    if voids:
        cols = {
            'VOID_ID': [v['VOID_ID'] for v in voids],
            'X_CENTRO': [v['X_CENTRO'] for v in voids],
            'Y_CENTRO': [v['Y_CENTRO'] for v in voids],
            'Z_CENTRO': [v['Z_CENTRO'] for v in voids],
            'RADIO': [v['RADIO'] for v in voids],
        }
        t = Table(cols)
        t.write(OUT_CATALOGO_VOIDS, format='ascii.ecsv', overwrite=True)
    else:
        t = Table(names=('VOID_ID','X_CENTRO','Y_CENTRO','Z_CENTRO','RADIO'),
                  dtype=('i4','f8','f8','f8','f8'))
        t.write(OUT_CATALOGO_VOIDS, format='ascii.ecsv', overwrite=True)

    # Trazado filamentos
    if filamentos:
        cols = {
            'FILAMENTO_ID': [f['FILAMENTO_ID'] for f in filamentos],
            'X': [f['X'] for f in filamentos],
            'Y': [f['Y'] for f in filamentos],
            'Z': [f['Z'] for f in filamentos],
            'CLUSTER_ORIGEN': [f['CLUSTER_ORIGEN'] for f in filamentos],
            'CLUSTER_DESTINO': [f['CLUSTER_DESTINO'] for f in filamentos],
        }
        t = Table(cols)
        t.write(OUT_TRAZADO_FILAMENTOS, format='ascii.ecsv', overwrite=True)
    else:
        t = Table(names=('FILAMENTO_ID','X','Y','Z','CLUSTER_ORIGEN','CLUSTER_DESTINO'),
                  dtype=('i4','f8','f8','f8','i4','i4'))
        t.write(OUT_TRAZADO_FILAMENTOS, format='ascii.ecsv', overwrite=True)

    # Estadisticas
    n_total = len(labels)
    n_clusters_gal = np.sum(labels == 'CUMULO')
    n_filament_gal = np.sum(labels == 'FILAMENTO')
    n_void_gal = np.sum(labels == 'VACIO')

    num_clusters = len(clusters)
    radio_promedio_clusters = np.mean([c['RADIO'] for c in clusters]) if clusters else 0.0
    if clusters:
        biggest_cluster = max(clusters, key=lambda c: c['N_GALAXIAS'])
    else:
        biggest_cluster = None

    num_voids = len(voids)
    radio_promedio_voids = np.mean([v['RADIO'] for v in voids]) if voids else 0.0
    biggest_void = max(voids, key=lambda v: v['RADIO']) if voids else None

    num_filamentos = len(set([f['FILAMENTO_ID'] for f in filamentos])) if filamentos else 0
    # longitud promedio de filamentos (aprox): contar longitud entre puntos consecutivos
    longitudes = []
    if filamentos:
        # agrupar por id
        fil_table = {}
        for f in filamentos:
            fil_table.setdefault(f['FILAMENTO_ID'], []).append((f['X'], f['Y'], f['Z']))
        for fid, pts in fil_table.items():
            pts = np.array(pts)
            if pts.shape[0] >= 2:
                segs = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
                longitudes.append(np.sum(segs))
    longitud_promedio = np.mean(longitudes) if longitudes else 0.0
    total_puntos_filamentos = len(filamentos)

    with open(OUT_ESTADISTICAS, 'w') as f:
        f.write("========================================\n")
        f.write("ESTRUCTURA A GRAN ESCALA - ESTADÍSTICAS\n")
        f.write("========================================\n\n")
        f.write(f"Número total de galaxias: {n_total}\n\n")
        f.write("CLASIFICACIÓN POR DENSIDAD\n")
        f.write("---------------------------\n")
        f.write(f"Galaxias en CUMULOS: {n_clusters_gal} ({100.0 * n_clusters_gal / n_total:.2f}%)\n")
        f.write(f"Galaxias en FILAMENTOS: {n_filament_gal} ({100.0 * n_filament_gal / n_total:.2f}%)\n")
        f.write(f"Galaxias en VACIOS: {n_void_gal} ({100.0 * n_void_gal / n_total:.2f}%)\n\n")

        f.write("CÚMULOS IDENTIFICADOS\n")
        f.write("---------------------\n")
        f.write(f"Número total de cúmulos: {num_clusters}\n")
        f.write(f"Radio promedio: {radio_promedio_clusters:.3f} millones años luz\n")
        if biggest_cluster:
            f.write(f"Cúmulo más grande: ID {biggest_cluster['CLUSTER_ID']} ({biggest_cluster['N_GALAXIAS']} galaxias, radio {biggest_cluster['RADIO']:.3f})\n")
        f.write("\n")

        f.write("VACÍOS IDENTIFICADOS\n")
        f.write("--------------------\n")
        f.write(f"Número total de vacíos: {num_voids}\n")
        f.write(f"Radio promedio: {radio_promedio_voids:.3f} millones años luz\n")
        if biggest_void:
            f.write(f"Vacío más grande: ID {biggest_void['VOID_ID']} (radio {biggest_void['RADIO']:.3f} millones años luz)\n")
        f.write("\n")

        f.write("FILAMENTOS TRAZADOS\n")
        f.write("-------------------\n")
        f.write(f"Número de filamentos: {num_filamentos}\n")
        f.write(f"Longitud promedio: {longitud_promedio:.3f} millones años luz\n")
        f.write(f"Total de puntos trazados: {total_puntos_filamentos}\n")
        f.write("\n")
        f.write(f"Densidad mediana global usada: {dens_mediana:.6e}\n")

def main():
    t0 = time.time()
    print("Leyendo datos...")
    ids, coords, original_table = leer_datos()
    print(f"{coords.shape[0]} galaxias leídas.")

    print("Calculando densidad local (N vecinos)... Esto puede tardar varios segundos)...")
    dens = calcular_densidad_local(coords, n_vecinos=N_VECINOS)
    print("Clasificando galaxias...")
    labels, med, umbral_bajo, umbral_alto = clasificar_galaxias(dens)

    print("Identificando cúmulos...")
    clusters = identificar_clusters(coords, labels)
    print(f"{len(clusters)} cúmulos identificados.")

    print("Identificando vacíos...")
    voids = identificar_voids(coords, labels)
    print(f"{len(voids)} vacíos identificados.")

    print("Trazando filamentos entre cúmulos...")
    filamentos = trazar_filamentos(clusters)
    print(f"{len(set([f['FILAMENTO_ID'] for f in filamentos])) if filamentos else 0} filamentos generados, {len(filamentos)} puntos.")

    print("Generando gráficas (PDF)...")
    generar_graficas(coords, labels, clusters, voids, filamentos)

    print("Escribiendo catálogos y estadísticas...")
    escribir_catalogos(clusters, voids, filamentos, ids, labels, dens, med)

    t1 = time.time()
    print(f"Análisis completado en {t1 - t0:.1f} s.")
    print(f"Archivos generados:\n - {OUT_CATALOGO_CLUSTERS}\n - {OUT_CATALOGO_VOIDS}\n - {OUT_TRAZADO_FILAMENTOS}\n - {OUT_ESTADISTICAS}\n - {PDF_DISTRIBUCION}\n - {PDF_CLASIFICADAS}\n - {PDF_CLUSTERS_VOIDS}\n - {PDF_RED_COMPLETA}")

if __name__ == '__main__':
    main()
