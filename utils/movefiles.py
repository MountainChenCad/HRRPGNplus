import os
import re
import random
from collections import defaultdict

# Directorio que contiene los archivos .mat
# Cambia esta ruta al directorio donde están tus archivos
directory = "../data/test"

# Obtener todos los archivos .mat en el directorio
mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]

# Agrupar archivos por categoría
category_files = defaultdict(list)
for file in mat_files:
    # Extraer el nombre de la categoría (parte antes del primer guion bajo)
    match = re.match(r'([^_]+)_', file)
    if match:
        category = match.group(1)
        category_files[category].append(file)

# Para cada categoría, mantener 200 archivos y eliminar el resto
for category, files in category_files.items():
    print(f"Categoría: {category}, Archivos totales: {len(files)}")

    if len(files) <= 200:
        print(f"  Manteniendo todos los {len(files)} archivos (menos o igual a 200)")
        continue

    # Seleccionar aleatoriamente los archivos a mantener
    files_to_keep = random.sample(files, 200)
    files_to_delete = [f for f in files if f not in files_to_keep]

    print(f"  Manteniendo 200 archivos, eliminando {len(files_to_delete)} archivos")

    # Eliminar los archivos excedentes
    for file_to_delete in files_to_delete:
        file_path = os.path.join(directory, file_to_delete)
        try:
            os.remove(file_path)
            # Descomenta la línea de abajo para ver cada archivo que se elimina
            # print(f"  Eliminado: {file_to_delete}")
        except Exception as e:
            print(f"  Error al eliminar {file_to_delete}: {e}")

print("¡Proceso completado!")