import os
import re
import pkg_resources

# Carpeta del proyecto
project_folder = "."

# Patrón para detectar imports
import_pattern = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)")

found_packages = set()

for root, _, files in os.walk(project_folder):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    match = import_pattern.match(line)
                    if match:
                        found_packages.add(match.group(1))

# Generar requirements.txt con las versiones actuales
with open("requirements.txt", "w") as f:
    for pkg in sorted(found_packages):
        try:
            version = pkg_resources.get_distribution(pkg).version
            f.write(f"{pkg}=={version}\n")
        except pkg_resources.DistributionNotFound:
            # Ignorar módulos de la stdlib
            print(f"Advertencia: {pkg} no está instalado o es de la librería estándar")
