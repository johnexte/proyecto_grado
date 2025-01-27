import os
import sys
file_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(file_path))
file_dir = os.path.dirname(os.path.realpath('__file__'))
results_dir = file_dir + '/results/'
anexos_dir = file_dir + '/anexos/'
example_dir = file_dir + '/example/'

import subprocess

def check_install_libraries(libraries):
    for lib in libraries:
        try:
            __import__(lib)
        except ImportError:
            print(f"{lib} is not installed. Installing...")
            try:
                subprocess.check_call(["python3", "-m", "pip", "install", lib])
                print(f"{lib} has been successfully installed.\n")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {lib}. Error: {e}")

# Lista de bibliotecas a verificar e instalar si es necesario
libraries_to_check = ["pandapower", 
                    "matplotlib", 
                    "networkx",
                    "tabulate",
                    "numpy",
                    "scipy",
                    "pandas",
                    "seaborn",
                    "skimage",
                    "plotly"]
                        

# Llamar a la función para verificar e instalar las bibliotecas
check_install_libraries(libraries_to_check)
