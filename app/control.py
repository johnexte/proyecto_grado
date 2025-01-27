from . import config
import os

def read_file(file_name: str, path: str) -> list:
    """ Function to read a file and return its content as a list.
    Args:
        file_name (_type_): file name to read
        path (_type_): path where the file is located
    Returns:
        _type_: list with the content of the file
    """
    lines = []
    with open(os.path.join(path, file_name), "r") as file:
        file.readline()
        for line in file:
            lines.append(line.split(","))
    return lines

def new_folder(path: str, folder_name: str) -> str:
    """ Function to create a new folder in a given path.
    Args:
        path (_type_): path to create the folder
        folder_name (_type_): name of the folder to create
    Returns:
        _type_: path of the new folder
    """
    new_path = os.path.join(path, folder_name)
    os.makedirs(new_path, exist_ok=True)
    return new_path
    
def clear() -> None:
    """ Function to clear the console screen. """
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")
        
def empty_results_folder(results: str) -> None:
    """ Function to delete the results folder and its content. 
    Args:
        results (_type_): path to the results folder
    """
    opc = input("¿Está seguro que desea eliminar la carpeta de resultados? (S/n): ")
    if opc == "S":
        for folder in os.listdir(results):
            folder_path = f'"{os.path.join(results, folder)}"'
            delete_path(folder_path)
        print("Carpeta de resultados eliminada")
    else:
        print("Operación cancelada")

def empty_folder(path: str) -> None:
    """ Function to delete the content of a folder.
    Args:
        path (_type_): path to the folder to empty
    """
    for file in os.listdir(path):
        file_path = f'"{os.path.join(path, file)}"'
        delete_path(file_path)
        
def delete_path(path: str) -> None:
    """ Function to delete a file or folder.
    Args:
        path (_type_): path to the file or folder to delete
    """          
    if os.name == "nt":
                os.system(f"del /f /q {path}")
                os.system(f"rmdir /s /q {path}")
    else:
        os.system(f"rm -rf {path}")
    