from classes.simulation import simulation as sim
from classes import plot
import numpy as np
import pandapower as pp
from . import control
from . import config
import os
import json
import pandas as pd
from pandapower.networks.power_system_test_cases import _get_cases_path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

case = 'case39'
n = 100
weight = 1
join_case = "d2d"
results = None
tripped_lines = [(0,38),(23,36)]

control.new_folder(config.results_dir, case)
control.new_folder(os.path.join(config.results_dir, case), 'plots')

def set_case():
    control.new_folder(config.results_dir, case)
    control.new_folder(os.path.join(config.results_dir, case), 'plots')

def run_simulation():
    simulation = sim()
    return simulation.cascading_event_simulation(os.path.join(config.results_dir, case, 'power_grid.png'), case, n, weight, join_case, tripped_lines)

def save_plots(hist, comms_graph, power_graph, edges):
    fig1, fig2, fig3, fig4, fig0 = plot.plot_results(hist)
    fig1.savefig(os.path.join(config.results_dir, case, 'plots', 'bus_consequences.png'))
    fig2.savefig(os.path.join(config.results_dir, case, 'plots', 'line_consequences.png'))
    fig3.savefig(os.path.join(config.results_dir, case, 'plots', 'bus_risks.png'))
    fig4.savefig(os.path.join(config.results_dir, case, 'plots', 'line_risks.png'))
    fig0.savefig(os.path.join(config.results_dir, case, 'plots', 'line_probs.png'))
    fig5, fig6 = plot.plot_communications_graph(comms_graph, power_graph, edges, "scale-free", n, case)
    fig5.savefig(os.path.join(config.results_dir, case, 'plots', 'comms_graph.png'))
    fig6.savefig(os.path.join(config.results_dir, case, 'plots', 'graphs.png'))
    plt.close('all')

def save_vars_json(results):
    filtered_vars = {}
    for key, value in results.items():
        # Convertir numpy arrays a listas y redes de pandapower a diccionarios
        value = convert_values(value)
        # Verifica si la variable es serializable
        if is_serializable(value):
            filtered_vars[key] = value
    unique_filename = get_unique_filename(os.path.join(config.results_dir, case), 'variables.json')
    with open(os.path.join(config.results_dir, case,  'variables.json'), 'w') as f:
        json.dump(filtered_vars, f, indent=4)
    input("Press Enter to continue...")
    
def show_3d_plot(stage: int, show_plot: bool = False):
    path = os.path.join(config.results_dir, case, 'power_grid.png')
    fig = plot.plot_geographical_distribution(results, stage, case, 'cubic', path, show_plot)
    fig.write_image(os.path.join(config.results_dir, case, 'plots', '3d_plot.png'))

def run_all():
    results, power_graph, comms_graph, graphs_edges, hist = run_simulation()
    results["historial"] = hist
    save_plots(hist, comms_graph, power_graph, graphs_edges)
    save_vars_json(results)
    figs = plot.plot_coms_interaction(comms_graph, power_graph, graphs_edges, results)
    create_stop_motion(figs, os.path.join(config.results_dir, case, 'plots', 'stop_motion.mp4'))
   
def is_serializable(value):
    try:
        json.dumps(value)
        return True
    except Exception as e:
        print(f"Value {value} is not serializable, error: {e}")
        return False

def handle_nan(data):
        # Si es un diccionario, recorrer los valores
        if isinstance(data, dict):
            return {k: handle_nan(v) for k, v in data.items()}
        # Si es una lista, recorrer los elementos
        elif isinstance(data, list):
            return [handle_nan(v) for v in data]
        # Si es un valor Numérico, revisamos si es NaN
        elif isinstance(data, (float, int)) and pd.isna(data):
            return None
        return data

def convert_values(obj):
    if isinstance(obj, dict):
        return {str(k): convert_values(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_values(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return convert_values(obj.tolist())
    elif hasattr(obj, 'toarray'):  # Sparse matrix
        # Convertir sparse matrix a lista de listas
        return convert_values(obj.toarray().tolist())
    elif isinstance(obj, pp.pandapowerNet):
        # Convertir las tablas relevantes de pandapower en diccionarios
        obj = {
            "bus": handle_nan(obj.bus.to_dict()),
            "line": handle_nan(obj.line.to_dict()),
            "gen": handle_nan(obj.gen.to_dict()),
            "load": handle_nan(obj.load.to_dict()),
            "trafo": handle_nan(obj.trafo.to_dict()),
            "bus_geodata": handle_nan(obj.bus_geodata.to_dict()),
            "shunt": handle_nan(obj.shunt.to_dict()),
            "ext_grid": handle_nan(obj.ext_grid.to_dict()),
            "res_bus": handle_nan(obj.res_bus.to_dict()),
            "res_line": handle_nan(obj.res_line.to_dict()),
            "res_trafo": handle_nan(obj.res_trafo.to_dict()),
            "res_gen": handle_nan(obj.res_gen.to_dict()),
            "res_load": handle_nan(obj.res_load.to_dict()),
            "res_shunt": handle_nan(obj.res_shunt.to_dict()),
            "res_ext_grid": handle_nan(obj.res_ext_grid.to_dict())}
    elif isinstance(obj, pd.DataFrame):
        return convert_values(obj.to_dict())  # Convierte DataFrame a diccionario
    elif isinstance(obj, pd.Index):
        return convert_values(obj.tolist())  # Convierte Index a lista
    elif isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    elif isinstance(obj, float) and np.isnan(obj):
        return None  # Convertir NaN a None (null en JSON)
    else:
        return obj

       
def get_unique_filename(base_path, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(base_path, new_filename)):
        new_filename = f"{base} ({counter}){ext}"
        counter += 1
    return new_filename
        
def update_test_cases_file():
    path = _get_cases_path()
    with open(os.path.join(config.anexos_dir, 'test_cases.txt'), 'w') as f:
        for i, filename in enumerate(os.listdir(path)):
            f.write(f"{i}. {filename.replace(".json","")}\n")

def create_stop_motion(figures, output_file):
    """
    Crea un video stop motion a partir de una lista de figuras de Matplotlib.

    :param figures: Lista de objetos Figure de Matplotlib.
    :param output_file: Nombre del archivo de salida (debe terminar en .mp4).
    :param frame_time: Tiempo entre cuadros en segundos.
    :param resolution: Resolución del video en formato (ancho, alto).
    :param dpi: Densidad de puntos por pulgada al guardar las figuras.
    """
    dpi = 450
    frames = []
    for fig in figures:
        for ax in fig.get_axes():
            ax.set_axis_off()  # Oculta los ejes
            
        fig.canvas.draw()
        # Convertir la figura en una imagen RGBA
        image = fig.canvas.buffer_rgba()
        frames.append(image)

    # Crear una figura base para la animación
    fig, ax = plt.subplots(figsize=(15, 10),dpi=dpi)
    ax.axis('off')

    # Inicializar con un cuadro vacío
    img = ax.imshow(frames[0], animated=True)

    def init():
        img.set_data(frames[0])
        return img,

    def update(frame):
        img.set_data(frames[frame])
        return img,

    # Crear la animación
    animation = FuncAnimation(
        fig, update, frames=len(frames), init_func=init, interval=2000, blit=True
    )

    # Guardar la animación como MP4
    writer = FFMpegWriter(fps=1)  # 0.5 FPS = 1 frame cada 2 segundos
    animation.save(output_file, writer=writer, dpi=dpi)

    plt.close('all')
