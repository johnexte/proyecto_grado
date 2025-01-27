from matplotlib                 import pyplot as plt
from seaborn                    import barplot
from scipy.interpolate          import griddata
from skimage                    import io
import matplotlib.patches       as mpatches
import pandas                   as pd
import numpy                    as np
import skimage.io               as sio
import plotly.graph_objects     as go
import networkx                 as nx
from pandapower import plotting as pp
from .graph import comms_graph, power_graph

def plot_results(results: dict, lim: int = 3):
    """
    Función que genera los gráficos de la red de potencia.
    """
    # Gráfica 1: Stacked Buses Consequences
    fig1, ax_bus_consequences = plt.subplots(figsize=(15, 7))
    data = []
    if len(results["bus_consequences"]) < lim:
        rango = range(len(results["bus_consequences"]))
    else:
        rango = range(lim)
        
    for i in rango:
        stage = f"Stage {i}"
        consequences = list(results["bus_consequences"][i].values())
        topological_consequences = list(results["topological_consequences"][i].values())
        electrical_consequences = list(results["electrical_consequences"][i].values())
        topological_consequences = topological_consequences / np.linalg.norm(consequences)
        electrical_consequences = electrical_consequences / np.linalg.norm(consequences)
        for node in range(len(consequences)):
            data.append({
                "Node": node + 1,
                "Stage": stage,
                "Topological": topological_consequences[node],
                "Electrical": electrical_consequences[node]
            })
    df = pd.DataFrame(data)
    nodes = df["Node"].unique()
    width = 1/len(df["Stage"].unique())
    x = np.arange(len(nodes))

    for i in rango:
        stage = f"Stage {i}"
        stage_data = df[df["Stage"] == stage]
        topological = stage_data["Topological"]
        electrical = stage_data["Electrical"]
        ax_bus_consequences.bar(x + i * width, topological, width-0.1, label=f"{stage} - Topological")
        ax_bus_consequences.bar(x + i * width, electrical, width-0.1, bottom=topological, label=f"{stage} - Electrical")
    ax_bus_consequences.set_xticks(x + width * (len(df["Stage"].unique()) - 1) / 2)
    ax_bus_consequences.set_xticklabels(nodes)
    ax_bus_consequences.set_xlabel("Nodes")
    ax_bus_consequences.set_ylabel("Consequences (Normalized)")
    ax_bus_consequences.set_title("Node Consequences by Stage")
    ax_bus_consequences.legend(title="Consequences")
    plt.tight_layout()
    
    # Gráfica 2: Line Consequences
    data = []
    for i in rango:
        line_consequences = results["line_consequences"][i]
        stage = f"Stage {i}"
        for line, value in line_consequences.items():
            data.append({"Line": str((line[0] + 1, line[1] + 1)), "Consequence": value, "Stage": stage})
    df = pd.DataFrame(data)
    fig2 = plt.figure(figsize=(15, 7))
    barplot(
        data=df, 
        x="Line", 
        y="Consequence", 
        hue="Stage", 
        palette="Set2",
        gap=0.1
    )
    plt.xlabel("Línea")
    plt.ylabel("Consecuencia")
    plt.title("Consecuencias de las Líneas de la Red de Potencia")
    plt.xticks(rotation=90)
    plt.legend(title="Stage")
    plt.tight_layout()

    # Gráfica 5: Line Probabilities
    data = []
    for i in rango:
        line_consequences = results["line_probs"][i]
        stage = f"Stage {i}"
        for line, value in line_consequences.items():
            data.append({"Line": str((line[0] + 1, line[1] + 1)), "Probability": value, "Stage": stage})
    df = pd.DataFrame(data)
    fig5 = plt.figure(figsize=(15, 7))
    barplot(
        data=df, 
        x="Line", 
        y="Probability", 
        hue="Stage", 
        palette="Set2",
        gap=0.1
    )
    plt.xlabel("Línea")
    plt.ylabel("Probabilidad")
    plt.title("Probabilidad de falla de las Líneas de la Red de Potencia")
    plt.xticks(rotation=90)
    plt.legend(title="Stage")
    plt.tight_layout()
    
    # Gráfica 3: Buses Risk
    fig3 = plt.figure(figsize=(15, 7))  
    data = []
    for i in rango:
        risks = list(results["risks"][i].values())
        stage = f"Stage {i}"
        for node, risk in enumerate(risks, start=1):
            data.append({"Node": node, "Risk": risk, "Stage": stage})

    df = pd.DataFrame(data)
    barplot(
        data=df, 
        x="Node", 
        y="Risk", 
        hue="Stage", 
        palette="Set2",
        gap=0.1
    )
    plt.xlabel("Nodo")
    plt.ylabel("Riesgo")
    plt.title("Riesgos de los Nodos de la Red de Potencia")
    plt.xticks(range(len(results["risks"])), [str(i + 1) for i in range(len(results["risks"]))])
    plt.legend(title="Stage", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
        
    # Gráfica 4: Lines Risk
    fig4 = plt.figure(figsize=(15, 7))
    data = []
    for i in rango:
        line_risks = results["line_risks"][i]
        stage = f"Stage {i}"
        for line, risk in line_risks.items():
            data.append({"Line": str((line[0] + 1, line[1] + 1)), "Risk": risk, "Stage": stage})
    df = pd.DataFrame(data)
    barplot(
        data=df, 
        x="Line", 
        y="Risk", 
        hue="Stage", 
        palette="Set2",
        gap=0.1
    )
    plt.xlabel("Línea")
    plt.ylabel("Riesgo")
    plt.title("Riesgos de las Líneas de la Red de Potencia")
    plt.xticks(rotation=90)
    plt.legend(title="Stage", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    return fig1, fig2, fig3, fig4, fig5
    
def plot_geographical_distribution(results: tuple, stage: int, case: str, interpol: str, path: str, show_plot: bool = True):
    net = results
    
    buses_risk = results["risks"][stage]
    lines_risk = results["line_risks"][stage]
    
    buses_coords = net.bus_geodata.copy()
    buses_coords['risk'] = buses_risk
    buses_coords['id'] = buses_coords.index + 1
    del buses_coords["coords"]
    lines_coords = pd.DataFrame()
    for start, end in lines_risk:
        if start in buses_coords.index and end in buses_coords.index:
            coord = ((buses_coords.loc[start, 'x'] + buses_coords.loc[end, 'x']) / 2, 
                     (buses_coords.loc[start, 'y'] + buses_coords.loc[end, 'y']) / 2)
            lines_coords = pd.concat([lines_coords, pd.DataFrame([{'x': coord[0], 'y': coord[1], 'id': (start, end), 'risk': lines_risk[(start, end)]}])], ignore_index=True)
    
    coordinates = pd.concat([buses_coords, lines_coords])

    
    # Crear una malla para interpolar los datos
    x = np.array(coordinates.x)
    y = np.array(coordinates.y)
    z = np.array(coordinates.risk)

    xi = np.linspace(min(x), max(x), 1000) 
    yi = np.linspace(min(y), max(y), 800)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolación cúbica (spline)
    
    zi = griddata((x, y), z, (xi, yi), method=interpol, rescale=True)

    image = sio.imread (path, as_gray=True)
    io.imshow(image)


    fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Jet')])
    
    fig.add_surface(x=xi, y=yi, z=np.ones(zi.shape),
                    surfacecolor=image,
                    colorscale='gray',
                    showscale=False)         

    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

    fig.update_layout(title=dict(text='Steady state risk geographical distribution'),
                      width=800, height=1000,
                      autosize = True)
    if show_plot:
        fig.show()
    return fig

def plot_communications_graph(comms_graph: comms_graph, power_graph: power_graph, edges, type, n, case):
    pos_G1 = nx.spring_layout(power_graph.graph.to_undirected(), seed=42, weight=None)  # Layout para G1
    pos_G2 = nx.spring_layout(comms_graph.graph.to_undirected(), seed=42, weight=None)  # Layout para G2
    
    fig1 = plt.figure(figsize=(15, 7))
    nx.draw(comms_graph.graph, pos=pos_G2, with_labels=True, font_weight='bold')
    plt.close()
    
    fig2 = plt.figure(figsize=(15, 7))

    # Paso 3: Posicionar los vértices
    # Posicionamos los vértices de G1 a la izquierda y de G2 a la derecha
    

    # Modificamos la posición de los vértices de G1 para que estén a la izquierda
    for node in pos_G1:
        pos_G1[node] = (pos_G1[node][0] - 1, pos_G1[node][1])

    # Modificamos la posición de los vértices de G2 para que estén a la derecha
    for node in pos_G2:
        pos_G2[node] = (pos_G2[node][0] + 1, pos_G2[node][1])

    # Dibuja G1
    nx.draw(power_graph.graph, pos=pos_G1, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_weight='bold', label="Power Grid for case " + case)
    # Dibuja G2
    node_colors = ['yellow' if node == comms_graph.centre_node.node_id else 'lightgreen' for node in comms_graph.nodes]
    nx.draw(comms_graph.graph, pos=pos_G2, with_labels=True, node_color=node_colors, node_size=500, font_size=10, font_weight='bold', label=f"{type} communications graph with n {n}")
    
    # Paso 5: Graficar los arcos punteados que unen los dos grafos
    for (v1, v2) in edges:
        pos1 = pos_G1[v1]
        pos2 = pos_G2[v2]
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k:', lw=1, zorder=0)  # 'k:' para línea punteada

    # Mostrar el gráfico
    plt.title("Grafos G1 y G2 con Conexiones Punteadas")
    plt.axis('off')  # Apagar los ejes para mejor visualización
    legend_labels = [
        mpatches.Patch(color='skyblue', label=f"Power Grid for {case}"),
        mpatches.Patch(color='lightgreen', label=f"{type.capitalize()} communications graph with n {n}"),
        mpatches.Patch(color='yellow', label="Control Centre")
    ]
    # Añadir la leyenda
    #plt.legend(loc="upper center", handles=legend_labels)
    return fig1, fig2

def plot_coms_interaction(comms_graph: comms_graph, power_graph: power_graph, edges, results: dict, plot_type="normal"):
    figs = []
    i = 0
    for key, result in results.items():
        if type(key) != int:
            continue
        i += 1
        fig = plt.figure(figsize=(15, 10))

        if  plot_type == "example":
            pos_G1 = {node: np.array([-0.1, -i]) for i, node in enumerate(power_graph.graph.nodes)} 
            pos_G2 = {node: np.array([0.1, -i]) for i, node in enumerate(comms_graph.graph.nodes)}
        elif plot_type == "normal":
            pos_G1 = nx.spring_layout(power_graph.graph.to_undirected(), seed=42, weight=None)  # Layout para G1
            pos_G2 = nx.spring_layout(comms_graph.graph, seed=42, weight=None)  # Layout para G2

            # Modificamos la posición de los vértices de G1 para que estén a la izquierda
            for node in pos_G1:
                pos_G1[node] = (pos_G1[node][0] - 1, pos_G1[node][1])

            # Modificamos la posición de los vértices de G2 para que estén a la derecha
            for node in pos_G2:
                pos_G2[node] = (pos_G2[node][0] + 1, pos_G2[node][1])
        
        # Dibuja G1
        node_colors = ['red' if node in result["buses_failed"] else 'skyblue' for node in power_graph.graph.nodes]
        if plot_type == "example":
            nx.draw_networkx_edges(power_graph.graph, pos=pos_G1, arrows=True,
                            connectionstyle="arc3,rad=0.5")
        else:

            edges_to_draw = [edge for edge in power_graph.graph.edges if edge not in result["lines_triped"]]
            edge_color = ['red' if edge == result["vulnerable"] else 'black' for edge in power_graph.graph.edges]
            nx.draw_networkx_edges(power_graph.graph, pos=pos_G1, arrows=True, edgelist=edges_to_draw, edge_color=edge_color)
        nx.draw_networkx_nodes(power_graph.graph, pos=pos_G1, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(power_graph.graph, pos=pos_G1, labels={node: node for node in power_graph.graph.nodes}, font_size=10, font_weight='bold')
        
        # Dibuja G2
        node_colors = []
        for node in comms_graph.graph.nodes:
            if node in result["nodes_with_packets"]:
                node_colors.append('red')
            elif node == comms_graph.centre_node.node_id:
                node_colors.append('yellow')
            else:
                node_colors.append('lightgreen')
        if plot_type == "example":
            nx.draw_networkx_edges(comms_graph.graph, pos=pos_G2, arrows=True,
                        connectionstyle="arc3,rad=0.5")
        else:
            nx.draw_networkx_edges(comms_graph.graph, pos=pos_G2, arrows=True)
        nx.draw_networkx_nodes(comms_graph.graph, pos=pos_G2, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(comms_graph.graph, pos=pos_G2, labels={node: node for node in comms_graph.graph.nodes}, font_size=10, font_weight='bold')
        
        # Paso 5: Graficar los arcos punteados que unen los dos grafos
        for (v1, v2) in edges:
            pos1 = pos_G1[v1]
            pos2 = pos_G2[v2]
            plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k:', lw=1, zorder=0)  # 'k:' para línea punteada

        # Mostrar el gráfico
        titulo = f"i = {i}"
        if result["tij"]:
            titulo += f" t = {result["t"]}"
            titulo += f" tij = {result["tij"]}"
        titulo += "\n"
        titulo += f"Buses con falla = {result['buses_failed']}\n"
        if result["action"] == "move":
            titulo += f"Paquete enviado de nodo {result['current_node']} a nodo {result['next_node']}"
        elif result["action"] == "start":
            titulo += f"Paquete iniciado en el nodo {result['started_node']} con destino {result['destination_node']}"
        elif result["action"] == "delivered":
            titulo += f"Paquete entregado, se arregla el nodo {result['bus_repaired']}"
        elif result["action"] == "tripped":
            titulo += f"Se atacó la linea {result["line_tripped"]}"
        for node in result["queues"]:
            titulo += f"\nPaquetes en nodo {node} = {result['queues'][node]}"

        plt.title(titulo)
            
        plt.axis('off')  # Apagar los ejes para mejor visualización
        legend_labels = [
            mpatches.Patch(color='skyblue', label=f"Power Grid"),
            mpatches.Patch(color='lightgreen', label=f"communications graph"),
            mpatches.Patch(color='yellow', label="Control Centre")
        ]
        # Añadir la leyenda
        plt.legend(handles=legend_labels)
        figs.append(fig)
    return figs

def plot_net(power_graph, path, case):
    pos_G1 = nx.spring_layout(power_graph.graph.to_undirected(), seed=42, weight=None)  # Layout para G1
    nx.draw(power_graph.graph, pos=pos_G1, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_weight='bold', label="Power Grid for case " + case)
    # pp.simple_plot(net, plot_loads=True, plot_sgens=True, plot_gens=True, ext_grid_color='r', show_plot=False)
    plt.savefig(path)
    plt.close('all')