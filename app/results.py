import json
import os
import config
import matplotlib.pyplot as plt 

def plot_case118():
    case118 = json.load(open(os.path.join(config.results_dir, "case118", "variables.json")))
    line_consequences = case118["line_consequences"]
    bins = [i * 0.02 for i in range(11)]
    plt.hist(line_consequences.values(), bins=bins, edgecolor='black')
    plt.xlabel('Consecuencias (agrupadas en rangos de 0.02)')
    plt.ylabel('Número de líneas')
    plt.title('Histograma de Consecuencias por Línea')
    plt.xticks([i * 0.05 for i in range(5)])
    plt.yticks([i * 20 for i in range(9)])
    print(len([val for val in line_consequences.values() if val >= 0 and val <= 0.02]))
    print(len(line_consequences.values()))
    print([val for val in line_consequences.values() if val <= 0])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.xlim(0, 0.2)  # Establecer el rango del eje x
    plt.savefig("figure.png",dpi=300)
    
def plot_case(case, labels):
    casen = json.load(open(os.path.join(config.results_dir,case, "variables.json")))
    line_consequences = casen["historial"]["line_consequences"][0]
    plt.bar(line_consequences.keys(),line_consequences.values())
    plt.xlabel("Lineas")
    plt.ylabel("Consecuencia")
    plt.xticks(rotation=90)
    if not labels:
        plt.xticks([])
    plt.savefig("figure.png",dpi=300)
    # plt.show()
    
    
plot_case118()
# plot_case("case118",labels=False)