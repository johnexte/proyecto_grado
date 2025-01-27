from . import config
from . import control as control
from . import model

main_menu_options = [
    "Run all",
    "Set case",
    "Run simulation",
    "Delete results folder",
    "Save plots",
    "Show 3d plot",
    "Set communications graph",
    "Save communications graph",
    "Save variables to json", 
    "Run example"
]

def printMenu(menu_options)-> None:
    print("Menu Options:")
    for i, option in enumerate(menu_options, 1):
        print(f"    {i}. {option}")
    print("    0. Exit")

def executeOption(option: int):
    options = {
        1: lambda: model.run_all(),
        2: lambda: set_case(),
        3: lambda: model.run_simulation(),
        4: lambda: control.empty_results_folder(config.results_dir),
        5: lambda: model.save_plots(),
        6: lambda: show_3d_plot(),
        7: lambda: set_communications_graph(),
        8: lambda: save_communications_graph(),
        9: lambda: model.save_vars_json(),
        10: lambda: model.run_example()
        }
    return options.get(option, lambda: print("Invalid option\n"))()

def set_case():
    case = input("Enter the case name (default: case14): ")
    n = int(input("Enter comms graph size (comms graph > power graph): "))
    try:
        model.case = case
        model.n = n
        model.set_case()
    except Exception as e:
        print(f"Error: {e}")
        
def show_3d_plot():
    try:
        stage = int(input("Enter the stage number: "))
        model.show_3d_plot(stage, True)
    except Exception as e:
        print(f"Error: {e}")
        print("Please enter a valid stage number")
        
def set_communications_graph():
    # try:
    n = int(input("Enter the number of nodes: "))
    default = input("Use default values? (y/n): ")
    if default == "y":
        comms_graph_type = "scale-free"
        weight = 1
        join_type = "degree_to_degree"
    else:
        comms_graph_type = input("Enter the communications graph type (scale-free/random): ")
        weight = float(input("Enter the weight: "))
        join_type = input("Enter the join type (degree_to_degree/random): ")
    
    model.set_communications_graph(comms_graph_type, n, weight, join_type)
        
def save_communications_graph():
    try:
        model.save_communications_graph()
    except Exception as e:
        print(f"Error: {e}")
        print("No communications graph to save")

def main() -> None:
    control.clear()
    model.update_test_cases_file()
    while True:        
        printMenu(main_menu_options)
        print("Case: ", model.case)
        try:
            option = int(input("Select an option: "))
            if option == 0:
                break
            executeOption(option)
        except ValueError as e:
            print(f"Error: {e}")
            
        input("Press Enter to continue...")
        control.clear()
        
if __name__ == "__main__":
    main()
        
    