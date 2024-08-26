# from ortools.constraint_solver import pywrapcp, routing_enums_pb2
# from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
# # OR-Tools code for Clarke-Wright
# def create_data_model(No, N, distance_matrix, n_vehicles, depot=0):
#     data = {}
#     data['distance_matrix'] = distance_matrix.tolist()  # Convert numpy array to list
#     data['num_vehicles'] = n_vehicles
#     data['depot'] = depot  # Depot is node 0
#     return data

# def print_solution(manager, routing, solution):
#     print('Objective: {} miles'.format(solution.ObjectiveValue()))
#     index = routing.Start(0)
#     route_distance = 0
#     while not routing.IsEnd(index):
#         print(manager.IndexToNode(index), end=" -> ")
#         previous_index = index
#         index = solution.Value(routing.NextVar(index))
#         route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
#     print(manager.IndexToNode(index))
#     print('Route distance: {} miles'.format(route_distance))

# def main():
#     generator = InstanceGenerator(n_customers=20, n_vehicles=3, max_demand=10, max_distance=50)
#     No, N, M, demand, load_capacity, distance, mst_baseline_value, mst_baseline_route, distance_matrix, coordinates = generator.instanSacy()
#     n_vehicles = 3


#     data = create_data_model(No, N, distance_matrix, n_vehicles)

#     manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
#     routing = pywrapcp.RoutingModel(manager)

#     def distance_callback(from_index, to_index):
#         from_node = manager.IndexToNode(from_index)
#         to_node = manager.IndexToNode(to_index)
#         return data['distance_matrix'][from_node][to_node]

#     transit_callback_index = routing.RegisterTransitCallback(distance_callback(0,20))
#     routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

#     search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#     search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS

#     solution = routing.SolveWithParameters(search_parameters)
#     if solution:
#         print_solution(manager, routing, solution)
#     else:
#         print('No solution found!')

# if __name__ == '__main__':
#     main()

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = []
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output.append(manager.IndexToNode(index))
    print(plan_output)
    print('Route distance: {} miles'.format(route_distance))
    
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = [
        [0, 29, 20, 21],
        [29, 0, 15, 17],
        [20, 15, 0, 28],
        [21, 17, 28, 0],
    ]
    data['demands'] = [0, 1, 1, 2,5,6,3,8,9,0]  # demands per customer
    data['vehicle_capacities'] = [4, 4]  # vehicle capacities
    data['num_vehicles'] = 2  # number of vehicles
    data['depot'] = 0  # depot index
    return data

def main():
    """Solve the CVRP using Clarke-Wright Savings Algorithm."""
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Clarke-Wright Savings Algorithm
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print_solution(manager, routing, solution)
        print("Solution found!")

if __name__ == '__main__':
    main()
