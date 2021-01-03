from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from pandas import read_csv, DataFrame
import math
import os
import time


class AugeratNodePosition:
    """Stores coordinates of a node of Augerat's instances (set P)."""

    def __init__(self, values):
        # Node ID
        self.name = np.uint32(values[0]).item()
        if self.name == 1:
            self.name = "Source"
        # x coordinate
        self.x = np.float64(values[1]).item()
        # y coordinate
        self.y = np.float64(values[2]).item()


class AugeratNodeDemand:
    """Stores attributes of a node of Augerat's instances (set P)."""

    def __init__(self, values):
        # Node ID
        self.name = np.uint32(values[0]).item()
        if self.name == 1:
            self.name = "Source"
        # demand coordinate
        self.demand = np.float64(values[1]).item()


class DataSet:
    """Reads an Augerat instance and stores the network as DiGraph.

    Args:
        path (str) : Path to data folder.
        instance_name (str) : Name of instance to read.
    """

    def __init__(self, path, instance_name):
        self.data = {}

        # Read vehicle capacity
        with open(path + instance_name) as fp:
            for i, line in enumerate(fp):
                if i == 1:
                    best = line.split()[-1][:-1]
                    self.best_known_solution = int(best)
                if i == 5:
                    self.max_load = int(line.split()[2])
        fp.close()

        # Read nodes from txt file
        if instance_name[5] == "-":
            self.n_vertices = int(instance_name[3:5])
        else:
            self.n_vertices = int(instance_name[3:6])
        df_augerat = read_csv(
            path + instance_name,
            sep="\t",
            skiprows=6,
            nrows=self.n_vertices,
        )
        # Scan each line of the file and add nodes to the network
        self.data["locations"] = []
        for line in df_augerat.itertuples():
            values = line[1].split()
            node = AugeratNodePosition(values)
            self.data["locations"].append((node.x, node.y))

        # Read demand from txt file
        df_demand = read_csv(
            path + instance_name,
            sep="\t",
            skiprows=range(7 + self.n_vertices),
            nrows=self.n_vertices,
        )
        self.data["demands"] = []
        for line in df_demand.itertuples():
            values = line[1].split()
            node = AugeratNodeDemand(values)
            self.data["demands"].append(node.demand)

        # vehicles
        self.data["num_vehicles"] = self.n_vertices
        self.data["vehicle_capacities"] = [self.max_load] * self.n_vertices
        self.data["depot"] = 0

    def compute_euclidean_distance_matrix(self, locations):
        """2D Euclidian distance between two nodes"""
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    # Euclidean distance
                    distances[from_counter][to_counter] = int(
                        math.hypot(
                            (from_node[0] - to_node[0]), (from_node[1] - to_node[1])
                        )
                    )
        return distances

    def print_solution(self, manager, routing, solution):
        """Prints solution on console."""
        total_distance = 0
        total_load = 0
        for vehicle_id in range(self.data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            plan_output = "Route for vehicle {}:\n".format(vehicle_id)
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += self.data["demands"][node_index]
                plan_output += " {0} Load({1}) -> ".format(node_index, route_load)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += " {0} Load({1})\n".format(
                manager.IndexToNode(index), route_load
            )
            plan_output += "Distance of the route: {}m\n".format(route_distance)
            plan_output += "Load of the route: {}\n".format(route_load)
            # if route_load > 0:
            #    print(plan_output)
            total_distance += route_distance
            total_load += route_load
        # print('Total distance of all routes: {}m'.format(total_distance))
        # print('Total load of all routes: {}'.format(total_load))
        return total_distance

    def main(self, option):
        """Solve the CVRP problem."""

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            self.n_vertices, self.data["num_vehicles"], self.data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        distance_matrix = self.compute_euclidean_distance_matrix(self.data["locations"])

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return self.data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = option

        # search_parameters.local_search_metaheuristic = (
        #    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        # )
        search_parameters.time_limit.seconds = 10

        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            best_value = self.print_solution(manager, routing, solution)
        else:
            best_value = None
        return best_value


if __name__ == "__main__":
    keys = [
        "instance",
        "nodes",
        "algorithm",
        "res",
        "best known solution",
        "gap",
        "time (s)",
        "vrp",
        "time limit (s)",
    ]
    instance = []
    nodes = []
    alg = []
    res = []
    best_known_solution = []
    gap = []
    run_time = []
    vrp = []
    time_limit = []
    for option in [routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC]:
        print("")
        print("===============")
        for file_name in os.listdir("./data/"):
            if file_name[-3:] == "vrp":  # and file_name == "A-n32-k5.vrp":
                print(file_name)
                data = DataSet(path="./data/", instance_name=file_name)
                instance.append(file_name)
                nodes.append(data.n_vertices)
                best_known_solution.append(data.best_known_solution)
                alg.append("ortools, path cheapest arc")
                vrp.append("cvrp")
                time_limit.append(10 * 1)

                start = time.time()
                best_value = data.main(option)
                res.append(best_value)
                if best_value:
                    gap.append(
                        (best_value - data.best_known_solution)
                        / data.best_known_solution
                        * 100
                    )
                else:
                    gap.append(None)
                run_time.append(float(time.time() - start))

                values = [
                    instance,
                    nodes,
                    alg,
                    res,
                    best_known_solution,
                    gap,
                    run_time,
                    vrp,
                    time_limit,
                ]
                df = DataFrame(dict(zip(keys, values)), columns=keys)
                df.to_csv("ortools_cvrp_augerat.csv", sep=";", index=False)
