import numpy as np


class ChargingNetwork:
    def __init__(self,
        vertiports,
        flight_time,
        energy_consumption,
        od_matrix,
        num_time_steps=288,
        soc_transition_time=np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                0.0153,0.0158,0.0166,0.0172,0.018,
                                0.0188,0.0197,0.0207,0.0219,0.0231,
                                0.0245,0.026,0.0278,0.03,0.0323,
                                0.0351,0.0384,0.0423,0.0472,0.0536,
                                0.0617,0.0726,0.0887,0.1136,0.1582,
                                0.2622,0.9278,])*60
        ):
        self.vertiports = vertiports
        self.od_matrix = od_matrix
        self.flight_time = flight_time
        self.energy_consumption = energy_consumption
        self.num_time_steps = num_time_steps + np.max(self.flight_time) + 1
        self.soc_transition_time = soc_transition_time

        self.num_vertiports = len(vertiports)
        self.num_soc_levels = len(soc_transition_time) # This number does not include 0% SOC


    def populate_network(self):
        nodes, supply = self._create_nodes()
        edge, cost, c = self._create_edges()

        return nodes, supply, edge, cost, c


    def _create_nodes(self):
        base_nodes = ['Source', 'Sink']
        vertiport_nodes = [(v, t, k) for v in range(self.num_vertiports) for t in range(1, self.num_time_steps) for k in range(self.num_soc_levels+1)]
        nodes = base_nodes + vertiport_nodes
        supply = {'Source': -200, 'Sink': 200}

        return nodes, supply

    
    def _create_edges(self):
        edge, cost = self._create_basic_edges()
        edge_idling, cost_idling = self._create_idling_edges()
        edge_charging, cost_charging = self._create_charging_edges()
        edge_flight, cost_flight = self._create_flight_edges_network()

        edge = edge + edge_idling + edge_charging + edge_flight
        cost = cost + cost_idling + cost_charging + cost_flight

        return edge, cost, dict(zip(edge, cost))


    def _create_basic_edges(self):
        augmented_path = [('Source', 'Sink')]
        cost = [0]
        source_to_vertiport = [('Source', (v, 1, k)) for v in range(self.num_vertiports) for k in range(self.num_soc_levels+1)]
        cost = cost + [1 for _ in range(len(source_to_vertiport))]
        vertiport_to_sink = [((v, self.num_time_steps-1, k), 'Sink') for v in range(self.num_vertiports) for k in range(self.num_soc_levels+1)]
        cost = cost + [0 for _ in range(len(vertiport_to_sink))]
        basic_edges = augmented_path + source_to_vertiport + vertiport_to_sink

        return basic_edges, cost

    def _create_idling_edges(self):
        idling_edges = [((v,t,k), (v,t+1,k)) for v in range(self.num_vertiports) for t in range(1, self.num_time_steps-1) for k in range(self.num_soc_levels+1)]
        cost = [0 for _ in range(len(idling_edges))]

        return idling_edges, cost
    
    def _create_charging_edges(self):
        charging_edges = []
        cost = []
        for v in range(self.num_vertiports):
            for t in range(1, self.num_time_steps):
                for initial_k in range(self.num_soc_levels):
                    for final_k in range(initial_k+1, self.num_soc_levels+1):
                        charging_time = np.ceil(self.soc_transition_time[initial_k:final_k].sum()/5)
                        if t + charging_time < self.num_time_steps:
                            charging_edges.append(((v,t,initial_k), (v,int(t+charging_time),final_k)))
                            cost.append(0)

        return charging_edges, cost

    def _create_flight_edges_network(self):
        flight_edges = []
        cost = []
        for v1 in range(self.num_vertiports):
            for v2 in range(self.num_vertiports):
                if self.od_matrix[v1,v2] == 1:
                    flight_edges_v1_v2, cost_v1_v2 = self._create_flight_edges_between_two_vertiport(v1, v2)
                    flight_edges = flight_edges + flight_edges_v1_v2
                    cost = cost + cost_v1_v2

        return flight_edges, cost

    def _create_flight_edges_between_two_vertiport(self, v1, v2):
        flight_time = self.flight_time[v1, v2]
        energy_consumption = self.energy_consumption[v1, v2]

        flight_edges = []
        cost = []
        for t in range(1, self.num_time_steps-flight_time):
            for k in range(energy_consumption, self.num_soc_levels+1):
                flight_edges.append(((v1,t,k), (v2,t+flight_time,k-energy_consumption)))
                cost.append(0)
        
        return flight_edges, cost


class FlightTask:
    def __init__(self, name, start_time, flight_time, origin, destination, charging_time=None):
        self.name = name
        self.start_time = start_time
        self.flight_time = flight_time
        self.land_time = self.start_time + self.flight_time[origin, destination]
        self.origin = origin
        self.destination = destination
        
    def next_task(self, next_task):
        next_task_start_time = next_task.start_time
        ready_time = self.land_time + self.flight_time[self.destination, next_task.origin]
        if ready_time <= next_task_start_time:
            return True
        else:
            return False

class AssignmentNetwork:
    def __init__(self,
        list_of_tasks,
        soc_transition_time=np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                0.0153,0.0158,0.0166,0.0172,0.018,
                                0.0188,0.0197,0.0207,0.0219,0.0231,
                                0.0245,0.026,0.0278,0.03,0.0323,
                                0.0351,0.0384,0.0423,0.0472,0.0536,
                                0.0617,0.0726,0.0887,0.1136,0.1582,
                                0.2622,0.9278])*60
        ):
        self.list_of_tasks = list_of_tasks

    def populate_network(self):
        nodes, supply = self._create_nodes()
        edge, cost, c = self._create_edges()

        return nodes, supply, edge, cost, c


    def _create_nodes(self):
        base_nodes = ['Source', 'Sink']
        assignment_nodes = [(task.name, status) for task in self.list_of_tasks for status in ['start', 'finish']]
        nodes = base_nodes + assignment_nodes
        supply = {'Source': -200, 'Sink': 200}

        return nodes, supply

    
    def _create_edges(self):
        edge, cost = self._create_basic_edges()
        edge_reassignment, cost_reassignment = self._create_reassignment_edges()

        edge = edge + edge_reassignment
        cost = cost + cost_reassignment

        return edge, cost, dict(zip(edge, cost))


    def _create_basic_edges(self):
        augmented_path = [('Source', 'Sink')]
        cost = [0]
        source_to_task = [('Source', (task.name, 'start')) for task in self.list_of_tasks]
        cost = cost + [1 for _ in range(len(source_to_task))]
        task_to_sink = [((task.name, 'finish'), 'Sink') for task in self.list_of_tasks]
        cost = cost + [0 for _ in range(len(task_to_sink))]
        task_to_task = [((task.name, 'start'), (task.name, 'finish')) for task in self.list_of_tasks]
        cost = cost + [0 for _ in range(len(task_to_task))]
        basic_edges = augmented_path + source_to_task + task_to_sink + task_to_task

        return basic_edges, cost

    def _create_reassignment_edges(self):
        reassignment_edges = []
        number_of_tasks = len(self.list_of_tasks)
        for i in range(number_of_tasks):
            for j in range(i+1, number_of_tasks):
                if self.list_of_tasks[i].next_task(self.list_of_tasks[j]):
                    reassignment_edges.append(((self.list_of_tasks[i].name, 'finish'), (self.list_of_tasks[j].name, 'start')))
        
        cost = [0 for _ in range(len(reassignment_edges))]

        return reassignment_edges, cost
    




