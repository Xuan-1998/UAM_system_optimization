import pandas as pd
import numpy as np
from gurobipy import Model, GRB
from gurobipy import quicksum
import gurobipy as gp
import pickle

class LagrangianRelaxedFleetSize:
    def __init__(self, model_path, edges, cost, flight_time, energy_consumption, flight_demand):
        self.model_path = model_path
        self.edges = edges
        self.c = cost
        self.flight_time = flight_time
        self.energy_consumption = energy_consumption
        self.flight_demand = flight_demand
        self.list_of_cuts = []
        self.list_of_b = []
        self.list_of_dual_ofv = []
        self.lambda_s = {i:2 for i in self.flight_demand.keys()}
        self.primal_ofv, self.grad = self._solve_for_x(self.lambda_s)
        self.num_init = 1

    def solve(self, path, load_init=False, max_iter=2000):
        if load_init:
            print('Loading values from the previous run')
            self._load_init(path)
            dual_obj, self.lambda_s = self._solve_for_u(self.list_of_cuts, self.list_of_b)
            self.primal_ofv, self.grad = self._solve_for_x(self.lambda_s)
            self.num_init = len(self.list_of_dual_ofv)+1
            print('Number of existing iterations: ', self.num_init-1)

        for _ in range(max_iter):
            cut, b = self._get_cut(self.primal_ofv, self.lambda_s, self.grad)
            self.list_of_cuts.append(cut)
            self.list_of_b.append(b)

            dual_obj, self.lambda_s = self._solve_for_u(self.list_of_cuts, self.list_of_b)
            if dual_obj <= self.primal_ofv:
                break
            else:
                self.primal_ofv, self.grad = self._solve_for_x(self.lambda_s)
                self.list_of_dual_ofv.append(dual_obj)
            print('Iteration:', self.num_init, 'Dual OFV: ', dual_obj)
            self.num_init += 1
            self._log_results(path)
        
    def _log_results(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.list_of_cuts, self.list_of_b, self.list_of_dual_ofv), f)

    def _load_init(self, path):
        with open(path, 'rb') as f:
            self.list_of_cuts, self.list_of_b, self.list_of_dual_ofv = pickle.load(f)
        

    def _get_flight_edges(self, i, model_param):
        K = 32
        v1, v2, t = i
        flight_time_ij = self.flight_time[v1, v2]
        soc_level_drop = int(self.energy_consumption[v1, v2])
        volume = quicksum(model_param[(v1, t, k), (v2, t+flight_time_ij, k-soc_level_drop)] for k in range(soc_level_drop, K+1))
        return volume
    

    def _solve_for_x(self, u):
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        cp = gp.read(self.model_path, env=env)
        flow_vars = cp.getVars()
        flow = {self.edges[idx]: flow_vars[idx] for idx in range(len(flow_vars))}

        primal_objective = quicksum(self.c[i,j] * flow[i,j] for i,j in self.edges) + quicksum(u[i] * (self.flight_demand[i] - self._get_flight_edges(i, flow)) for i in self.flight_demand.keys())

        cp.setObjective(primal_objective, GRB.MINIMIZE)
        cp.optimize()

        x0 = {self.edges[idx]: flow_vars[idx].X for idx in range(len(flow_vars))}
        grad = np.array([self.flight_demand[i] - self._get_flight_edges(i, x0) for i in self.flight_demand.keys()])
        primal_ofv = cp.objVal

        return primal_ofv, grad

    def _solve_for_u(self, list_of_cuts, list_of_b):
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        cp = gp.read(self.model_path, env=env)
        z = cp.addVar(name='z')
        lambda_s = cp.addVars(self.flight_demand.keys(), name='lambda', lb=0, ub=4)
        cp.update()

        for i in range(len(list_of_cuts)):
            cut = list_of_cuts[i]
            b = list_of_b[i]
            cut = {list(self.flight_demand.keys())[idx]: cut[idx] for idx in range(len(cut))}

            cp.addConstr(z <= quicksum(lambda_s[j] * cut[j] for j in self.flight_demand.keys()) + b)

        cp.setObjective(z, GRB.MAXIMIZE)
        cp.optimize()

        dual_ofv = cp.objVal
        lamb = {i: lambda_s[i].X for i in self.flight_demand.keys()}

        return dual_ofv, lamb

    
    def _get_cut(self, primal_ofv, lamb, grad):
        lamb = np.array([lamb[i] for i in lamb.keys()])
        grad = np.array([grad[i].getConstant() for i in range(len(grad))])
        b = primal_ofv - np.dot(lamb, grad)
        return grad, b