from gymnasium import Env
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
import pandas as pd
import os, random

class aircraft:
    """
    Defines a simple aircraft class that can keep track of its soc, current vertiport, and delay timestep
    """
    def __init__(self, currentVertiport, aircraft_id, intial_soc=80, flight_time=2, charge_time=1, soc_change=10, 
                soc_transition_time=np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                0.0153,0.0158,0.0166,0.0172,0.018,
                0.0188,0.0197,0.0207,0.0219,0.0231,
                0.0245,0.026,0.0278,0.03,0.0323,
                0.0351,0.0384,0.0423,0.0472,0.0536,
                0.0617,0.0726,0.0887,0.1136,0.1582,
                0.2622,0.9278,])*60):
        self.aircraft_id = aircraft_id
        self.soc = intial_soc
        self.delay_timestep = 0
        self.flight_time = flight_time
        self.charge_time = charge_time
        self.soc_change = soc_change
        self.currentVertiport = currentVertiport
        self.soc_transition_allowed = []
        for i in range(len(soc_transition_time)):
            idx = i
            cnt = 0
            while soc_transition_time[i:idx+1].sum() <= 5:
                cnt += 1
                idx += 1
            self.soc_transition_allowed.append(cnt)
        self.soc_transition_allowed = np.array(self.soc_transition_allowed + [0,0,0,0,0,0,0])

    def fly(self):
        """
        Change vertiports if possible
        """
        if self.delay_timestep == 0 and self.soc > self.soc_change:
            self.delay_timestep += self.flight_time
            self.soc -= self.soc_change
            self.currentVertiport = (self.currentVertiport+1)%2
            return True
        return False

    def charge(self):
        """
        Set to charge
        """
        if self.delay_timestep == 0:
            self.delay_timestep += self.charge_time
            self.soc += self.soc_transition_allowed[int(self.soc / 2.5)] * 2.5

    def update(self):
        """
        Decrease the delay timestep
        """
        if self.delay_timestep > 0:
            self.delay_timestep -= 1


class vertiport():
    """
    Keeps track of the number of passengers at a given vertiport
    """
    def __init__(self, name, vehicleCap = 4) -> None:
        self.passengers = 0
        self.name = name
        self.vehicleCapacity = vehicleCap

    def departVehicles(self, vehiclesNumbers):
        self.passengers = max(0, self.passengers - vehiclesNumbers * self.vehicleCapacity)

    def addPassengers(self, number):
        self.passengers += number

base = "C:/AM/Projects/AI/RL/UAM_system_optimization/input/demand_variation/passenger_arrival/"
files = ["alpha_7_demand_1500/" + i for i in os.listdir(base + "alpha_7_demand_1500")]
def reset():
    global files
    files = ["alpha_7_demand_1500/" + i for i in os.listdir(base + "alpha_7_demand_1500")]

def getRandomFile():
    """
    Gets a random file to generate the demand
    """
    if len(files) == 0:
        reset()
    f = random.choice(files)
    files.remove(f)
    return base + f

def getPassengerArrival(f):
    """
    Parses the data from the passenger demand csv file
    """
    data = pd.read_csv(f)
    data = data.dropna()
    data['passenger_arrival_time'] = np.ceil(data['passenger_arrival_time'] / (60*5))
    data = data.sort_values(by="passenger_arrival_time", ascending=True).reset_index()
    data = data.drop(["passenger_id", "index"], axis = 1)
    data["origin_vertiport_id"] = data["origin_vertiport_id"].map({"LAX":0, "DTLA":1})
    data["destination_vertiport_id"] = data["destination_vertiport_id"].map({"LAX":0, "DTLA":1})
    return data

class system(Env):
    def __init__(self, fleet_size):
        self.action_space = MultiDiscrete([3] * fleet_size)
        self.fleet_size = fleet_size

        a = {"delay_timestep" : MultiDiscrete([5] * fleet_size), "soc": Box(1, 100, (fleet_size,)), "vertiport" : MultiDiscrete([2] * fleet_size), "vertiportPassengers" : Box(0, np.inf, (2,), dtype=np.int32)}
        self.observation_space = Dict(a)        
        
        self.fleet = [aircraft(i, 0) for i in range(fleet_size//2)] + [aircraft(i, 1) for i in range(fleet_size - fleet_size//2)]
        self.vertiports = [vertiport("LAX"), vertiport("DTLA")]
        self.time = 0
        self.passengerArrival = getPassengerArrival(getRandomFile())
        self.stopTime = int(self.passengerArrival["passenger_arrival_time"].iloc[-1]) + 1
        self.spillValue = 0

    def step(self, action):
        flights = [0, 0]
        passengers = [0, 0]

        for i in range(len(action)): # 0: fly, 1: charge, 2: nothing
            self.fleet[i].update()
            if action[i] == 0:
                if self.fleet[i].fly():
                    flights[(self.fleet[i].currentVertiport+1)%2] += 1
            elif action[i] == 1:
                self.fleet[i].charge()
        self.vertiports[0].departVehicles(flights[0])
        self.vertiports[1].departVehicles(flights[1])
        self.spillValue = (self.vertiports[0].passengers + self.vertiports[1].passengers)

        # Add the passengers for the current timestep
        for i, row in self.passengerArrival[self.passengerArrival["passenger_arrival_time"] == self.time].iterrows():
            passengers[int(row["origin_vertiport_id"])] += 1
        self.vertiports[0].addPassengers(passengers[0])
        self.vertiports[1].addPassengers(passengers[1])

        state = self.getState()
        self.time += 1
        return state, -self.spillValue, self.time == self.stopTime, False, {}

    def render(self):
        pass

    def reset(self, seed = None, options=None):
        self.fleet = [aircraft(0, i) for i in range(self.fleet_size//2)] + [aircraft(1, i + self.fleet_size//2) for i in range(self.fleet_size - self.fleet_size//2)]
        self.vertiports = [vertiport("LAX"), vertiport("DTLA")]
        self.time = 0
        self.passengerArrival = getPassengerArrival(getRandomFile())
        self.stopTime = int(self.passengerArrival["passenger_arrival_time"].iloc[-1]) + 1

        return (self.getState(), {})

    def getState(self):
        a = {"delay_timestep" : np.array([self.fleet[i].delay_timestep for i in range(self.fleet_size)], dtype=np.int64),
             "vertiport" : np.array([self.fleet[i].currentVertiport for i in range(self.fleet_size)], dtype=np.int64),
             "soc" : np.array([self.fleet[i].soc for i in range(self.fleet_size)], dtype=np.float32),
             "vertiportPassengers" : np.array([self.vertiports[0].passengers, self.vertiports[1].passengers], dtype=np.int32)}
        return a
