# UAM System Optimization

## Problem Formulation

- $n_{i}^{k}(t)$ = \# of idle aircrafts at vertiport i of SOC k at time t
- $u_{ij}^{k}(t)$ = \# of flights departing for vertiport j from vertiport i at time t
- $C_{i}^{xy}(t)$ = \# of aircrafts at vertiport i that begin to charge at t with an initial SOC of x and a target SOC of y
- $\gamma_{k}$ is the charging time needed to transition from $SOC_{k-1}$ to $SOC_{k}$
- $\Gamma$ is the \# of SOCs. $\Gamma = 32$
- $\tau_{ij}$ is the flight time from vertiport i to j
- $\kappa_{ij}$ is the \# of SOCs dropped in flight from vertiport i to j

The dynamic equation is then:

```math
n_{i}^{k}(t) = n_{i}^{k}(t-1) + \sum _{j \in V-\{i\}} u_{j,i}^{k+\kappa_{ji}}(t-\tau_{ji}) - \sum_{j \in V-\{i\}} u_{ij}^{k} (t) + \sum_{x=0}^{k-1} C_{i}^{x,k} (t-\sum_{i=x+1}^{k} \gamma_{i}) - \sum_{y=k+1}^{\Gamma} C_{i}^{k,y}(t)
```

Therefore, an aircraft can be in one of the three states at any given point in time: (1) idle (2) charging (3) flight

$\sum_{k \in\{1, \cdots, K\}} u_{12}^k(t) \geq f_{12}(t), \sum_{k \in\{1, \cdots, K\}} u_{21}^k(t) \geq f_{21}(t)$ The flight schedule must be satisfied

$u_{12}^0(t) = u_{21}^0(t) = 0$ Cannot fly when SOC = 0. 0 is the reserved SOC

## Stationarity Condition
T = number of timesteps + 1 + max flight time

$n_{i}^{k}(0)=n_{i}^{k}(T)$, $u_{ij}^{k}(0)=u_{ij}^{k}(T)$, $C_{i}^{xy}(0)=C_{i}^{xy}(T)$

## Objective Function

$min \sum_{i}\sum_{k} n_{i}^{k}(t=0) + \sum_{i} \sum_{x} \sum_{y} C_i^{xy}(t = 0) + 0.00001 \cdot  \sum_{t} \sum_{i} \sum_{j} \sum_{k} u_{i,j}^{k}(t)$
