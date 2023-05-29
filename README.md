# UAM_system_optimization

For solving the optimization problem as follows:

- $n_{i}^{k}(t)$ = \# of idle aircraft at vertiport i of SOC k at time t
- $u_{i,j}^{k}(t)$ = \# of flights departing for vertiport j from vertiport i at time t
- $C_{i}^{xy}(t)$ = \# of aircraft at vertiport i that begin to charge at t with an initial SOC of x and a target SOC of y
- $\gamma_{i}$ is the charging time needed to transition from $SOC_{k-1}$ to $SOC_{k}$
- $|\gamma|$ is the \# of SOCs 
- $\tau_{ij}$ is the flight time from vertiport i to j
- $\kappa_{ij}$ is the \# of SOCs dropped in flight from vertiport i to j

The dynamic equation is then:

```math
n_{i}^{k}(t) = n_{i}^{k}(t-1) + \sum _{j \in V-\{i\}} u_{j,i}^{k+\kappa_{j,i}}(t-\tau_{j,i}) - \sum_{j \in V-\{i\}} u_{i,j}^{k} (t) + \sum_{x=0}^{k-1} C_{i}^{x,k} (t-\sum_{i=x+1}^{k} \gamma_{i}) - \sum_{y=k+1}^{|\gamma|} C_{i}^{k,y}(t)

Therefore, an aircraft can be in one of the three states at any given point in time: (1) idle (2) charging (3) flight

\sum_{k \in\{1, \cdots, K\}} u_{12}^k(t) \geq f_{12}(t), \sum_{k \in\{1, \cdots, K\}} u_{21}^k(t) \geq f_{21}(t),: \textit{ The flight schedule must be satisfied. }

u_{12}^0(t) = u_{21}^0(t) = 0 \textit{  Can't fly when SOC = 0}

We keep all the constraints we discussed previously and the optimization problem becomes:

min \sum_{i}\sum_{k} n_{i}^{k}(t=0) + \sum_{i} \sum_{j} \sum_{k} u_{i,j}^k(t = 0) + \sum_i \sum_{x} \sum_{y} C_i^{xy}(t = 0)


Please note that Markdown doesn't natively support LaTeX, but some Markdown processors, like the one used on GitHub and Jupyter Notebook, support MathJax which can render LaTeX-style equations. This varies by platform, so please ensure that your Markdown processor supports LaTeX-style equations before using them.
