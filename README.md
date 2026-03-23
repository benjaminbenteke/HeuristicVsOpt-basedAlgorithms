# Heuristics vs. Optimization-Based Algorithms for GNEPs

This project investigates and compares two classes of algorithms—heuristic methods and optimization-based methods—in their ability to characterize the solution set of a Generalized Nash Equilibrium Problem (GNEP).

A GNEP extends the classical Nash equilibrium by allowing each player's feasible strategy set to depend on the strategies chosen by other players. This interdependence significantly increases the complexity of finding equilibria and makes it a rich area for algorithmic exploration.

* Heuristic algorithms: 
* Optimization-based algorithms:

The project compares these two approaches on how well they describe the entire set of equilibria rather than a single solution. We achieve this by measuring the:

* Accuracy – How closely the algorithm identifies the true equilibrium set.
* Diversity – 
* Coverage – 
Characterization of solution set – 
## Run EIA
### Run all exercises
```
python main.py --ex all
```
### Run a single exercise (e.g., Exercise 1)
```
python main.py --ex 1
```
### Run multiple exercises (e.g., Exercise 1 and 3)
```
python main.py --ex 1 3
```

