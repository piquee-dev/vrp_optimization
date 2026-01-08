# Vehicle Routing Problem Optimization

A comprehensive implementation and analysis of the Vehicle Routing Problem (VRP) using multiple optimization algorithms, clustering techniques, and parallel processing.

---

## Project Overview

This project addresses the Vehicle Routing Problem, an extension of the classical Traveling Salesman Problem (TSP), focusing on optimizing delivery routes for multiple vehicles. The implementation demonstrates theoretical complexity analysis, practical algorithm development, and performance evaluation through statistical methods.

> **Academic Context**: Advanced Algorithms Course  
> **Focus Areas**: Combinatorial Optimization, Graph Theory, Computational Complexity

---

## Team Members

| Name | Contribution |
|------|--------------|
| Pranav Kumar | Algorithm Development & Implementation |
| Aryan Mihir Saraf | Complexity Analysis & Testing |

---

## Problem Statement

The project models a delivery optimization scenario where a company must efficiently route vehicles to visit multiple cities. Each vehicle must:

- Visit each assigned city exactly once
- Return to the starting depot
- Minimize total travel time/distance
- Respect vehicle capacity and time constraints

### Key Constraints

**Basic Requirements:**
- Calculate optimal routes on a road network connecting multiple cities
- Minimize total route duration with depot return
- Provide statistical analysis of algorithm performance

**Advanced Features:**
- Graph partitioning into clusters representing multiple vehicles
- Multiprocessing implementation for concurrent route optimization
- Time constraint simulation for trip duration assessment

---

## Theoretical Foundation

### Complexity Classification

> **Problem Class**: NP-Hard  
> **Verification Class**: NP-Complete  
> **Computational Complexity**: O(n!) for brute force, improved with heuristics

### Formal Proof of NP-Hardness

The project includes a rigorous reduction proof demonstrating that TSP (and by extension, VRP) is NP-Hard through reduction from the Hamiltonian Cycle problem:

1. Construct complete graph G' from Hamiltonian Cycle graph G
2. Assign edge costs: existing edges = 1, non-existing edges = 2
3. Set total cost threshold K = N (number of vertices)
4. Prove equivalence between Hamiltonian Cycle existence and TSP solution

### Graph Theory Concepts

**Hamiltonian Path and Cycle:**
- Path visiting each vertex exactly once
- Cycle forming closed loop through all vertices
- NP-Complete problem with no known polynomial solution

**Eulerian Path and Cycle:**
- Path visiting each edge exactly once
- Existence determined by vertex degree properties

---

## Algorithm Implementations

### Genetic Algorithm

Primary optimization approach using evolutionary computation principles:

**Core Components:**
- **Initialization**: Random population generation with chromosome encoding
- **Fitness Evaluation**: Distance/time-based fitness function
- **Selection**: Fitness-proportionate parent selection
- **Crossover**: Multi-point recombination operators
- **Mutation**: Random gene swapping with controlled rate
- **Elitism**: Preservation of best solutions across generations

**Parameters:**
```
Population Size: 30
Generations: 50
Mutation Rate: 0.05
Crossover Rate: 0.80
Selection Method: Fitness-proportionate
```

### Clustering Algorithm

**K-Means Clustering:**
- Optimal cluster determination using Silhouette Score analysis
- Standardized coordinate scaling for improved convergence
- Cluster range evaluation: 2-30 clusters
- Each cluster represents one vehicle's service area

### Optimization Techniques

**Metaheuristic Approaches:**
- Simulated Annealing for local optima escape
- Tabu Search with memory structures
- Variable Neighborhood Search for solution space exploration
- Iterated Local Search for solution perturbation

---

## Technical Implementation

### Technology Stack

**Programming Language:** Python 3.x

**Core Libraries:**
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization and convergence plotting
- **scikit-learn**: K-Means clustering and Silhouette analysis
- **Multiprocessing**: Parallel execution for cluster optimization

### Key Features

**Parallel Processing:**
- Concurrent route optimization across multiple clusters
- Multiprocessing pool for CPU utilization
- Independent genetic algorithm execution per cluster

**Performance Monitoring:**
- Generation-wise convergence tracking
- Best solution evolution visualization
- Statistical performance analysis

**Scalability:**
- Tested with 500 nodes
- Configurable cluster sizes
- Dynamic parameter adjustment

---

## Data Structures

### Graph Representation

**Adjacency Matrix:**
- Complete graph representation
- Distance/time between all node pairs
- Symmetric for undirected graphs

**Coordinate System:**
- 2D Euclidean space for node positions
- Distance calculation using Euclidean metric
- Depot positioned at origin (0, 0)

### Route Encoding

**Chromosome Structure:**
- Permutation encoding for city visit order
- Fixed start/end depot points
- Variable-length sequences for different cluster sizes

---

## Results and Analysis

### Performance Metrics

**Convergence Analysis:**
- Multi-cluster convergence visualization
- Generation-wise improvement tracking
- Comparative performance across cluster sizes

**Optimization Quality:**
- Route time minimization achieved
- Solution diversity maintenance
- Premature convergence avoidance

### Statistical Validation

- Silhouette Score optimization for clustering
- Fitness function evolution analysis
- Parameter sensitivity studies

---

## Installation and Usage

### Prerequisites

```bash
Python 3.7 or higher
pip package manager
```

### Required Dependencies

```bash
pip install numpy matplotlib scikit-learn jupyter
```

### Running the Notebook

```bash
jupyter notebook VRP.ipynb
```

### Execution Steps

1. Open the Jupyter notebook
2. Execute cells sequentially for theory and implementation
3. Modify parameters in configuration cells as needed
4. Run genetic algorithm with specified parameters
5. Analyze convergence plots and optimal routes
6. Experiment with different cluster configurations

---

## Project Structure

```
VRP.ipynb
├── Introduction and Problem Definition
├── Theoretical Requirements
│   ├── Graph Theory Fundamentals
│   ├── Complexity Analysis
│   ├── Hamiltonian and Eulerian Concepts
│   └── TSP and VRP Problem Definitions
├── Complexity Proofs
│   ├── TSP NP-Hardness Proof
│   └── VRP NP-Hardness Reduction
├── Algorithm Design
│   ├── Genetic Algorithm Theory
│   ├── Clustering Methods
│   └── Optimization Strategies
├── Technical Implementation
│   ├── Data Structure Definitions
│   ├── Fitness Functions
│   ├── Genetic Operators
│   ├── Clustering Implementation
│   └── Multiprocessing Framework
├── Experimental Results
│   ├── Convergence Analysis
│   ├── Performance Visualization
│   └── Statistical Evaluation
└── References and Bibliography
```

---

## Configuration Parameters

### Genetic Algorithm Tuning

```python
SPEED = 72                    # Vehicle speed (km/h)
TIME_SCALE_FACTOR = 0.05      # Time scaling factor
POPULATION_SIZE = 30          # Number of solutions per generation
NUM_GENERATIONS = 50          # Iteration count
MUTATION_RATE = 0.05          # Probability of mutation (5%)
```

### Clustering Configuration

```python
n_nodes = 500                 # Total number of delivery points
cluster_range = range(2, 31)  # Cluster evaluation range
random_state = 42             # Reproducibility seed
n_init = 10                   # K-Means initialization attempts
```

---

## Visualization Outputs

### Generated Plots

**Silhouette Score Analysis:**
- Optimal cluster number identification
- Score comparison across cluster ranges

**Convergence Graphs:**
- Best fitness evolution per generation
- Multi-cluster performance comparison
- Time improvement trajectories

**Route Visualization:**
- Geographic node distribution
- Optimal path representation per cluster
- Depot connectivity illustration

---

## Future Enhancements

**Advanced Constraints:**
- Vehicle capacity limitations
- Time window restrictions for deliveries
- Multiple depot scenarios
- Dynamic traffic considerations

**Algorithm Improvements:**
- Hybrid genetic-local search approaches
- Adaptive parameter tuning
- Population diversity enhancement
- Multi-objective optimization

**Scalability Enhancements:**
- Distributed computing implementation
- GPU acceleration for fitness evaluation
- Real-time route adjustment capabilities
- Integration with mapping APIs

---

## Mathematical Formulation

### Objective Function

```
Minimize: Σ(i,j)∈E c(i,j) * x(i,j)

Subject to:
- Σj x(i,j) = 1  ∀i ∈ V (each city visited once)
- Σi x(i,j) = 1  ∀j ∈ V (each city departed once)
- Capacity constraints
- Time window constraints
```

Where:
- x(i,j) = 1 if edge (i,j) is in the route, 0 otherwise
- c(i,j) = cost/time to travel from i to j
- V = set of all vertices (cities)
- E = set of all edges

---

## References

### Academic Sources

- Stanford University - Computational Complexity Theory
- Graph Theory and Combinatorial Optimization Literature
- Genetic Algorithms in Search, Optimization, and Machine Learning
- Vehicle Routing Problem: Latest Advances and New Challenges

### Implementation Resources

- scikit-learn Documentation: Clustering Algorithms
- NumPy Documentation: Array Operations and Linear Algebra
- Matplotlib Documentation: Scientific Visualization

---

## License

This project is submitted as part of academic coursework. All rights reserved by the project team members.

---

## Acknowledgments

Special thanks to the course instructor and teaching assistants for guidance on algorithm design and complexity analysis. The theoretical foundations draw from established research in combinatorial optimization and evolutionary computation.

---

## Contact

For questions regarding implementation details or theoretical aspects of the project, please refer to the detailed explanations within the Jupyter notebook or consult the referenced academic literature.

---

## Disclaimer

> **Important**: This is an academic project designed for educational purposes. The algorithms and implementations are optimized for learning and demonstration rather than production deployment. Real-world applications may require additional considerations for scalability, robustness, and regulatory compliance.
