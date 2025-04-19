```markdown
# Parallel Algorithm for Constructing Independent Spanning Trees in Bubble-Sort Networks

## Overview

This project implements and extends the parallel algorithm for constructing multiple Independent Spanning Trees (ISTs) in bubble-sort networks, as described in the paper *"A Parallel Algorithm for Constructing Multiple Independent Spanning Trees in Bubble-Sort Networks"* by Shih-Shun Kao et al., published in the *Journal of Parallel and Distributed Computing* (2023). The algorithm constructs \( n-1 \) ISTs in a bubble-sort network \( B_n \), achieving asymptotically optimal time complexity of \( \mathcal{O}(n \cdot n!) \) and enabling full parallelization.

The project includes:
- A parallel implementation of the algorithm using MPI (inter-node communication), OpenMP (intra-node parallelism), and METIS (graph partitioning).
- Documentation, including a presentation summarizing the paper's contributions.
- Example scripts and configurations for running the algorithm on sample bubble-sort networks.

## Project Structure

```
parallel-ist-bubble-sort/
├── src/
│   ├── main.c               # Main program implementing Algorithm 1
│   ├── parent1.c            # Core parent computation logic
│   ├── graph_utils.c        # Utilities for bubble-sort network generation
│   ├── metis_interface.c    # METIS partitioning functions
│   └── Makefile             # Build script
├── docs/
│   ├── presentation.md      # Presentation summarizing the paper
│   └── paper.pdf            # Original research paper
├── examples/
│   ├── b4_config.txt        # Configuration for B_4 (n=4)
│   └── b5_config.txt        # Configuration for B_5 (n=5)
├── scripts/
│   ├── run_mpi.sh           # Script to run on MPI cluster
│   └── plot_results.py      # Script to visualize ISTs
├── README.md                # This file
└── LICENSE                  # License file
```

## Prerequisites

To build and run the project, you need:

- **Compiler**: GCC or compatible (supporting C11)
- **MPI**: OpenMPI or MPICH (for inter-node parallelism)
- **OpenMP**: Included with GCC (for intra-node parallelism)
- **METIS**: Version 5.1.0 or later (for graph partitioning)
- **Python**: (Optional) For visualization scripts (requires `matplotlib`, `networkx`)
- **Operating System**: Linux/Unix (recommended) or compatible

### Installation

1. **Install Dependencies**:
   ```bash
   sudo apt-get install build-essential openmpi-bin openmpi-dev libmetis-dev
   pip install matplotlib networkx
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/parallel-ist-bubble-sort.git
   cd parallel-ist-bubble-sort
   ```

3. **Build the Project**:
   ```bash
   cd src
   make
   ```

## Usage

### Running the Algorithm

1. **Prepare Configuration**:
   - Edit `examples/b4_config.txt` or `examples/b5_config.txt` to specify the dimension \( n \) and other parameters (e.g., number of MPI nodes).

2. **Run on a Cluster**:
   ```bash
   cd scripts
   ./run_mpi.sh 4 examples/b4_config.txt
   ```
   - Replace `4` with the number of MPI processes.
   - Output: Parent assignments for each vertex in \( n-1 \) ISTs, saved to `output/`.

3. **Visualize Results** (Optional):
   ```bash
   python plot_results.py output/b4_results.txt
   ```
   - Generates visualizations of the ISTs (requires `matplotlib` and `networkx`).

### Example

For \( B_4 \) (24 vertices, 3 ISTs):
```bash
mpirun -np 4 ./src/main examples/b4_config.txt
```
- Partitions the graph using METIS.
- Computes parents for each vertex in parallel using MPI and OpenMP.
- Outputs paths from vertex `4231` to root `1234` in each IST.

## Features

- **Parallel Implementation**:
  - **MPI**: Distributes vertices across nodes, minimizing communication via METIS partitioning.
  - **OpenMP**: Parallelizes parent computations within nodes.
  - **METIS**: Balances load and reduces edge cuts for efficient partitioning.
- **Scalability**: Tested for \( n \leq 5 \); extensible to larger \( n \) with sufficient resources.
- **Correctness**: Verified against the paper's case analysis (Cases A, B, C).
- **Visualization**: Scripts to plot ISTs for small networks.

## Documentation

- **Paper**: See `docs/paper.pdf` for the original research.
- **Presentation**: `docs/presentation.md` summarizes the paper's contributions and parallelization strategy.
- **Code Comments**: Extensive comments in `src/` explain the implementation.

## Future Work

- Extend the algorithm to \( (n, k) \)-bubble-sort graphs or butterfly graphs.
- Optimize IST height for faster routing.
- Implement GPU-based parallelism (e.g., CUDA) for permutation computations.
- Benchmark on large-scale HPC clusters for \( n \geq 6 \).

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact

For questions or contributions, please contact:
- **Your Name**: i221131@nu.edu.pk
- **GitHub Issues**: Open an issue at `https://github.com/yourusername/parallel-ist-bubble-sort`
```