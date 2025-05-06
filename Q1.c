#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <metis.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>

// Calculate factorial at compile time to avoid runtime computation
#define FACTORIAL_2 2LL
#define FACTORIAL_3 6LL
#define FACTORIAL_4 24LL
#define FACTORIAL_5 120LL
#define FACTORIAL_6 720LL
#define FACTORIAL_7 5040LL
#define FACTORIAL_8 40320LL
#define FACTORIAL_9 362880LL
#define FACTORIAL_10 3628800LL
#define FACTORIAL_11 39916800LL
#define FACTORIAL_12 479001600LL

#define N 3     // Length of permutations
#define FACTORIAL FACTORIAL_13  // Use the appropriate pre-computed factorial

// Set the number of OpenMP threads
#define NUM_OMP_THREADS 1
// Constants for workload balancing and memory management
#define CHUNK_SIZE 1000       // Number of vertices to process at once
#define BATCH_SIZE 1000000    // Number of parent relationships to collect at once
#define DYNAMIC_SCHED_CHUNK 16
#define OUTPUT_SAMPLE_COUNT 100 // Number of permutations to sample for output
// Thread balancing settings to avoid performance degradation with more threads
#define USE_THREAD_AFFINITY true
#define USE_IMPROVED_WORKLOAD_BALANCE true

// Function to initialize OpenMP environment with optimal settings
void initialize_openmp() {
    // Set number of threads based on available hardware
    int max_threads = omp_get_max_threads();
    int optimal_threads = (NUM_OMP_THREADS > max_threads) ? max_threads : NUM_OMP_THREADS;
    
    // Set thread affinity to minimize cache thrashing and NUMA effects
    if (USE_THREAD_AFFINITY) {
        setenv("OMP_PROC_BIND", "close", 1);  // Keep threads close to minimize memory access latency
        setenv("OMP_PLACES", "cores", 1);     // Bind to physical cores
    }
    
    omp_set_num_threads(optimal_threads);
    omp_set_dynamic(0);  // Disable dynamic adjustment of threads
    
    // Improve thread management
    omp_set_nested(0);  // Disable nested parallelism which can lead to oversubscription
    omp_set_max_active_levels(1); // Ensure only one level of parallelism
}

// Pre-computed factorials for better performance
long long fact_cache[21] = {0};

// Initialize factorial cache
void init_factorial_cache() {
    fact_cache[0] = 1LL;
    for (int i = 1; i <= 20; i++) {
        fact_cache[i] = fact_cache[i-1] * i;
    }
}

// Factorial function with caching for runtime calculations
long long factorial(int n) {
    if (n <= 20 && fact_cache[n] != 0) {
        return fact_cache[n];
    }
    
    long long result = 1LL;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Function to unrank a permutation (convert index to permutation)
void unrank_permutation(long long index, int n, int perm[]) {
    // Create a list of available elements
    int available[N];
    for (int i = 0; i < n; i++) {
        available[i] = i + 1;
    }
    
    // Convert index to factorial number system
    for (int i = 0; i < n; i++) {
        long long fact = factorial(n - i - 1);
        int digit = index / fact;
        index = index % fact;
        
        // Select the digit-th available element
        perm[i] = available[digit];
        
        // Remove the used element - optimized to avoid unnecessary memmove
        if (digit < n - i - 1) {
            int val_to_move = available[digit];
            for (int j = digit; j < n - i - 1; j++) {
                available[j] = available[j + 1];
            }
        }
    }
}

// Function to rank a permutation (convert permutation to index)
long long rank_permutation(int perm[], int n) {
    long long rank = 0;
    int seen[N+1] = {0};  // Track which elements we've seen so far
    
    // Pre-compute factorials for better performance
    long long facts[N];
    for (int i = 0; i < n; i++) {
        facts[i] = factorial(n - i - 1);
    }
    
    for (int i = 0; i < n; i++) {
        int smaller_count = 0;
        // Count elements smaller than perm[i] that haven't been seen yet
        for (int j = 1; j < perm[i]; j++) {
            if (!seen[j]) {
                smaller_count++;
            }
        }
        rank += smaller_count * facts[i];
        seen[perm[i]] = 1;
    }
    
    return rank;
}

// Check if two permutations differ by a single adjacent swap
bool are_adjacent(int perm1[], int perm2[]) {
    // Count positions where permutations differ
    int diff_positions[N];
    int diff_count = 0;
    
    for (int i = 0; i < N; i++) {
        if (perm1[i] != perm2[i]) {
            diff_positions[diff_count++] = i;
        }
    }
    
    // Two permutations are adjacent if they differ at exactly 2 positions
    // AND these positions are adjacent
    // AND the values are swapped between the two positions
    if (diff_count != 2) {
        return false;
    }
    
    int pos1 = diff_positions[0];
    int pos2 = diff_positions[1];
    
    return (abs(pos1 - pos2) == 1) && 
           (perm1[pos1] == perm2[pos2]) && 
           (perm1[pos2] == perm2[pos1]);
}

// Function to check if the vertex is identity permutation (1_n)
bool is_identity(int *v, int n) {
    for (int i = 0; i < n; i++) {
        if (v[i] != i + 1) {
            return false;
        }
    }
    return true;
}

// Function to copy a permutation
void copy_perm(int *src, int *dest, int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = src[i];
    }
}

// Function to find position of a value in the permutation (v^-1(x))
int find_position(int *v, int n, int x) {
    for (int i = 0; i < n; i++) {
        if (v[i] == x) {
            return i;
        }
    }
    return -1; // Not found (shouldn't happen for valid permutations)
}

// Calculate r(v) - the value that should be in the first incorrect position from the right
int calculate_r(int *v, int n) {
    for (int i = n - 1; i >= 0; i--) {
        if (v[i] != i + 1) {
            // Return the value that should be in this position
            return i + 1;
        }
    }
    return -1; // Should not reach here for non-identity permutations
}

// Swap function as defined in the algorithm
void swap(int *v, int n, int x, int *result) {
    // First copy the permutation to result
    copy_perm(v, result, n);
    
    // Find position i of x in v
    int i = find_position(v, n, x);
    
    // If i is not the last position, swap with the next element
    if (i < n - 1) {
        // Swap positions i and i+1
        int temp = result[i];
        result[i] = result[i + 1];
        result[i + 1] = temp;
    }
}

// Helper function to check if Swap(v,x) equals identity
bool swap_equals_identity(int *v, int n, int x) {
    int result[N];
    swap(v, n, x, result);
    bool is_id = is_identity(result, n);
    return is_id;
}

// FindPosition function as defined in the algorithm
void find_position_func(int *v, int n, int t, int *result) {
    // If t = 2 and Swap(v,t) = 1_n then p = Swap(v, t-1)
    if (t == 2 && swap_equals_identity(v, n, t)) {
        swap(v, n, t - 1, result);
    }
    // Else if v_{n-1} ∈ {t, n-1} then j = r(v), p = Swap(v, j)
    else if (v[n - 2] == t || v[n - 2] == n - 1) {
        int j = calculate_r(v, n);
        swap(v, n, j, result);
    }
    // Else p = Swap(v, t)
    else {
        swap(v, n, t, result);
    }
}

// Main Parent1 function
void parent1(int *v, int n, int t, int *result) {
    // If v_n = n then
    if (v[n - 1] == n) {
        // If t != n-1 then p = FindPosition(v)
        if (t != n - 1) {
            find_position_func(v, n, t, result);
        }
        // Else p = Swap(v, v_{n-1})
        else {
            swap(v, n, v[n - 2], result);
        }
    }
    // Else
    else {
        // If v_n = n-1 and v_{n-1} = n and Swap(v,n) != 1_n then
        if (v[n - 1] == n - 1 && v[n - 2] == n && !swap_equals_identity(v, n, n)) {
            // If t = 1 then p = Swap(v,n)
            if (t == 1) {
                swap(v, n, n, result);
            }
            // Else p = Swap(v, t-1)
            else {
                swap(v, n, t - 1, result);
            }
        }
        // Else
        else {
            // If v_n = t then p = Swap(v,n)
            if (v[n - 1] == t) {
                swap(v, n, n, result);
            }
            // Else p = Swap(v,t)
            else {
                swap(v, n, t, result);
            }
        }
    }
}

// Print a permutation
void print_perm(int *v, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", v[i]);
        if (i < n-1) printf(",");
    }
    printf("]");
}

// Function to find the adjacent permutations of a given permutation - optimized version
int find_adjacent_permutations(int perm[], int n, long long *adj_indices) {
    int count = 0;
    int temp_perm[N];
    
    // Avoid copying entire array repeatedly
    memcpy(temp_perm, perm, n * sizeof(int));
    
    // Generate all permutations that differ by exactly one adjacent swap
    for (int i = 0; i < n - 1; i++) {
        // Swap adjacent elements
        int temp = temp_perm[i];
        temp_perm[i] = temp_perm[i + 1];
        temp_perm[i + 1] = temp;
        
        // Compute the index of this adjacent permutation
        adj_indices[count++] = rank_permutation(temp_perm, n);
        
        // Swap back to original
        temp_perm[i] = temp_perm[i + 1];
        temp_perm[i + 1] = temp;
    }
    
    return count;
}

// Function to build the graph in CSR format required by METIS - implementing Masteronly hybrid strategy
void build_graph(idx_t **xadj, idx_t **adjncy) {
    // Total number of vertices is FACTORIAL
    long long num_perms = FACTORIAL;
    
    // Allocate CSR arrays
    *xadj = (idx_t *)malloc((num_perms + 1) * sizeof(idx_t));
    if (!*xadj) {
        printf("Failed to allocate memory for xadj array\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
    
    // Since each permutation has exactly N-1 adjacent permutations,
    // we can optimize by directly computing xadj values
    (*xadj)[0] = 0;
    for (long long i = 0; i < num_perms; i++) {
        (*xadj)[i+1] = (*xadj)[i] + (N-1);
    }
    
    // Allocate memory for adjncy based on total edge count
    long long total_edges = num_perms * (N-1);
    *adjncy = (idx_t *)malloc(total_edges * sizeof(idx_t));
    if (!*adjncy) {
        printf("Failed to allocate memory for adjacency list\n");
        free(*xadj);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
    
    // Fill adjacency list using Masteronly hybrid strategy - see slide 10/175
    // MPI is only used outside of parallel regions
    #pragma omp parallel
    {
        int thread_perm[N];
        long long adj_indices[N-1];
        
        #pragma omp for schedule(dynamic, DYNAMIC_SCHED_CHUNK)
        for (long long i = 0; i < num_perms; i++) {
            // OpenMP parallel section for numerical computation
            unrank_permutation(i, N, thread_perm);
            int adj_count = find_adjacent_permutations(thread_perm, N, adj_indices);
            
            long long pos = (*xadj)[i];
            for (int j = 0; j < adj_count; j++) {
                (*adjncy)[pos + j] = (idx_t)adj_indices[j];
            }
        }
    }
    // MPI calls would be outside the parallel region
}

// Function to generate a visual representation of a spanning tree
void generate_tree_image(int tree_idx, int *parents, long factorial, long identity_idx) {
    // Create a Graphviz context
    GVC_t *gvc = gvContext();
    
    // Create an undirected graph
    Agraph_t *graph = agopen("SpanningTree", Agundirected, NULL);
    
    // Set graph attributes
    char title[64];
    snprintf(title, sizeof(title), "Spanning Tree T%d", tree_idx + 1);
    agattr(graph, AGRAPH, "label", title);
    
    // Create nodes for permutations that participate in the tree
    Agnode_t **nodes = (Agnode_t **)malloc(factorial * sizeof(Agnode_t *));
    
    // Create permutation buffer to reuse
    int perm[N];
    
    for (long i = 0; i < factorial; i++) {
        // Create a unique node ID
        char node_id[32];
        snprintf(node_id, sizeof(node_id), "node%ld", i);
        
        // Add the node to the graph
        nodes[i] = agnode(graph, node_id, 1);
        
        // Generate the permutation
        unrank_permutation(i, N, perm);
        
        // Create a label for the node (the permutation)
        char label[64];
        snprintf(label, sizeof(label), "[");
        for (int j = 0; j < N; j++) {
            snprintf(label + strlen(label), sizeof(label) - strlen(label), "%d", perm[j]);
            if (j < N-1) snprintf(label + strlen(label), sizeof(label) - strlen(label), ",");
        }
        snprintf(label + strlen(label), sizeof(label) - strlen(label), "]");
        
        // Set the node label
        agattr(graph, AGNODE, "label", "");
        agsafeset(nodes[i], "label", label, "");
        
        // Highlight the identity node (root)
        if (i == identity_idx) {
            agsafeset(nodes[i], "color", "red", "");
            agsafeset(nodes[i], "style", "filled", "");
            agsafeset(nodes[i], "fillcolor", "lightpink", "");
            agsafeset(nodes[i], "fontweight", "bold", "");
        }
    }
    
    // Add edges based on parent relationships
    for (long i = 0; i < factorial; i++) {
        if (i != identity_idx && parents[i] >= 0 && parents[i] < factorial) {
            agedge(graph, nodes[parents[i]], nodes[i], NULL, 1);
        }
    }
    
    // Set output file name
    char filename[64];
    snprintf(filename, sizeof(filename), "tree_T%d.png", tree_idx + 1);
    
    // Render the graph to a PNG file
    gvLayout(gvc, graph, "dot");
    gvRenderFilename(gvc, graph, "png", filename);
    gvFreeLayout(gvc, graph);
    
    // Free resources
    agclose(graph);
    gvFreeContext(gvc);
    free(nodes);
}

// Modified main function implementing efficient hybrid MPI+OpenMP strategy
int main(int argc, char *argv[]) {
    double start_time_wall = omp_get_wtime();
    clock_t start_time = clock();

    // Set environment variables before initializing OpenMP
    setenv("OMP_PROC_BIND", "true", 1);  // Bind threads to processors for better performance
    setenv("OMP_WAIT_POLICY", "active", 1);  // Use active waiting for better performance
    
    // Initialize MPI with thread support for hybrid model
    int provided;
    // Request MPI_THREAD_FUNNELED - only master thread makes MPI calls
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0 && provided < MPI_THREAD_FUNNELED) {
        printf("Warning: MPI implementation does not support requested thread level\n");
    }
    
    // Initialize OpenMP after MPI
    initialize_openmp();
    
    // Initialize factorial cache - do this outside of parallel region
    init_factorial_cache();
    
    if (rank == 0) {
        printf("Running with %d MPI processes and %d OpenMP threads per process\n", 
               size, omp_get_max_threads());
        printf("OMP_PROC_BIND=%s\n", getenv("OMP_PROC_BIND"));
        printf("OMP_WAIT_POLICY=%s\n", getenv("OMP_WAIT_POLICY"));
        printf("N=%d, FACTORIAL=%lld\n", N, (long long)FACTORIAL);
    }
    
    // Skip graph construction and use range partitioning
    // Distribute vertices evenly across processes
    long long vertices_per_process = FACTORIAL / size;
    long long start_idx = rank * vertices_per_process;
    long long end_idx = (rank == size - 1) ? FACTORIAL : start_idx + vertices_per_process;
    long long local_count = end_idx - start_idx;
    
    // Print process distribution
    if (rank == 0) {
        printf("Process distribution (range-based): ");
        for (int i = 0; i < size; i++) {
            long long proc_count = (i == size - 1) ? 
                                  FACTORIAL - (i * vertices_per_process) : 
                                  vertices_per_process;
            printf("%lld ", proc_count);
        }
        printf("\n");
    }
    
    printf("Process %d has %lld vertices (range %lld to %lld)\n", 
           rank, local_count, start_idx, end_idx - 1);
    
    // Find the identity permutation index (which is always 0 in lexicographic order)
    long long identity_idx = 0;
    
    if (rank == 0) {
        printf("Identity permutation index: %lld\n", identity_idx);
    }
    
    // Check if this process owns the identity permutation
    bool has_identity = (identity_idx >= start_idx && identity_idx < end_idx);
    
    // Create output files for each tree (rank 0)
    FILE **tree_files = NULL;
    if (rank == 0) {
        tree_files = (FILE **)malloc((N-1) * sizeof(FILE *));
        for (int t = 0; t < N-1; t++) {
            char filename[64];
            snprintf(filename, sizeof(filename), "tree_T%d_parents.txt", t+1);
            tree_files[t] = fopen(filename, "w");
            if (!tree_files[t]) {
                printf("Error: Failed to open output file %s\n", filename);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Write header
            fprintf(tree_files[t], "# Parent relationships for Tree %d\n", t+1);
            fprintf(tree_files[t], "# Vertex -> Parent\n");
            
            // Write identity permutation as self-parent (root)
            fprintf(tree_files[t], "%lld -> %lld\n", identity_idx, identity_idx);
            fflush(tree_files[t]); // Flush to ensure header is written before parallel work begins
        }
    }
    
    // Create thread-local file buffers to reduce contention
    int num_threads = omp_get_max_threads();
    
    // Create temporary output files for each process and tree
    char **temp_filenames = (char **)malloc((N-1) * sizeof(char *));
    FILE **temp_files = (FILE **)malloc((N-1) * sizeof(FILE *));
    
    // Create thread-local buffers to reduce critical section overhead
    char ***thread_buffers = (char ***)malloc(num_threads * sizeof(char **));
    size_t buffer_size = 64*1024; // 64KB per buffer
    
    for (int t = 0; t < num_threads; t++) {
        thread_buffers[t] = (char **)malloc((N-1) * sizeof(char *));
        for (int i = 0; i < N-1; i++) {
            thread_buffers[t][i] = (char *)malloc(buffer_size);
            thread_buffers[t][i][0] = '\0'; // Initialize to empty string
        }
    }
    
    for (int t = 0; t < N-1; t++) {
        temp_filenames[t] = (char *)malloc(64 * sizeof(char));
        snprintf(temp_filenames[t], 64, "tree_T%d_rank%d.tmp", t+1, rank);
        temp_files[t] = fopen(temp_filenames[t], "w");
        if (!temp_files[t]) {
            printf("Error: Failed to open temporary file %s\n", temp_filenames[t]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Set files to unbuffered to avoid memory contention
        setvbuf(temp_files[t], NULL, _IONBF, 0);
    }
    
    // Print actual thread count being used
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("Process %d actually using %d OpenMP threads\n", rank, omp_get_num_threads());
            fflush(stdout);
        }
    }
    
    // Thread statistics
    double *thread_times = (double *)calloc(num_threads, sizeof(double));
    long long *thread_vertices = (long long *)calloc(num_threads, sizeof(long long));
    
    double compute_start = MPI_Wtime();
    
    // Process local vertices in chunks to reduce memory usage
    long long num_chunks = (local_count + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Adjust chunk size for better load balancing
    long long effective_chunk_size = CHUNK_SIZE;
    if (USE_IMPROVED_WORKLOAD_BALANCE) {
        // Make chunk size smaller when more threads are used
        effective_chunk_size = CHUNK_SIZE / (num_threads > 1 ? num_threads / 2 : 1);
        if (effective_chunk_size < 10) effective_chunk_size = 10;
        
        num_chunks = (local_count + effective_chunk_size - 1) / effective_chunk_size;
    }
    
    // Use guided scheduling for better load balancing
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        double thread_start = omp_get_wtime();
        
        // Thread-local permutation buffers
        int current_perm[N]; 
        int parent_perm[N];
        size_t buffer_used[N-1] = {0};
        
        // Each thread gets a contiguous range of chunks
        #pragma omp for schedule(guided)
        for (long long chunk = 0; chunk < num_chunks; chunk++) {
            long long chunk_start = start_idx + chunk * effective_chunk_size;
            long long chunk_end = chunk_start + effective_chunk_size;
            if (chunk_end > end_idx) chunk_end = end_idx;
            long long chunk_size = chunk_end - chunk_start;
            
            // Process all vertices in this chunk
            for (long long i = 0; i < chunk_size; i++) {
                long long v_idx = chunk_start + i;
                thread_vertices[thread_id]++;
                
                // Skip the identity permutation since it's the root
                if (v_idx == identity_idx) continue;
                
                // Generate the permutation for this vertex
                unrank_permutation(v_idx, N, current_perm);
                
                // Process all trees
                for (int t = 0; t < N-1; t++) {
                    // Get parent of this vertex in tree t+1
                    parent1(current_perm, N, t+1, parent_perm);
                    
                    // Find the index of the resulting parent permutation
                    long long result_idx = rank_permutation(parent_perm, N);
                    
                    // Format the result into thread-local buffer
                    int len = snprintf(thread_buffers[thread_id][t] + buffer_used[t], 
                                      buffer_size - buffer_used[t],
                                      "%lld -> %lld\n", v_idx, result_idx);
                    
                    buffer_used[t] += len;
                    
                    // If buffer is getting full, flush it to the file
                    if (buffer_used[t] > buffer_size - 100) {
                        #pragma omp critical(file_write)
                        {
                            fprintf(temp_files[t], "%s", thread_buffers[thread_id][t]);
                        }
                        buffer_used[t] = 0;
                        thread_buffers[thread_id][t][0] = '\0';
                    }
                }
            }
            
            // Occasionally report progress - less frequently to reduce overhead
            if (thread_id == 0 && chunk % 50 == 0) {
                printf("Rank %d: %.1f%% complete\n", rank, 100.0 * chunk / num_chunks);
                fflush(stdout);
            }
        }
        
        // Flush any remaining data in thread buffers
        for (int t = 0; t < N-1; t++) {
            if (buffer_used[t] > 0) {
                #pragma omp critical(file_write)
                {
                    fprintf(temp_files[t], "%s", thread_buffers[thread_id][t]);
                }
            }
        }
        
        // Record thread time
        thread_times[thread_id] = MPI_Wtime() - thread_start;
    }
    
    // Free thread buffers
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < N-1; i++) {
            free(thread_buffers[t][i]);
        }
        free(thread_buffers[t]);
    }
    free(thread_buffers);
    
    // Close temporary files
    for (int t = 0; t < N-1; t++) {
        fclose(temp_files[t]);
    }
    free(temp_files);
    
    double compute_end = MPI_Wtime();
    
    // Print thread statistics
    if (rank == 0) {
        printf("\nProcess %d thread statistics:\n", rank);
        for (int t = 0; t < num_threads; t++) {
            printf("  Thread %d processed %lld vertices in %.2f seconds\n", 
                   t, thread_vertices[t], thread_times[t]);
        }
    }
    
    // Free thread statistics
    free(thread_times);
    free(thread_vertices);
    
    if (rank == 0) {
        printf("Computation of parent relationships completed in %.2f seconds\n", 
               compute_end - compute_start);
    }
    
    // Synchronize before entering reduction phase
    MPI_Barrier(MPI_COMM_WORLD);
    double reduction_start = MPI_Wtime();
    
    // Now gather results on rank 0 - MPI communication handled by master thread
    if (rank == 0) {
        printf("Gathering results...\n");
        
        // For each tree, gather results from all processes
        for (int t = 0; t < N-1; t++) {
            printf("Gathering tree %d of %d...\n", t+1, N-1);
            
            // First, process rank 0's own results
            FILE *my_temp = fopen(temp_filenames[t], "r");
            if (my_temp) {
                // Use large buffer for more efficient I/O
                char buffer[1024*1024]; // 1MB buffer
                size_t bytes_read;
                
                while ((bytes_read = fread(buffer, 1, sizeof(buffer), my_temp)) > 0) {
                    fwrite(buffer, 1, bytes_read, tree_files[t]);
                }
                
                fclose(my_temp);
                remove(temp_filenames[t]); // Clean up temp file
            }
            
            // Now receive data from other processes one at a time
            for (int src = 1; src < size; src++) {
                printf("Receiving from process %d of %d for tree %d...\n", src, size-1, t+1);
                
                // Receive temp filename
                MPI_Status status;
                char remote_filename[256];
                // MPI communication by master thread only
                MPI_Recv(remote_filename, 256, MPI_CHAR, src, t, MPI_COMM_WORLD, &status);
                
                // Wait for confirmation that file is ready
                int ready;
                MPI_Recv(&ready, 1, MPI_INT, src, t, MPI_COMM_WORLD, &status);
                
                // Open the remote file and copy contents using efficient buffering
                FILE *remote_file = fopen(remote_filename, "r");
                if (remote_file) {
                    char buffer[1024*1024]; // 1MB buffer
                    size_t bytes_read;
                    
                    while ((bytes_read = fread(buffer, 1, sizeof(buffer), remote_file)) > 0) {
                        fwrite(buffer, 1, bytes_read, tree_files[t]);
                    }
                    
                    fclose(remote_file);
                    remove(remote_filename); // Clean up
                } else {
                    printf("Warning: Could not open file %s from rank %d\n", remote_filename, src);
                }
            }
            
            // Close and flush the tree file
            fflush(tree_files[t]);
        }
    } else {
        // Send our temp filenames to rank 0
        for (int t = 0; t < N-1; t++) {
            // MPI communication handled by main thread only
            
            // Send the temporary filename
            MPI_Send(temp_filenames[t], 64, MPI_CHAR, 0, t, MPI_COMM_WORLD);
            
            // Send ready signal
            int ready = 1;
            MPI_Send(&ready, 1, MPI_INT, 0, t, MPI_COMM_WORLD);
        }
    }
    
    // Clean up temp filenames memory
    for (int t = 0; t < N-1; t++) {
        free(temp_filenames[t]);
    }
    free(temp_filenames);
    
    double reduction_end = MPI_Wtime();
    
    // Continue with existing code for closing output files, image generation, etc.
    // Create output files for each tree (rank 0)
    if (rank == 0) {
        for (int t = 0; t < N-1; t++) {
            fclose(tree_files[t]);
        }
        free(tree_files);
        
        printf("Data reduction completed in %.2f seconds\n", reduction_end - reduction_start);
        printf("Tree parent relationships have been written to tree_T*_parents.txt files\n");
        
        // Generate tree images if N <= 8 - master thread only operation
        if (N <= 8) {
            printf("Generating tree images for N=%d...\n", N);
            
            for (int t = 1; t <= N-1; t++) {
                // Open the tree file for reading
                char filename[64];
                snprintf(filename, sizeof(filename), "tree_T%d_parents.txt", t);
                FILE *f = fopen(filename, "r");
                if (!f) {
                    printf("Error: Could not open tree file for reading to generate image\n");
                    continue;
                }
                
                // Create an array to store parent relationships
                int *tree_parents = (int *)malloc(FACTORIAL * sizeof(int));
                memset(tree_parents, -1, FACTORIAL * sizeof(int));
                
                // Skip header lines
                char line[256];
                fgets(line, sizeof(line), f); // Skip header
                fgets(line, sizeof(line), f); // Skip header
                
                // Read all parent relationships
                while (fgets(line, sizeof(line), f)) {
                    long long vertex, parent;
                    if (sscanf(line, "%lld -> %lld", &vertex, &parent) == 2) {
                        tree_parents[vertex] = (int)parent;
                    }
                }
                fclose(f);
                
                // Generate the tree image
                generate_tree_image(t-1, tree_parents, FACTORIAL, identity_idx);
                
                // Free the parent array
                free(tree_parents);
                
                printf("Generated image for tree %d\n", t);
            }
            
            printf("Generated %d tree images in PNG format\n", N-1);
        } else {
            printf("Tree visualization skipped for N > 8\n");
        }
        
        // Sample and print a limited number of parent relationships for each tree
        printf("\nSampling Indexed Spanning Trees (first %d entries):\n", OUTPUT_SAMPLE_COUNT);
        
        for (int t = 1; t <= N-1; t++) {
            printf("Tree %d (sample):\n", t);
            
            // Open the tree file again for reading
            char filename[64];
            snprintf(filename, sizeof(filename), "tree_T%d_parents.txt", t);
            FILE *f = fopen(filename, "r");
            if (!f) {
                printf("Error: Could not open tree file for reading\n");
                continue;
            }
            
            // Skip header lines
            char line[256];
            fgets(line, sizeof(line), f); // Skip header
            fgets(line, sizeof(line), f); // Skip header
            
            // Print identity vertex first
            int perm_temp[N], parent_temp[N];
            unrank_permutation(identity_idx, N, perm_temp);
            printf("  Vertex ");
            print_perm(perm_temp, N);
            printf(" -> Parent ");
            print_perm(perm_temp, N);
            printf(" (Root)\n");
            
            // Sample some other entries
            long long sample_interval = FACTORIAL / OUTPUT_SAMPLE_COUNT;
            if (sample_interval < 1) sample_interval = 1;
            
            // Skip the identity line
            fgets(line, sizeof(line), f);
            
            for (int i = 0; i < OUTPUT_SAMPLE_COUNT - 1 && !feof(f); i++) {
                // Get the next line
                if (fgets(line, sizeof(line), f)) {
                    // Parse the vertex and parent
                    long long vertex, parent;
                    sscanf(line, "%lld -> %lld", &vertex, &parent);
                    
                    // Generate and print the permutations
                    unrank_permutation(vertex, N, perm_temp);
                    printf("  Vertex ");
                    print_perm(perm_temp, N);
                    printf(" -> Parent ");
                    unrank_permutation(parent, N, parent_temp);
                    print_perm(parent_temp, N);
                    printf("\n");
                }
                
                // Skip ahead to next sample position
                for (long long skip = 1; skip < sample_interval && !feof(f); skip++) {
                    fgets(line, sizeof(line), f);
                }
            }
            
            fclose(f);
            printf("\n");
        }
    }
    
    // Record final timing information - only master thread handles this
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double elapsed_wall_time = omp_get_wtime() - start_time_wall;
    
    // Report timing statistics - following MPI_THREAD_FUNNELED model
    if (rank == 0) {
        printf("Performance summary:\n");
        printf("Total elapsed time (CPU): %.2f seconds\n", elapsed_time);
        printf("Total elapsed time (wall): %.2f seconds\n", elapsed_wall_time);
        printf("Computation phase: %.2f seconds\n", compute_end - compute_start);
        printf("Reduction phase: %.2f seconds\n", reduction_end - reduction_start);
    }
    
    // Gather timing statistics from all processes
    if (rank == 0) {
        double *proc_times = (double*)malloc(size * sizeof(double));
        // Only collect times if we have multiple processes
        if (size > 1) {
            // First record our own time
            proc_times[0] = compute_end - compute_start;
            
            // Then receive from other processes
            for (int p = 1; p < size; p++) {
                MPI_Status status;
                MPI_Recv(&proc_times[p], 1, MPI_DOUBLE, p, 999, MPI_COMM_WORLD, &status);
            }
            
            // Print all timing information
            printf("\nComputation times by process:\n");
            for (int p = 0; p < size; p++) {
                printf("  Rank %d: %.2f seconds\n", p, proc_times[p]);
            }
        }
        free(proc_times);
    } else {
        // Other ranks send their computation times to rank 0
        double my_compute_time = compute_end - compute_start;
        MPI_Send(&my_compute_time, 1, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
    }
    
    // Ensure all threads reach MPI_Finalize (required by MPI_THREAD_FUNNELED)
    #pragma omp barrier
    
    MPI_Finalize();
    return 0;
}