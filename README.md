# Path Planning

This project implements various path planning algorithms using geographic data. The initial implementation includes Dijkstra's algorithm, and the framework is designed to accommodate additional algorithms for testing and comparison.

## Purpose

This project aims to find the shortest path between two points while avoiding obstacles, simulating human walking paths, designed to find the shortest path for a robot from a start point to a goal point in an environment with obstacles. Below is a detailed breakdown of the code. In ording to find the best path, in second phases, I try to use the road data from OSM for finding the best path.

## Path Planning Algorithms

It  is structured to allow the implementation of various path planning algorithms. Currently, it includes the following:

- **Dijkstra's Algorithm**
-  **A* Algorithm**
-  **Path Planning with road**

## Data Structures

The project uses several key data structures:

- **GeoDataFrame**: Used to handle geographic data from shapefiles.
- **Graph**: Utilized by `networkx` to represent the network of paths.
- **Node**: Represents a point in the grid with coordinates, cost, and parent node index.

## Key Features

- **Obstacle Handling**: Accounts for the radius and buffers obstacles to ensure collision-free paths.
- **Visualization**: Supports animated visualization of the path planning process and plots the final path along with obstacles.
- **Geospatial Data Processing**: load and save paths to shapefiles, making it easy to integrate with GIS tools.

## Algorithm: 

### Dijkstra
- **Initialization**: Computes grid boundaries and resolution based on start and goal points.
- **Open and Closed Sets**: Uses dictionaries to store nodes to be explored and nodes already explored.
- **Node Expansion**: Expands the node with the lowest cost from the open set and checks its neighbors.
- **Termination**: The algorithm terminates when the goal node is found or the open set is empty.
- **Path Backtracking**: Backtracks the path using parent indices.

### A* 
- **Initialization**: Computes grid boundaries and resolution based on start and goal points.
- **Open and Closed Sets**: Uses dictionaries to store nodes to be explored and nodes already explored.
- **Node Expansion**: Expands the node with the lowest cost from the open set and checks its neighbors.
- **Termination**: The algorithm terminates when the goal node is found or the open set is empty.
- **Path Backtracking**: Backtracks the path using parent indices.

## Result

### Dijkstra

- Processing
  
![image](https://github.com/user-attachments/assets/8225d1fb-85f0-45d9-8841-8061ae9d9071)

- Result
  
![Dijkstra](https://github.com/skyflying/path_planning/blob/main/Result/Dijkstra_result.png)

### A*

- Result

![A star](https://github.com/skyflying/path_planning/blob/main/Result/Figure_2.png)

### A* with roads planning



## Future Work
Future work could include:

1. Implementing additional path planning algorithms (e.g., A*).
2. Enhancing obstacle handling with more complex geometries.
3. Improving performance for large datasets.

