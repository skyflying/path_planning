# Path Planning

This project implements various path planning algorithms using geographic data. The initial implementation includes Dijkstra's algorithm, and the framework is designed to accommodate additional algorithms for testing and comparison. T

### Purpose

This project aims to find the shortest path between two points while avoiding obstacles, simulating human walking paths. The primary goal is to test and compare different path planning algorithms. The initial implementation uses Dijkstra's algorithm.

### Path Planning Algorithms

The project is structured to allow the implementation of various path planning algorithms. Currently, it includes the following:

- **Dijkstra's Algorithm**: An algorithm for finding the shortest paths between nodes in a graph. It is particularly useful for graphs with non-negative weights.

### Data Structures

The project uses several key data structures:

- **GeoDataFrame**: Used to handle geographic data from shapefiles.
- **Graph**: Utilized by `networkx` to represent the network of paths.
- **Node**: Represents a point in the grid with coordinates, cost, and parent node index.

### Key Features

- **Obstacle Handling**: The algorithm takes into account obstacles and calculates paths that avoid them.
- **Visualization**: The project includes functions to visualize the paths, obstacles, and boundary polygons using `matplotlib`.
- **Shapefile Handling**: The project can load and save paths to shapefiles, making it easy to integrate with GIS tools.
