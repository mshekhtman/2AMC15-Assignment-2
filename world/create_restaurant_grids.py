"""
create_restaurant_grids.py - Specialized grids for simplified 10D environment
"""

import numpy as np
from pathlib import Path

# World may not be importable, depending on how you have set up your
# conda/pip/venv environment. Here we try to fix that by forcing the world to
# be in your python path. If it still doesn't work, come to a tutorial, look up
# how to fix module import errors, or ask ChatGPT.
try:
    from world import Grid
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from world import Grid

def create_simple_restaurant():
    """Create a simple restaurant layout optimized for 10D state testing."""
    # Create a 12x10 restaurant layout (manageable size for testing)
    grid = Grid(12, 10)
    
    # Clear the interior
    grid.cells[1:-1, 1:-1] = 0
    
    # Kitchen area (left side)
    kitchen_obstacles = [
        (2, 2), (3, 2), (4, 2),  # Counter
        (2, 3), (3, 3),          # Equipment
        (2, 7), (3, 7), (4, 7)   # Prep area
    ]
    
    for x, y in kitchen_obstacles:
        grid.place_object(x, y, "obstacle")
    
    # Dining tables (right side) - arranged for clear navigation paths
    table_positions = [
        (7, 3), (8, 3),      # Table 1
        (10, 3), (11, 3),    # Table 2
        (7, 6), (8, 6),      # Table 3  
        (10, 6), (11, 6)     # Table 4
    ]
    
    for x, y in table_positions:
        grid.place_object(x, y, "obstacle")
    
    # Delivery targets (next to tables) - positioned to test direction finding
    target_positions = [
        (6, 3),   # Target 1 - west of table 1
        (9, 3),   # Target 2 - between tables 1&2
        (6, 6),   # Target 3 - west of table 3
        (9, 6)    # Target 4 - between tables 3&4
    ]
    
    for x, y in target_positions:
        grid.place_object(x, y, "target")
    
    # Save the grid
    save_path = Path("grid_configs/A2/simple_restaurant.npy")
    grid.save_grid_file(save_path)
    print(f"Simple restaurant grid saved to {save_path}")
    return grid

def create_corridor_test():
    """Create a corridor layout to test clearance detection."""
    grid = Grid(15, 8)
    
    # Clear interior
    grid.cells[1:-1, 1:-1] = 0
    
    # Create narrow corridor with obstacles
    # Vertical obstacles creating narrow passages
    corridor_obstacles = [
        (3, 2), (3, 3), (3, 5), (3, 6),  # Left wall with gaps
        (7, 2), (7, 3), (7, 5), (7, 6),  # Middle wall with gaps
        (11, 2), (11, 3), (11, 5), (11, 6) # Right wall with gaps
    ]
    
    for x, y in corridor_obstacles:
        grid.place_object(x, y, "obstacle")
    
    # Targets at end of corridors
    targets = [(5, 4), (9, 4), (13, 4)]  # In the gaps
    
    for x, y in targets:
        grid.place_object(x, y, "target")
    
    save_path = Path("grid_configs/A2/corridor_test.npy")
    grid.save_grid_file(save_path)
    print(f"Corridor test grid saved to {save_path}")
    return grid

def create_maze_challenge():
    """Create a more complex maze to test navigation abilities."""
    grid = Grid(16, 12)
    
    # Clear interior
    grid.cells[1:-1, 1:-1] = 0
    
    # Create maze-like structure
    maze_obstacles = [
        # Outer maze walls
        (3, 2), (3, 3), (3, 4), (3, 5), (3, 7), (3, 8), (3, 9),
        (5, 2), (5, 4), (5, 6), (5, 8), (5, 10),
        (7, 2), (7, 3), (7, 5), (7, 7), (7, 9), (7, 10),
        (9, 2), (9, 4), (9, 6), (9, 8), (9, 10),
        (11, 2), (11, 3), (11, 4), (11, 6), (11, 7), (11, 8), (11, 9),
        (13, 3), (13, 5), (13, 7), (13, 9)
    ]
    
    for x, y in maze_obstacles:
        grid.place_object(x, y, "obstacle")
    
    # Targets in different sections of maze
    targets = [
        (4, 6),   # Target 1 - requires navigation through gap
        (8, 4),   # Target 2 - center area
        (12, 5),  # Target 3 - right side
        (6, 9),   # Target 4 - bottom area
        (14, 8)   # Target 5 - far corner
    ]
    
    for x, y in targets:
        grid.place_object(x, y, "target")
    
    save_path = Path("grid_configs/A2/maze_challenge.npy")
    grid.save_grid_file(save_path)
    print(f"Maze challenge grid saved to {save_path}")
    return grid

def create_open_space():
    """Create an open space layout for basic testing."""
    grid = Grid(10, 8)
    
    # Clear most of interior
    grid.cells[1:-1, 1:-1] = 0
    
    # Just a few scattered obstacles
    sparse_obstacles = [(4, 3), (6, 5), (7, 2)]
    
    for x, y in sparse_obstacles:
        grid.place_object(x, y, "obstacle")
    
    # Targets spread around
    targets = [(3, 6), (8, 6), (5, 2)]
    
    for x, y in targets:
        grid.place_object(x, y, "target")
    
    save_path = Path("grid_configs/A2/open_space.npy")
    grid.save_grid_file(save_path)
    print(f"Open space grid saved to {save_path}")
    return grid

def create_assignment_grid():
    """Create the main grid for Assignment 2 evaluation."""
    # Create a 14x11 grid (similar to A1_grid.npy dimensions)
    grid = Grid(14, 11)
    
    # Clear interior
    grid.cells[1:-1, 1:-1] = 0
    
    # Restaurant layout with clear structure
    # Kitchen area (top-left)
    kitchen = [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3)]
    
    # Dining room tables (main area)
    dining_tables = [
        # Row 1
        (6, 3), (7, 3),
        (9, 3), (10, 3),
        (12, 3), (13, 3),
        
        # Row 2  
        (6, 6), (7, 6),
        (9, 6), (10, 6),
        (12, 6), (13, 6),
        
        # Row 3
        (6, 8), (7, 8),
        (9, 8), (10, 8)
    ]
    
    # Central service station
    service_station = [(8, 5)]
    
    all_obstacles = kitchen + dining_tables + service_station
    
    for x, y in all_obstacles:
        grid.place_object(x, y, "obstacle")
    
    # Delivery targets positioned strategically
    # Next to tables but ensuring navigable paths
    delivery_targets = [
        (5, 3),   # Table 1 service
        (8, 3),   # Between tables
        (11, 3),  # Table 3 service
        (5, 6),   # Table 4 service  
        (11, 6),  # Table 6 service
        (8, 8),   # Bottom area service
        (4, 8)    # Corner service
    ]
    
    for x, y in delivery_targets:
        grid.place_object(x, y, "target")
    
    save_path = Path("grid_configs/A2/assignment2_main.npy")
    grid.save_grid_file(save_path)
    print(f"Assignment 2 main grid saved to {save_path}")
    return grid

def visualize_grid_layout(grid_path: Path):
    """Print a text visualization of the grid layout."""
    if not grid_path.exists():
        print(f"Grid {grid_path} not found.")
        return
    
    grid = Grid.load_grid(grid_path)
    
    print(f"\nGrid Layout: {grid_path.name}")
    print("Legend: . = empty, # = obstacle/boundary, T = target, C = charger")
    print("-" * (grid.n_cols * 2 + 1))
    
    for y in range(grid.n_rows):
        line = "|"
        for x in range(grid.n_cols):
            cell_value = grid.cells[x, y]
            if cell_value == 0:
                line += " ."
            elif cell_value == 1 or cell_value == 2:
                line += " #"
            elif cell_value == 3:
                line += " T"
            elif cell_value == 4:
                line += " C"
            else:
                line += " ?"
        line += "|"
        print(line)
    
    print("-" * (grid.n_cols * 2 + 1))
    
    # Count elements
    targets = np.sum(grid.cells == 3)
    obstacles = np.sum((grid.cells == 1) | (grid.cells == 2))
    empty = np.sum(grid.cells == 0)
    
    print(f"Targets: {targets}, Obstacles: {obstacles}, Empty cells: {empty}")

if __name__ == "__main__":
    # Ensure grid_configs directory exists
    Path("grid_configs/A2").mkdir(exist_ok=True)
    
    print("Creating specialized restaurant grids for 10D environment...")
    
    # Create all test grids
    create_open_space()
    create_simple_restaurant()
    create_corridor_test() 
    create_maze_challenge()
    create_assignment_grid()
    
    print("\nAll grids created successfully!")
    print("\nGrid visualizations:")
    
    # Show layouts
    for grid_name in ["open_space.npy", "simple_restaurant.npy", 
                     "corridor_test.npy", "assignment2_main.npy"]:
        visualize_grid_layout(Path(f"grid_configs/A2/{grid_name}"))
    
    print("\nRecommended usage:")
    print("- open_space.npy: Basic agent testing")
    print("- simple_restaurant.npy: Feature validation") 
    print("- corridor_test.npy: Clearance detection testing")
    print("- maze_challenge.npy: Advanced navigation")
    print("- assignment2_main.npy: Main evaluation grid")