# Machine Proximity Relationship Visualization Tool

A Python tool for visualizing component relationships and proximity requirements in manufacturing or facility layouts. The tool generates interactive force-directed graphs that show how different components should be positioned relative to each other based on their relationship ratings.

## Features

- **Force-directed graph layout** - Components are automatically positioned based on their relationship strengths
- **Multiple rating types** - Support for 6 different relationship ratings (A, E, I, O, U, X)
- **Customizable seeds** - Generate different layout variations by changing the random seed
- **Dual output** - Creates visualizations with and without "Unimportant" (U) relationships
- **Component legend** - Automatically generates a reference legend mapping numbers to component names
- **Smart positioning** - Legend automatically positions itself in the clearest corner

## Rating System

The tool uses the following relationship ratings:

| Rating | Meaning | Color | Line Style | Behavior |
|--------|---------|-------|------------|----------|
| **A** | Absolutely Necessary | Orange | Solid (4.0 width) | Strong attraction |
| **E** | Especially Important | Blue | Solid (3.0 width) | Strong attraction |
| **I** | Important | Green | Solid (2.5 width) | Medium attraction |
| **O** | Ordinary Importance | Black | Solid (2.0 width) | Weak attraction |
| **U** | Unimportant | Grey | Solid (1.0 width) | Very weak attraction |
| **X** | Not Desirable | Red | Dashed (2.5 width) | Repulsion (push apart) |

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup with Virtual Environment (Recommended)

1. **Clone or download** this repository to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd path/to/IMS045
   ```

3. **Create a virtual environment**:
   
   On Windows:
   ```bash
   python -m venv .venv
   ```
   
   On macOS/Linux:
   ```bash
   python3 -m venv .venv
   ```

4. **Activate the virtual environment**:
   
   On Windows (PowerShell):
   ```bash
   .venv\Scripts\Activate.ps1
   ```
   
   On Windows (Command Prompt):
   ```bash
   .venv\Scripts\activate.bat
   ```
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

5. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project uses the following Python packages (automatically installed via requirements.txt):

- `matplotlib` (3.10.8) - Visualization and plotting
- `networkx` (3.6.1) - Graph creation and algorithms
- `numpy` (2.4.1) - Numerical computations

## Usage

### Basic Usage

Run the visualization with default settings (uses `relations.json` and seed=42):

```bash
python visualize_relations.py
```

### Specify Custom Input File

```bash
python visualize_relations.py path/to/your/relations.json
```

### Generate Different Layouts

Use different seeds to create alternative spatial arrangements:

```bash
python visualize_relations.py --seed 100
python visualize_relations.py -s 200
```

### Combine Options

```bash
python visualize_relations.py data.json --seed 456
```

### Get Help

```bash
python visualize_relations.py --help
```

## Input File Format

The tool expects a JSON file with the following structure:

```json
{
    "components": [
        "rough turning",
        "fine turning",
        "hardening",
        "cooling",
        ...
    ],
    "relations": [
        {"from": 1, "to": 0, "rating": "A"},
        {"from": 2, "to": 1, "rating": "E"},
        {"from": 3, "to": 0, "rating": "I"},
        ...
    ]
}
```

### Format Details:

- **components**: Array of component names (strings)
  - Components are automatically numbered starting from 0
  - Display numbers are 1-indexed for user-friendliness

- **relations**: Array of relationship objects
  - `from`: Index of source component (0-based)
  - `to`: Index of target component (0-based)
  - `rating`: Relationship rating (A, E, I, O, U, or X)

**Note**: Any component pairs without an explicit relation are automatically assigned a "U" (Unimportant) rating.

## Output Files

The tool generates three PNG files in the same directory as the input file:

1. **`relationship_diagram_with_U.png`** - Complete diagram including all U (Unimportant) relationships
2. **`relationship_diagram_without_U.png`** - Cleaner diagram with U relationships hidden
3. **`component_legend.png`** - Reference legend showing component numbers and names

All diagrams include the seed value in the legend for reproducibility.

## How It Works

### 1. Data Loading
The tool reads component names and their relationships from the JSON file.

### 2. Graph Construction
- Creates a NetworkX graph with components as nodes
- Adds edges with relationship ratings and force weights
- Auto-generates "U" ratings for undefined component pairs

### 3. Force-Directed Layout Algorithm
The layout algorithm simulates physical forces between nodes:

- **Attraction Forces**: Strong relationships (A, E) pull components together
  - Stronger ratings = closer ideal distance
  - Spring-like force proportional to distance from ideal

- **Repulsion Forces**: 
  - Base repulsion between all nodes prevents overlap
  - X-rated relationships actively push components apart
  - Uses inverse square law for realistic spacing

- **Simulation Process**:
  - Starts with randomized circular positions (varied by seed)
  - Iterates 500 times, applying forces and moving nodes
  - Temperature-based cooling gradually reduces movement
  - Enforces minimum distances for clarity

### 4. Edge Curvature Calculation
- Groups edges by angle to identify parallel connections
- Assigns alternating curvatures to prevent visual overlaps
- Stronger relationships get less curvature (more direct lines)

### 5. Visualization
- Draws curved edges using quadratic Bezier curves
- Renders nodes as white circles with black borders
- Adds numbered labels (1-indexed)
- Positions legend in the clearest corner automatically

## Command-Line Options

```
usage: visualize_relations.py [-h] [-s SEED] [json_file]

positional arguments:
  json_file             Path to JSON file with component relations
                        (default: relations.json)

options:
  -h, --help            Show this help message and exit
  -s SEED, --seed SEED  Random seed for layout generation (default: 42)
                        Use different seeds to create different layouts
```

## Examples

### Example 1: Generate standard layout
```bash
python visualize_relations.py
```

### Example 2: Try different spatial arrangements
```bash
python visualize_relations.py -s 1
python visualize_relations.py -s 50
python visualize_relations.py -s 999
```

### Example 3: Process custom data
```bash
python visualize_relations.py factory_layout.json -s 42
```

## Troubleshooting

### Virtual environment not activating
- On Windows, you may need to enable script execution:
  ```bash
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Import errors
- Ensure you've activated the virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

### Graph looks cluttered
- Try different seed values to find a clearer layout
- Use the "without U" version for a cleaner visualization
- Consider reducing the number of components if too many exist

### Components overlap
- The algorithm enforces minimum distances, but with many nodes, some proximity is inevitable
- Try increasing the `min_distance` parameter in the code (line ~118)
- Use a larger figure size in the code (line ~352)

## Technical Details

- **Language**: Python 3.7+
- **Graph Library**: NetworkX for graph data structures
- **Visualization**: Matplotlib for rendering
- **Algorithm**: Custom force-directed layout with configurable attraction/repulsion
- **Edge Drawing**: Quadratic Bezier curves for smooth, curved connections

## License

This tool is provided as-is for educational and commercial use.

## Author

Created for IMS045 course - Facility Layout and Material Handling
