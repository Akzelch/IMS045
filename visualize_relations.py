"""
Machine Proximity Relationship Visualization Tool

Rating System:
    A: Absolutely Necessary (Orange)
    E: Especially Important (Blue)
    I: Important (Green)
    O: Ordinary Importance (Black)
    U: Unimportant (Thin Grey)
    X: Not Desirable (Red)

Input: JSON file with components and their relations
Output: Two PNG files - relationship diagram and component legend
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpl_patches
import networkx as nx
import numpy as np
import sys
from pathlib import Path


# Color and style configuration for each rating
RATING_STYLES = {
    'A': {'color': 'orange', 'width': 4.0, 'label': 'A - Absolutely Necessary'},
    'E': {'color': 'blue', 'width': 3.0, 'label': 'E - Especially Important'},
    'I': {'color': 'green', 'width': 2.5, 'label': 'I - Important'},
    'O': {'color': 'black', 'width': 2.0, 'label': 'O - Ordinary Importance'},
    'U': {'color': 'grey', 'width': 1.0, 'label': 'U - Unimportant'},
    'X': {'color': 'red', 'width': 2.5, 'label': 'X - Not Desirable', 'style': 'dashed'}
}

# Force weights for layout: positive = attraction, negative = repulsion
RATING_FORCES = {
    'A': 6.0,   # Strongest attraction
    'E': 4.0,   # Strong attraction
    'I': 2.5,   # Medium attraction
    'O': 1.5,   # Weak attraction
    'U': 0.5,   # Very weak attraction
    'X': -3.0   # Repulsion (push apart)
}


def load_relations(json_file: str) -> dict:
    """Load component relations from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_relationship_graph(data: dict) -> nx.Graph:
    """Create a NetworkX graph from the relationship data.
    
    Missing relations between components are automatically assigned 'U' (Unimportant).
    """
    G = nx.Graph()
    
    # Add nodes (components represented by numbers)
    num_components = len(data['components'])
    for i in range(num_components):
        G.add_node(i)
    
    # Track which pairs have explicit relations
    defined_pairs = set()
    
    # Add edges with ratings and force weights from JSON
    for relation in data['relations']:
        from_node = relation['from']
        to_node = relation['to']
        rating = relation['rating'].upper()
        force = RATING_FORCES.get(rating, 1.0)
        G.add_edge(from_node, to_node, rating=rating, force=force)
        # Track this pair (store as sorted tuple to handle both directions)
        defined_pairs.add((min(from_node, to_node), max(from_node, to_node)))
    
    # Auto-generate U (Unimportant) relations for any missing pairs
    for i in range(num_components):
        for j in range(i + 1, num_components):
            pair = (i, j)
            if pair not in defined_pairs:
                force = RATING_FORCES.get('U', 0.5)
                G.add_edge(i, j, rating='U', force=force)
    
    return G


def force_directed_layout(G: nx.Graph, iterations: int = 500, seed: int = 42) -> dict:
    """
    Custom force-directed layout that respects attraction/repulsion based on ratings.
    
    - Strong ratings (A, E) pull nodes together
    - X ratings push nodes apart
    - All nodes have base repulsion to prevent overlap
    """
    np.random.seed(seed)
    
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Initialize positions in a circle
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    positions = {node: np.array([2.0 * np.cos(angles[i]), 2.0 * np.sin(angles[i])]) 
                 for i, node in enumerate(nodes)}
    
    # Simulation parameters
    temperature = 1.0  # Initial movement range
    cooling_rate = 0.995  # How fast temperature decreases
    min_distance = 0.8  # Minimum distance between nodes for clarity
    
    for iteration in range(iterations):
        forces = {node: np.array([0.0, 0.0]) for node in nodes}
        
        # Calculate repulsive forces between ALL node pairs (base repulsion for spacing)
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                diff = positions[node1] - positions[node2]
                dist = np.linalg.norm(diff)
                if dist < 0.01:
                    dist = 0.01
                    diff = np.random.rand(2) - 0.5
                
                # Base repulsion (inverse square law)
                repulsion_strength = 0.5 / (dist * dist)
                direction = diff / dist
                
                forces[node1] += direction * repulsion_strength
                forces[node2] -= direction * repulsion_strength
        
        # Calculate edge-based forces (attraction or repulsion based on rating)
        for u, v, data in G.edges(data=True):
            diff = positions[v] - positions[u]
            dist = np.linalg.norm(diff)
            if dist < 0.01:
                dist = 0.01
                diff = np.random.rand(2) - 0.5
            
            force_weight = data.get('force', 1.0)
            direction = diff / dist
            
            if force_weight > 0:
                # Attraction: pull nodes together (stronger for higher ratings)
                # Use spring-like force: stronger when far apart
                ideal_distance = 1.5 / (force_weight + 0.5)  # Closer for stronger relations
                displacement = dist - ideal_distance
                attraction = displacement * force_weight * 0.1
                
                forces[u] += direction * attraction
                forces[v] -= direction * attraction
            else:
                # Repulsion (X rating): push nodes apart
                repulsion = abs(force_weight) / (dist + 0.5)
                forces[u] -= direction * repulsion
                forces[v] += direction * repulsion
        
        # Apply forces with temperature-based damping
        for node in nodes:
            force_magnitude = np.linalg.norm(forces[node])
            if force_magnitude > 0:
                # Limit movement by temperature
                capped_force = forces[node] / force_magnitude * min(force_magnitude, temperature)
                positions[node] += capped_force
        
        # Cool down
        temperature *= cooling_rate
        
        # Enforce minimum distance between nodes for clarity
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                diff = positions[node1] - positions[node2]
                dist = np.linalg.norm(diff)
                if dist < min_distance:
                    if dist < 0.01:
                        diff = np.random.rand(2) - 0.5
                        dist = np.linalg.norm(diff)
                    direction = diff / dist
                    overlap = (min_distance - dist) / 2
                    positions[node1] += direction * overlap
                    positions[node2] -= direction * overlap
    
    # Center the layout
    center = np.mean([positions[n] for n in nodes], axis=0)
    for node in nodes:
        positions[node] -= center
    
    # Scale to reasonable size
    max_coord = max(np.max(np.abs(positions[n])) for n in nodes)
    if max_coord > 0:
        scale = 2.5 / max_coord
        for node in nodes:
            positions[node] *= scale
    
    return positions


def draw_curved_edge(ax, pos1, pos2, color, width, style='solid', alpha=0.8, curvature=0.2):
    """
    Draw a curved edge between two positions using a quadratic Bezier curve.
    
    Args:
        ax: matplotlib axes
        pos1, pos2: (x, y) tuples for start and end positions
        color: edge color
        width: line width
        style: 'solid' or 'dashed'
        alpha: transparency
        curvature: how much the edge curves (positive = curve one way, negative = other way)
    """
    # Calculate midpoint
    mid_x = (pos1[0] + pos2[0]) / 2
    mid_y = (pos1[1] + pos2[1]) / 2
    
    # Calculate perpendicular direction for the control point
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    length = np.sqrt(dx*dx + dy*dy)
    
    if length < 0.01:
        return  # Skip very short edges
    
    # Perpendicular unit vector
    perp_x = -dy / length
    perp_y = dx / length
    
    # Control point offset (perpendicular to the edge)
    offset = curvature * length
    ctrl_x = mid_x + perp_x * offset
    ctrl_y = mid_y + perp_y * offset
    
    # Create quadratic Bezier curve path
    verts = [
        (pos1[0], pos1[1]),  # Start
        (ctrl_x, ctrl_y),    # Control point
        (pos2[0], pos2[1])   # End
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    
    # Draw the curved edge
    linestyle = '--' if style == 'dashed' else '-'
    patch = mpl_patches.PathPatch(
        path,
        facecolor='none',
        edgecolor=color,
        linewidth=width,
        linestyle=linestyle,
        alpha=alpha,
        capstyle='round'
    )
    ax.add_patch(patch)


def calculate_edge_curvatures(G: nx.Graph, pos: dict) -> dict:
    """
    Calculate curvature for each edge to avoid parallel overlaps.
    
    Strategy:
    1. Group edges by their approximate angle
    2. Edges with similar angles get alternating curvatures
    3. Stronger relationships get less curvature (more direct)
    """
    edges = list(G.edges(data=True))
    curvatures = {}
    
    # Calculate angle for each edge
    edge_angles = {}
    for u, v, data in edges:
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        angle = np.arctan2(dy, dx)
        # Normalize angle to [0, pi) since edge direction doesn't matter
        if angle < 0:
            angle += np.pi
        if angle >= np.pi:
            angle -= np.pi
        edge_angles[(u, v)] = angle
    
    # Sort edges by angle to find near-parallel edges
    sorted_edges = sorted(edge_angles.items(), key=lambda x: x[1])
    
    # Base curvature values based on rating (stronger = less curved)
    rating_base_curvature = {
        'A': 0.15,
        'E': 0.20,
        'I': 0.25,
        'O': 0.30,
        'U': 0.35,
        'X': 0.25
    }
    
    # Assign curvatures with alternating directions
    angle_groups = []  # List of (angle, edge) groups
    angle_threshold = 0.3  # ~17 degrees - edges within this are considered "parallel-ish"
    
    for edge, angle in sorted_edges:
        # Find if this edge belongs to an existing angle group
        found_group = False
        for group in angle_groups:
            group_angle = group[0]
            if abs(angle - group_angle) < angle_threshold:
                group[1].append(edge)
                found_group = True
                break
        
        if not found_group:
            angle_groups.append([angle, [edge]])
    
    # Assign curvatures within each group
    for group_angle, group_edges in angle_groups:
        for i, edge in enumerate(group_edges):
            u, v = edge
            data = G.edges[u, v]
            rating = data.get('rating', 'O')
            
            # Base curvature from rating
            base = rating_base_curvature.get(rating, 0.25)
            
            # Add variation within the group
            # Alternate direction and add small offset for each edge in group
            direction = 1 if i % 2 == 0 else -1
            variation = (i // 2) * 0.08  # Increase curvature for more edges in same direction
            
            curvature = direction * (base + variation)
            curvatures[edge] = curvature
    
    return curvatures


def generate_relationship_diagram(G: nx.Graph, output_file: str, data: dict, include_u: bool = True, pos: dict = None, curvatures: dict = None):
    """Generate the relationship diagram PNG.
    
    Args:
        G: NetworkX graph with relationships
        output_file: Path to save PNG
        data: Original data dict
        include_u: Whether to include U (Unimportant) lines
        pos: Pre-computed positions (optional, will compute if None)
        curvatures: Pre-computed curvatures (optional, will compute if None)
    
    Returns:
        tuple: (pos, curvatures) for reuse in subsequent calls
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    num_nodes = G.number_of_nodes()
    
    # Use custom force-directed layout with attraction/repulsion
    if pos is None:
        print("Calculating force-directed layout (this may take a moment)...")
        pos = force_directed_layout(G, iterations=500, seed=42)
    
    # Calculate curvatures to avoid parallel edge overlaps
    if curvatures is None:
        curvatures = calculate_edge_curvatures(G, pos)
    
    # Draw edges by rating (draw U first so important ones are on top)
    rating_order = ['U', 'O', 'I', 'E', 'A', 'X']  # Draw in this order
    
    # Filter out U if not included
    if not include_u:
        rating_order = [r for r in rating_order if r != 'U']
    
    for rating in rating_order:
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('rating') == rating]
        if edges:
            style_config = RATING_STYLES.get(rating, RATING_STYLES['O'])
            edge_style = style_config.get('style', 'solid')
            
            for u, v, d in edges:
                pos1 = pos[u]
                pos2 = pos[v]
                curvature = curvatures.get((u, v), curvatures.get((v, u), 0.2))
                
                draw_curved_edge(
                    ax, pos1, pos2,
                    color=style_config['color'],
                    width=style_config['width'],
                    style=edge_style,
                    alpha=0.8,
                    curvature=curvature
                )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='white',
        node_size=1500,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Draw node labels (numbers)
    labels = {i: str(i + 1) for i in range(num_nodes)}  # 1-indexed for display
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=16,
        font_weight='bold',
        ax=ax
    )
    
    # Create legend (only include ratings that are shown)
    legend_patches = []
    legend_ratings = ['A', 'E', 'I', 'O', 'U', 'X'] if include_u else ['A', 'E', 'I', 'O', 'X']
    for rating in legend_ratings:
        style = RATING_STYLES[rating]
        linestyle = style.get('style', 'solid')
        patch = mpatches.Patch(
            color=style['color'],
            label=style['label'],
            linestyle=linestyle
        )
        legend_patches.append(patch)
    
    # Determine best legend position to avoid overlapping nodes
    # Check which corner has the fewest/most distant nodes
    corners = {
        'upper left': (-1, 1),
        'upper right': (1, 1),
        'lower left': (-1, -1),
        'lower right': (1, -1)
    }
    
    # Get normalized positions to compare with corners
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_range = max(all_x) - min(all_x) if max(all_x) != min(all_x) else 1
    y_range = max(all_y) - min(all_y) if max(all_y) != min(all_y) else 1
    x_center = (max(all_x) + min(all_x)) / 2
    y_center = (max(all_y) + min(all_y)) / 2
    
    # Calculate minimum distance from each corner to any node
    corner_distances = {}
    for corner_name, (cx, cy) in corners.items():
        # Scale corner to match data range
        corner_x = x_center + cx * x_range / 2
        corner_y = y_center + cy * y_range / 2
        
        # Find minimum distance to any node
        min_dist = float('inf')
        for node_pos in pos.values():
            dist = np.sqrt((node_pos[0] - corner_x)**2 + (node_pos[1] - corner_y)**2)
            min_dist = min(min_dist, dist)
        corner_distances[corner_name] = min_dist
    
    # Choose corner with maximum distance to nearest node
    best_corner = max(corner_distances, key=corner_distances.get)
    
    ax.legend(
        handles=legend_patches,
        loc=best_corner,
        fontsize=11,
        framealpha=0.9,
        title='Relationship Rating',
        title_fontsize=12
    )
    
    title_suffix = '' if include_u else ' (without U)'
    ax.set_title(f'Machine Proximity Relationship Diagram{title_suffix}', fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Relationship diagram saved to: {output_file}")
    
    return pos, curvatures


def generate_component_legend(data: dict, output_file: str):
    """Generate the component legend PNG showing which number represents which component."""
    components = data['components']
    num_components = len(components)
    
    # Calculate figure height based on number of components
    fig_height = max(4, num_components * 0.6 + 2)
    fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, num_components + 2)
    
    # Title
    ax.text(5, num_components + 1.2, 'Component Reference Legend', 
            fontsize=18, fontweight='bold', ha='center', va='center')
    
    # Header
    ax.text(1.5, num_components + 0.3, 'Number', fontsize=14, fontweight='bold', ha='center')
    ax.text(6, num_components + 0.3, 'Component Name', fontsize=14, fontweight='bold', ha='left')
    
    # Draw header line
    ax.axhline(y=num_components, color='black', linewidth=1.5, xmin=0.05, xmax=0.95)
    
    # Draw each component entry
    for i, component in enumerate(components):
        y_pos = num_components - 1 - i
        
        # Draw number circle
        circle = plt.Circle((1.5, y_pos + 0.5), 0.35, fill=True, 
                           facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(1.5, y_pos + 0.5, str(i + 1), fontsize=14, fontweight='bold', 
                ha='center', va='center')
        
        # Draw component name
        ax.text(6, y_pos + 0.5, component, fontsize=13, ha='left', va='center')
        
        # Draw separator line
        if i < num_components - 1:
            ax.axhline(y=y_pos, color='lightgray', linewidth=0.5, xmin=0.05, xmax=0.95)
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Component legend saved to: {output_file}")


def main():
    """Main function to run the visualization."""
    # Default file paths
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = Path(__file__).parent / 'relations.json'
    
    # Output file names
    output_dir = Path(json_file).parent
    diagram_with_u = output_dir / 'relationship_diagram_with_U.png'
    diagram_without_u = output_dir / 'relationship_diagram_without_U.png'
    legend_output = output_dir / 'component_legend.png'
    
    print(f"Loading relations from: {json_file}")
    
    # Load data
    data = load_relations(json_file)
    
    num_components = len(data['components'])
    num_explicit = len(data['relations'])
    num_total_pairs = num_components * (num_components - 1) // 2
    num_auto_u = num_total_pairs - num_explicit
    
    print(f"Found {num_components} components")
    print(f"  - {num_explicit} explicit relations in JSON")
    print(f"  - {num_auto_u} auto-generated U (Unimportant) relations")
    
    # Create graph
    G = create_relationship_graph(data)
    
    # Generate diagram WITH U lines (compute layout once)
    pos, curvatures = generate_relationship_diagram(G, str(diagram_with_u), data, include_u=True)
    
    # Generate diagram WITHOUT U lines (reuse layout)
    generate_relationship_diagram(G, str(diagram_without_u), data, include_u=False, pos=pos, curvatures=curvatures)
    
    # Generate component legend
    generate_component_legend(data, str(legend_output))
    
    print("\nVisualization complete!")
    print(f"  - Relationship diagram (with U):    {diagram_with_u}")
    print(f"  - Relationship diagram (without U): {diagram_without_u}")
    print(f"  - Component legend: {legend_output}")


if __name__ == '__main__':
    main()
