import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BSP:
    def __init__(self, dimensions, tree=None):
        self.dim = dimensions
        self.tree = tree

    def bisect(self, points, plane, fitness_scores):
        """
        Bisects a set of points using a hyperplane defined by `plane`.
        The plane is assumed to be an array that defines the dividing hyperplane.
        Fitness scores are also passed to update the nodes.
        """
        points_on_plane = []
        ahead = []
        behind = []
        fitness_ahead = []
        fitness_behind = []
        
        self.fitness_scores_updated = []

        for i, point in enumerate(points):
            normal = plane[1]
            point_on_plane = np.dot(point - plane[0], normal)
            if np.isclose(point_on_plane, 0):
                points_on_plane.append(point)
            elif point_on_plane > 0:
                ahead.append(point)
                fitness_ahead.append(fitness_scores[i])
                self.fitness_scores_updated.append(fitness_scores[i])
            else:
                behind.append(point)
                fitness_behind.append(fitness_scores[i])
                self.fitness_scores_updated.append(fitness_scores[i])
        
        return (np.array(ahead), np.array(fitness_ahead)), (np.array(behind), np.array(fitness_behind)), np.array(points_on_plane)

    def build_tree(self, points, fitness_scores):
        def bsp_helper(points, fitness_scores, plane):
            ahead, behind, on_plane = self.bisect(points, plane, fitness_scores)
            node_id = id(plane)
        
            min_fitness = float('inf')
            min_fitness_points = []

            #print(ahead)
            #print(behind)
            #print(on_plane)
            #print(self.fitness_scores_updated)

            for region_points, region_fitness in [(ahead[0], ahead[1]), (behind[0], behind[1])]:
                #print("Region Points: ", region_points)
                #print(region_points[1])
                #print("Regions Fitnesses: ", region_fitness)
                if len(region_fitness) == 0:
                    continue

                min_fitness_in_region = min(region_fitness)
                if min_fitness_in_region < min_fitness:
                    min_fitness = min_fitness_in_region
                    min_fitness_points = region_points[np.asarray(region_fitness)==min_fitness]
                    #print("Min Fitness Points: ", min_fitness_points)

                # Add the node to the graph
                graph.add_node(node_id, plane=plane, min_fitness=min_fitness, min_fitness_points=min_fitness_points)
            
                if len(behind[0]) > 0:
                    node_behind = bsp_helper(behind[0], behind[1], select_dividing_plane(behind[0], self.dim))
                    graph.add_edge(node_id, node_behind, position=-1)
            
                if len(ahead[0]) > 0:
                    node_ahead = bsp_helper(ahead[0], ahead[1], select_dividing_plane(ahead[0], self.dim))
                    graph.add_edge(node_id, node_ahead, position=1)
            
            return node_id
        
        graph = nx.DiGraph()
        if len(points) > 0:
            starting_plane = select_dividing_plane(points, self.dim)
        else:
            starting_plane = (np.zeros(self.dim), np.ones(self.dim))  # Default plane
        
        bsp_helper(points, fitness_scores, starting_plane)
        
        self.tree = graph

        min_fitness_points = []

            #print(ahead)
        return nx.relabel.convert_node_labels_to_integers(graph)

    def grow_tree(self, points, fitness_scores):
        def bsp_helper(points, fitness_scores, plane):
            ahead, behind, on_plane = self.bisect(points, plane, fitness_scores)
            node_id = id(plane)
        
            min_fitness = float('inf')
            min_fitness_points = []

            #print(ahead)
            #print(behind)
            #print(on_plane)
            #print(self.fitness_scores_updated)

            for region_points, region_fitness in [(ahead[0], ahead[1]), (behind[0], behind[1])]:
                #print("Region Points: ", region_points)
                #print(region_points[1])
                #print("Regions Fitnesses: ", region_fitness)
                if len(region_fitness) == 0:
                    continue

                min_fitness_in_region = min(region_fitness)
                if min_fitness_in_region < min_fitness:
                    min_fitness = min_fitness_in_region
                    min_fitness_points = region_points[np.asarray(region_fitness)==min_fitness]
                    #print("Min Fitness Points: ", min_fitness_points)

                # Add the node to the graph
                graph.add_node(node_id, plane=plane, min_fitness=min_fitness, min_fitness_points=min_fitness_points)
            
                if len(behind[0]) > 0:
                    node_behind = bsp_helper(behind[0], behind[1], select_dividing_plane(behind[0], self.dim))
                    graph.add_edge(node_id, node_behind, position=-1)
            
                if len(ahead[0]) > 0:
                    node_ahead = bsp_helper(ahead[0], ahead[1], select_dividing_plane(ahead[0], self.dim))
                    graph.add_edge(node_id, node_ahead, position=1)
            
            return node_id
        
        graph = self.tree
        if len(points) > 0:
            starting_plane = select_dividing_plane(points, self.dim)
        else:
            starting_plane = (np.zeros(self.dim), np.ones(self.dim))  # Default plane
        
        bsp_helper(points, fitness_scores, starting_plane)
        
        self.tree = graph

        return nx.relabel.convert_node_labels_to_integers(graph)

    def visualize_tree(self):
        print("in visualize tree")
        pos = nx.spring_layout(self.tree)
        #print(pos)
        node_labels = nx.get_node_attributes(self.tree, 'plane')
        #print(node_labels)
        plt.figure(figsize=(100, 100))
        #nx.draw(self.tree, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
        nx.draw(self.tree, pos, node_size=30, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

        #nx.draw_networkx_labels(self.tree, pos, labels=node_labels)
        plt.title("Current BSP")
        plt.show()
        plt.close()
        #print("at end  of plot")
    
    def add_point(self, point, fitness):
        """
        Add a new point with its fitness score to the appropriate place in the BSP tree.
        If necessary, bisect the region to accommodate the new point.
        """
        # Start from the root node of the tree
        current_node = list(self.tree.nodes)[0]  # Start from the root of the tree
        
        while current_node is not None:
            #print('Current Node: ', current_node)
            #print('Current Node Attributes: ', self.tree.nodes[current_node])
            
            # Get the plane defining the current node
            plane = self.tree.nodes[current_node]["plane"]
            min_fitness_points = self.tree.nodes[current_node]["min_fitness_points"]
            
            # Check if the point is in the region of the current node
            normal = plane[1]
            point_on_plane = np.dot(point - plane[0], normal)

            # If the point is on the plane, add it to this node's points
            if np.isclose(point_on_plane, 0):
                #print(f"Point {point} is on the plane of the node.")
                self.tree.nodes[current_node]["min_fitness_points"].append(point)
                self.tree.nodes[current_node]["fitness_scores"].append(fitness)
                break

            # If the point is ahead (positive side of the plane), move to the "ahead" node
            elif point_on_plane > 0:
                children = list(self.tree.successors(current_node))
                if len(children) == 1:
                    current_node = children[0]  # Move to the "ahead" child
                elif len(children) > 1:
                    current_node = children[1]  # Move to the "ahead" child (second child)
                else:
                    # No "ahead" child, create a new node and add the point here
                    new_plane = select_dividing_plane([point], self.dim)  # Select a new plane for this region
                    new_node_id = id(new_plane)
                    self.tree.add_node(new_node_id, plane=new_plane, min_fitness_points=[point], fitness_scores=[fitness])
                    self.tree.add_edge(current_node, new_node_id, position=1)  # Connect the parent to the new node
                    break

            # If the point is behind (negative side of the plane), move to the "behind" node
            elif point_on_plane < 0:
                children = list(self.tree.predecessors(current_node))
                if len(children) == 1:
                    current_node = children[0]  # Move to the "behind" child
                elif len(children) > 1:
                    current_node = children[1]  # Move to the "behind" child (second child)
                else:
                    # No "behind" child, create a new node and add the point here
                    new_plane = select_dividing_plane([point], self.dim)  # Select a new plane for this region
                    new_node_id = id(new_plane)
                    self.tree.add_node(new_node_id, plane=new_plane, min_fitness_points=[point], fitness_scores=[fitness])
                    self.tree.add_edge(current_node, new_node_id, position=-1)  # Connect the parent to the new node
                    break
        print("New point added successfully.")

    def find_lowest_fitness_region(self):
        """
        Traverses the BSP tree to find the region with the lowest fitness score.
        Returns the region points and their fitness scores with the minimum fitness.
        """
        lowest_fitness = float('inf')
        lowest_fitness_region = None
        lowest_fitness_points = None
        
        # Start from the root node
        nodes_to_visit = [list(self.tree.nodes)[0]]  # List with root node
        visited = set()

        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            
            if current_node in visited or self.tree.nodes[current_node] == {}:
                continue
            visited.add(current_node)
            
            #print("Final ATTRss: ", self.tree.nodes[current_node])

            # Get the node attributes: fitness points and the minimum fitness value
            min_fitness_points = self.tree.nodes[current_node]["min_fitness_points"]
            min_fitness = self.tree.nodes[current_node]["min_fitness"]
            
            # Update the lowest fitness if necessary
            if min_fitness < lowest_fitness:
                lowest_fitness = min_fitness
                lowest_fitness_region = current_node
                lowest_fitness_points = min_fitness_points

            # Add children (subtree) of the current node to the list for further traversal
            children = list(self.tree.successors(current_node)) + list(self.tree.predecessors(current_node))
            nodes_to_visit.extend(children)
        
        return lowest_fitness_region, lowest_fitness_points, lowest_fitness

    def same_region_check(self, point1, point2, tol):

        def traverse_bsp(point, current_node):

            while current_node is not None and current_node != {}:
                print("In same_region_check")
                #print("Current Node: " , self.tree.nodes[current_node])
                # Get the plane defining the current node
                plane = self.tree.nodes[current_node]["plane"]
                normal = plane[1]
            
                # Calculate the distance of the point from the plane
                point_on_plane = np.dot(point - plane[0], normal)
            
                if np.abs(point_on_plane) < tol:
                    # Point is within tau distance of the plane, treat it as on the plane
                    return current_node
                elif point_on_plane > 0:
                    # Point is ahead of the plane, move to the "ahead" child
                    children = list(self.tree.successors(current_node))
                    if len(children) > 0:
                        current_node = children[0]  # Move to the "ahead" child
                    else:
                        break  # No more children, return the current node
                else:
                    # Point is behind the plane, move to the "behind" child
                    children = list(self.tree.predecessors(current_node))
                    if len(children) > 0:
                        current_node = children[0]  # Move to the "behind" child
                    else:
                        break  # No more children, return the current node
            return current_node

        # Start from the root node
        root_node = list(self.tree.nodes)[0]  # Assuming the root is the first node

        # Traverse the tree for both points
        final_node1 = traverse_bsp(point1, root_node)
        final_node2 = traverse_bsp(point2, root_node)

        # If both points end up in the same node (within tau), they are in the same region
        return final_node1 == final_node2


def select_dividing_plane(points, dimensions):
    """
    Selects the dividing plane based on the median of the points.
    The plane is defined by the median point and the normal is aligned
    with the axis of greatest variance in the dataset.
    """
    variances = np.var(points, axis=0)
    dividing_axis = np.argmax(variances)  # Choose the axis with the greatest variance
    #sorted_points = points[points[:, dividing_axis].argsort()]
    #print("before sort: ", points)
    sorted_points = points[np.argsort(points[:, 0])]
    #print("after sort: ", sorted_points)    
    median_point = sorted_points[len(sorted_points) // 2]

    normal = np.zeros(dimensions)
    normal[dividing_axis] = 1  # The normal is along the chosen axis

    return median_point, normal

def find_region_with_lowest_fitness(curr_sim_matr, objective_computation, tree, point, fitness, dimensions):
    """
    Traverse the BSP tree to find the region where the point lies,
    check if it is in the lowest fitness region, and if so, bisect the region.
    Returns the node with the lowest fitness, its region (all points in its subtree), and all its child nodes (subtree).
    """
    current_node = list(tree.tree.nodes)[0]  # Start from the root of the tree
    lowest_fitness = float('inf')
    lowest_fitness_region = None
    all_subtree_points = []  # To store all points in the subtree
    node_with_lowest_fitness = None  # To store the node with the lowest fitness

    def dfs(node):
        """Recursive DFS to gather all points and nodes in the subtree rooted at 'node'."""
        # Get points of the current node
        min_fitness_points = tree.tree.nodes[node]["min_fitness_points"]
        all_subtree_points.extend(min_fitness_points)  # Add points to the region
        children = list(tree.tree.successors(node))
        for child in children:
            dfs(child)

    while current_node is not None:
        #print('Current Node: ', current_node)
        #print('Current Node Attributes: ', tree.tree.nodes[current_node])
        #print('Current Neighbors: ', list(tree.tree.neighbors(current_node)))
        
        # Get the plane defining the current node
        plane = tree.tree.nodes[current_node]["plane"]
        min_fitness = tree.tree.nodes[current_node]["min_fitness"]
        min_fitness_points = tree.tree.nodes[current_node]["min_fitness_points"]

        # Update the lowest fitness if needed
        if min_fitness < lowest_fitness:
            lowest_fitness = min_fitness
            lowest_fitness_region = min_fitness_points  # Points of the lowest fitness region
            node_with_lowest_fitness = current_node  # Store the node with the lowest fitness

        # Check if the point is in the lowest fitness region
        if point in lowest_fitness_region:
            print(f"Point {point} is in the lowest fitness region, bisecting the region.")
            # Bisect the region
            new_fitness_scores = objective_computation(curr_sim_matr, lowest_fitness_region[0]) 

            bsp = BSP(tree, dimensions)
            graph = bsp.grow_tree(lowest_fitness_region, new_fitness_scores)

            # Add DFS to gather the entire subtree points and nodes
            dfs(current_node)

            # Return the updated tree along with the node and its subtree
            return True, lowest_fitness_region, lowest_fitness, graph, node_with_lowest_fitness, all_subtree_points

        normal = plane[1]
        point_on_plane = np.dot(point - plane[0], normal)

        # Move to the appropriate subtree
        if point_on_plane > 0:
            neighbors = list(tree.tree.neighbors(current_node))
            if len(neighbors) == 1:
                current_node = neighbors[0]
            elif len(neighbors) > 1:
                current_node = neighbors[1]  # Move to the "ahead" node
            else:
                break  # No more nodes, end the search
        elif point_on_plane < 0:
            neighbors = list(tree.tree.neighbors(current_node))
            if neighbors:
                current_node = neighbors[0]  # Move to the "behind" node
            else:
                break  # No more nodes, end the search
        else:
            break  # Point is on the plane, we stop here

        if tree.tree.nodes[current_node] == {}:
            break

    # If we exit the loop, that means we traversed through the tree
    # and found the lowest fitness region and its subtree
    #print(f"Lowest fitness region: {lowest_fitness_region}")
    #print(f"Subtree points: {all_subtree_points}")

    # Return the node with the lowest fitness and all the points in its subtree
    return False, lowest_fitness_region, lowest_fitness, tree, node_with_lowest_fitness, all_subtree_points

