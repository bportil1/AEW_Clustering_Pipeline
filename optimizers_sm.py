import numpy as np
import scipy.sparse as sp
from bsp import *

class AdamOptimizer:
    def __init__(self, similarity_matrix, gamma, update_sim_matr, objective_function, gradient_function, num_iterations=100, lambda_v=.99, lambda_s=.9999, epsilon=1e-8, alpha=10):
        '''
        Adaptive momentum optimizer
        '''
        self.gamma = gamma
        self.similarity_matrix = similarity_matrix
        self.generate_edge_weights = update_sim_matr
        self.objective_function = objective_function
        self.gradient_function = gradient_function
        self.num_iterations = num_iterations
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.epsilon = epsilon
        self.alpha = alpha

    def optimize(self):
        '''
        Optimization function
        '''
        print("Beginning Optimizations")
        v_curr = np.zeros_like(self.gamma)
        s_curr = np.zeros_like(self.gamma)
        curr_sim_matr = self.similarity_matrix
        curr_gamma = self.gamma
        step = 0
        min_error = float("inf")
        
        for i in range(self.num_iterations):
            print(f"Current Iteration: {i+1}")
            print("Computing Gradient")
            gradient = self.gradient_function(curr_sim_matr, curr_gamma)
            print("Current Gradient: ", gradient)
            
            print("Computing Error")
            curr_error = self.objective_function(curr_sim_matr, curr_gamma)
            print("Current Error: ", curr_error)

            v_next = (self.lambda_v * v_curr) + (1 - self.lambda_v) * gradient
            s_next = (self.lambda_s * s_curr) + (1 - self.lambda_s) * (gradient ** 2)
            step += 1
            corrected_v = v_next / (1 - self.lambda_v ** step)
            corrected_s = s_next / (1 - self.lambda_s ** step)

            print("Current Gamma: ", curr_gamma)
            curr_gamma = curr_gamma - (self.alpha * (corrected_v)) / (self.epsilon + np.sqrt(corrected_s))
            v_curr = v_next
            s_curr = s_next

            print("Next Gamma: ", curr_gamma)

            if curr_error <= min_error:
                min_error = curr_error
                min_gamma = curr_gamma

            curr_sim_matr = self.generate_edge_weights(curr_gamma)
        return curr_gamma

class SimulatedAnnealingOptimizer:
    def __init__(self, similarity_matrix, gamma, update_sim_matr, objective_function, num_iterations, temperature=10, min_temp=.001, cooling_rate=.9):
        '''
        Simulated Annealing Optimizer
        '''
        self.gamma = gamma
        self.similarity_matrix = similarity_matrix
        self.objective_function = objective_function
        self.generate_edge_weights = update_sim_matr
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate

    def optimize(self):
        '''
        Optimization function
        '''
        print("Beggining Simulated Annealing Optimization")
        update_ctr = 0
        curr_gamma = self.gamma
        curr_energy = self.objective_function(self.similarity_matrix, self.gamma)
        curr_sim_matr = self.similarity_matrix
        
        for idx in range(self.num_iterations):
            new_position = self.solution_transition(curr_gamma)
            curr_adj_matr = self.generate_edge_weights(new_position)
            new_energy = self.objective_function(curr_adj_matr, new_position)
            
            print("Potential New Position: ", new_position)
            print("Potential New Position Error: ", new_energy)

            alpha = self.acceptance_probability_computation(curr_energy, new_energy)
            print("Potential New Position Acceptance Probability: ", alpha)

            if new_energy < curr_energy :
                curr_gamma = new_position
                curr_energy = new_energy
                update_ctr = 0
            elif np.random.rand() > (1-alpha):
                curr_gamma = new_position
                curr_energy = new_energy
                update_ctr = 0
            else: 
                update_ctr += 1

            self.temperature *= self.cooling_rate

            print("Current Gamma: ", curr_gamma)
            print("Current Error: ", curr_energy)
            print("Current Temperature: ", self.temperature)

            if self.temperature < self.min_temp or update_ctr > 20:
                break

        print("Final Error: ", curr_energy)
        print("Final Gamma: ", curr_gamma)
        return curr_gamma

    def solution_transition(self, curr_gamma):
        '''
        Compute new possible position
        '''
        new_position = curr_gamma + np.random.normal(0, self.temperature, size=len(curr_gamma))
        return new_position

    def acceptance_probability_computation(self, curr_energy, new_energy):
        '''
        Compute probability that new position will be accepted
        '''
        if new_energy < curr_energy:
            return 1.0
        else:
            return np.exp(-((new_energy-curr_energy) / self.temperature))

class ParticleSwarmOptimizer:
    def __init__(self, similarity_matrix, gamma, objective_function, update_sim_matr, num_particles, dimensions, max_iter, w=0.5, c1=1.5, c2=1.5):
        '''
        Particle Swarm Optimization function
        '''
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        self.objective_function = objective_function
        self.generate_edge_weights = update_sim_matr
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.positions = np.random.uniform(-100, 100, (num_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([self.objective_function(self.similarity_matrix, p) for p in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

    def update_velocity(self, particle_idx):
        '''
        Update particles movement rates
        '''
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)
        cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[particle_idx])
        new_velocity = self.w * self.velocities[particle_idx] + cognitive_velocity + social_velocity
        return new_velocity

    def update_position(self, particle_idx):
        '''
        Update particle's position
        '''
        new_position = self.positions[particle_idx] + self.velocities[particle_idx]
        return new_position

    def optimize(self):
        '''
        Optimization function
        '''
        curr_adj_matr = self.similarity_matrix
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                new_velocity = self.update_velocity(i)
                self.velocities[i] = new_velocity
                new_position = self.update_position(i)
                self.positions[i] = new_position
                
                print("Current Position for Agent ", i, ":", new_position)
                curr_adj_matr = self.generate_edge_weights(new_position)
                fitness = self.objective_function(curr_adj_matr, self.positions[i])
                
                print("Current Fitness for Agent ", i, ":", fitness)
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

            min_fitness_idx = np.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[min_fitness_idx] < self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[min_fitness_idx]
                self.global_best_position = self.personal_best_positions[min_fitness_idx]
                print(f"Iteration {iteration}/{self.max_iter}, Best Fitness: {self.global_best_fitness}")

        return self.global_best_position

class SwarmBasedAnnealingOptimizer:
    def __init__(self, similarity_matrix, gamma, objective_function, gradient_function, update_sim_matr, num_particles, dimensions, max_iter, h=0.95):
        '''
        Swarm Based Annealing Optimization
        '''       
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        self.objective_function = objective_function
        self.generate_edge_weights = update_sim_matr
        self.gradient_function = gradient_function
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.h = h
        self.provisional_minimum = float('inf')
        self.positions = np.random.uniform(-100, 100, (num_particles, len(self.gamma)))
        self.masses = np.ones((1, num_particles))[0] * (1/num_particles)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([self.objective_function(self.similarity_matrix, p) for p in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

    def update_mass(self, particle_idx):
        '''
        Update particle mass
        '''
        new_mass = self.masses[particle_idx] - ((self.masses[particle_idx] * self.h) * (self.personal_best_fitness[particle_idx] - self.provisional_minimum))
        return new_mass

    def update_position(self, particle_idx, eta, curr_adj_matrix):
        '''
        Update particle position
        returning nans maybe values too big, props if fitness gets rdiculous big
        '''
        gradient = self.gradient_function(curr_adj_matrix, self.positions[particle_idx])
        new_position = self.positions[particle_idx] - (self.h * gradient * self.personal_best_fitness[particle_idx]) + (np.sqrt(2 * self.h * self.masses[particle_idx]) * eta)
        return new_position

    def provisional_min_computation(self):
        '''
        Compute provisional minimum
        '''
        return np.sum([self.masses[y] * self.personal_best_fitness[y] for y in range(self.num_particles)]) / np.sum(self.masses)

    def optimize(self):
        '''
        Optimization function
        '''
        print("Beginning Optimization")
        curr_adj_matr = self.similarity_matrix
        self.provisional_minimum = self.provisional_min_computation()
        print("Provisional Minimum: ", self.provisional_minimum)
        
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                new_mass = self.update_mass(i)
                self.masses[i] = new_mass

            eta = np.random.normal(0, 1, size=len(self.gamma))
            for i in range(self.num_particles):
                curr_adj_matr = self.generate_edge_weights(self.positions[i])
                print("Initial Position for Agent ", i, ":", self.positions[i])
                new_position = self.update_position(i, eta, curr_adj_matr)
                self.positions[i] = new_position
                print("Current Position for Agent ", i, ":", new_position)
                fitness = self.objective_function(curr_adj_matr, self.positions[i])
                print("Current Fitness for Agent ", i, ":", fitness)
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                
            self.provisional_minimum = self.provisional_min_computation()
            print("Provisional Minimum: ", self.provisional_minimum)

        print("Completed Optimization")
        return self.global_best_position

class HdFireflySimulatedAnnealingOptimizer:
    def __init__(self, similarity_matrix, spread_gamma, update_sim_matr, objective_function, dimensions, pop_test=100, hdfa_iterations=5, gamma=1, alpha=.2): 
        self.similarity_matrix = similarity_matrix
        self.spread_gamma = spread_gamma
        self.objective_computation = objective_function
        self.generate_edge_weights = update_sim_matr

        self.pop_test = pop_test 
        self.dimensions = dimensions
        self.hdfa_iterations = hdfa_iterations
        self.alpha = alpha
        self.gamma = gamma

        self.pop_positions = self.initialize_positions('initial')
        self.pop_attractiveness = np.ones(self.pop_test)
        self.pop_fitness = np.zeros(self.pop_test)
    
        self.pop_alpha = np.zeros(self.pop_test)

        self.initialize_fitness()

        self.bsp_tree = self.initialize_bsp()
    
    def initialize_bsp(self):
        bsp = BSP(self.dimensions)
        bsp.build_tree(self.pop_positions, self.pop_fitness)
        return bsp

    def initialize_fitness(self):
        for idx in range(self.pop_test):
            curr_sim_matr = self.generate_edge_weights(self.pop_positions[idx])
            self.pop_fitness[idx] = self.objective_computation(curr_sim_matr, self.pop_positions[idx])

    def initialize_positions(self, stage):
        if stage == 'initial':
            return np.random.rand(self.pop_test, self.dimensions)
        elif stage == 'finder_tracker':
            return self.finder_tracker_assignments()    
    
    def l2_norm(self, ff_idx_1, ff_idx_2):
        return np.sqrt(np.sum((self.pop_positions[ff_idx_1] - self.pop_positions[ff_idx_2])**2))

    def compute_attractiveness(self, ff_idx_1, ff_idx_2):
        norm = self.l2_norm(ff_idx_1, ff_idx_2)**2
        return self.pop_attractiveness[ff_idx_1] * np.exp(-self.gamma*norm)

    def update_position(self, new_attr, ff_idx_1, ff_idx_2):
        #attractiveness = self.compute_attractiveness(ff_idx_1, ff_idx_2)
        return self.pop_positions[ff_idx_1] + new_attr * (self.pop_positions[ff_idx_2] - self.pop_positions[ff_idx_1]) + self.alpha*(np.random.rand()-.5)

    #def update_fitness(self, curr_sim_matr, ff_idx_1):
    #    self.pop_fitness[ff_idx_1] = self.objective_computation(curr_sim_matr, self.pop_positions[ff_idx_1])
        
    def grow_bsp(self, points, fitness_scores):
        bsp = BSP(self.bsp_tree, self.dimensions)
        self.bsp_tree = bsp.grow_tree(points, fitness_scores)

    def sort_ff_data(self):
        indices = np.argsort(self.pop_fitness)
        self.pop_fitness = self.pop_fitness[indices]
        self.pop_positions = self.pop_positions[indices][:]
        self.pop_attractiveness = self.pop_attractiveness[indices]
        self.pop_alpha = self.pop_alpha[indices]

    def finder_tracker_assignments(self, tol=.5):
        print("Reassigning Finder Trackers")
        for idx1 in range(self.pop_test):
            for idx2 in range(self.pop_test):
                if idx1 != idx2:
                    same_region = self.bsp_tree.same_region_check(self.pop_positions[idx1], self.pop_positions[idx2], tol)
                    if same_region:
                        dist = np.linalg.norm(self.pop_positions[idx1] - self.pop_positions[idx2])
                        if dist < tol:
                            self.sort_ff_data()
                            top_forty_percent = int(np.ceil(self.pop_test*.4))
                            bottom_sixty_percent = int(np.floor(self.pop_test*.6))
                            best_forty_positions = self.pop_positions[:top_forty_percent]
                            best_forty_fitness = self.pop_fitness[:top_forty_percent]
                            best_forty_attractiveness = self.pop_attractiveness[:top_forty_percent] 
                            best_forty_alpha = self.pop_alpha[:top_forty_percent]
                            bottom_sixty_positions = np.random.rand(np.floor(bottom_sixty_percent), self.dimensions)
                            bottom_sixty_attractiveness = np.ones(bottom_sixty_percent)
                            bottom_sixty_fitness = np.zeros(bottom_sixty_percent)
                            bottom_sixty_alpha = np.zeros(bottom_sixty_percent)                        
    
                            for idx in range(bottom_sixty_percent):

                                curr_sim_matr = self.generate_edge_weights(bottom_sixty_positions[idx])

                                self.pop_fitness[idx] = self.objective_computation(curr_sim_matr, bottom_sixty_positions[idx])
                            self.pop_positions = np.concatenate((best_forty_positions, bottom_sixty_positions))
                            self.pop_fitness = np.concatenate((best_forty_fitness, bottom_sixty_fitness)).ravel()
                            self.pop_attractiveness = np.concatenate((best_forty_attractiveness, bottom_sixty_attractiveness)).ravel()
                            self.pop_alpha = np.concatenate((best_forty_alpha, bottom_sixty_alpha))
                            print("Completed Finder/Tracker Reassignments")
                            return 0
        print("No Reassignments Needed")

    def optimize(self):
        print("Beggining Hd-Firefly-SA Optimization")
        last_alpha = float('inf')
        maturity_condition = True
        nonincreasing_alpha_counter = 0
        hdfa_ctr = 0
        min_reg_fitness = float('inf')
        new_fitness = float('inf')
        curr_sim_matr = self.similarity_matrix
        while hdfa_ctr < self.hdfa_iterations:
            print("Current HdFa Iteration: ", hdfa_ctr)
            for idx1 in range(self.pop_test):
                for idx2 in range(self.pop_test):
                    if self.pop_fitness[idx1] < self.pop_fitness[idx2]:
                        new_attr = self.compute_attractiveness(idx1, idx2)
                        new_position = self.update_position(new_attr, idx1, idx2)
                        curr_sim_matr = self.generate_edge_weights(new_position)
                        in_min_region, min_region, min_reg_fitness, new_graph, min_node, region_points = find_region_with_lowest_fitness(curr_sim_matr, self.objective_computation, self.bsp_tree, new_attr, new_position, self.dimensions)
                        new_fitness = self.objective_computation(curr_sim_matr, min_region)
                        if new_fitness < self.pop_fitness[idx1]:
                            self.pop_fitness[idx1] = new_fitness
                            self.pop_positions[idx1] = new_position
                        print("Potential New Position: ", new_position)
                        print("Potential New Fitness: ", new_fitness)
                
                if maturity_condition:
                    #if min_reg_fitness == float('inf'):
                    #   min_reg_fitness = self.pop_fitness[idx1]
                    #if new_fitness == float('inf'):
                    #   new_fitness = 0
                    if min_reg_fitness == float('inf') or new_fitness == float('inf'):
                        self.pop_alpha[idx1] = 1
                        #print("min reg_fitness :", min_reg_fitness)
                        #print("new_fitness: ", new_fitness)
                    else:
                        self.pop_alpha[idx1] = np.abs(min_reg_fitness - new_fitness)
                    #print("New pop alpha agent ", idx1,  " ", self.pop_alpha[idx1])
                    if idx1 > 1:
                        alpha_avg = np.average(self.pop_alpha[:idx1])
                        #print("New Alpha AVG: ", alpha_avg)
                    else:
                        alpha_avg = 1
                    print("Current Alpha Average: ", alpha_avg)
                    if alpha_avg <= 0 or nonincreasing_alpha_counter >= 1000:
                        break
                    if last_alpha > alpha_avg:  
                        nonincreasing_alpha_counter += 1
                    else:
                        nonincreasing_alpha_counter = 0

                    last_alpha = alpha_avg
                    print("Current NonIncreasing Alpha Counter: ", nonincreasin_alpha_counter)
            self.finder_tracker_assignments()
            hdfa_ctr += 1
        #print(in_min_region, " ", min_region, " ", min_reg_fitness)

        _, min_position, lowest_fitness = self.bsp_tree.find_lowest_fitness_region()

        print("Final Hd-FF Min Position: ", min_position[0])
        print("Final Hd-FF Error: ", lowest_fitness)

        sa = SimulatedAnnealingOptimizer(self.similarity_matrix,  min_position[0], self.generate_edge_weights, self.objective_computation, temperature=10, cooling_rate = .90)
        
        min_pt, min_fitness, path = sa.optimize()

        print("Final SA Min Position: ", min_pt)
        print("final SA Error: ", min_fitness)

        return min_pt, min_fitness, path

