import numpy as np
import copy

import textDisplay

from pacman import *
from searchAgents import GAAgent

class EvolvePacManBT():
    def __init__(self, args, pop_size, num_parents, numGames=5):
        args['numGames'] = numGames
        # args['numTraining'] = args['numGames'] ## DOESN'T WORK # suppress the output
        self.display_graphics = args['display']
        args['display'] = textDisplay.NullGraphics()

        self.args = args

        self.pop_size = pop_size
        self.num_parents = num_parents
        self.gene_pool = None

        self.__create_initial_pop()

    def __create_initial_pop(self):
        self.gene_pool = [GAAgent()]
        self.produce_next_generation(self.gene_pool)

    def produce_next_generation(self, parents):
        """ YOUR CODE HERE!"""
        new_generation = []
        for i in range(self.pop_size):
            new_gene = parents[np.random.randint(len(parents))].copy()
            new_gene.mutate()
            new_generation.append(new_gene)
        self.gene_pool = new_generation




    def evaluate_population(self):
        """ Evaluate the fitness, and sort the population accordingly."""
        """ YOUR CODE HERE!"""
        fit = []
        for i in range(self.pop_size):
            agents = self.gene_pool[i]
            self.args['pacman'] = agents
            games = runGames(**args)
            fitness_score = [o.state.getScore() for o in games]
            fit.append(np.mean(fitness_score))
        print( "fitness pop ", fit)
        return fit


    def select_parents(self, num_parents, fit):
        """ YOUR CODE HERE!"""
        sorted_vals = [x for _, x in sorted(zip(fit, self.gene_pool))]
        return sorted_vals[-num_parents:]



    def run(self, num_generations=10):
        display_args = copy.deepcopy(self.args)
        display_args['display'] = self.display_graphics
        display_args['numGames'] = 1

        for i in range(num_generations):
            fitness = self.evaluate_population()
            parents = self.select_parents(self.num_parents, fitness)
            self.gene_pool = parents
            self.produce_next_generation(parents)


            # TODO: Print some summaries
            if i % 10 == 0 and i>0:
                print("############################################################")
                print("############################################################")
                print("############################################################")
                print('i', i, fitness)
                display_args['pacman'] = self.gene_pool[0]
                print('best genome!')
                self.gene_pool[0].print_genome()
                runGames(**display_args)
                print("############################################################")
                print("############################################################")
                print("############################################################")

        print('best genome!')
        self.gene_pool[0].print_genome()
        runGames(**display_args)

if __name__ == '__main__':
    args = readCommand( sys.argv[1:] ) # Get game components based on input

    pop_size = 16
    num_parents = int(pop_size/4)+1
    numGames = 3
    num_generations = 2

    GA = EvolvePacManBT(args, pop_size, num_parents, numGames)
    GA.run(num_generations)


