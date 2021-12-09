# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
conway.py 
A simple Python/matplotlib implementation of Conway's Game of Life.
Author: Mahesh Venkitachalam

"""
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import pandas as pn
import numpy as np
from sklearn.neighbors import NearestNeighbors



#Problem parameter
NB_QUEENS = 5

#recorrido=[[0,7,9,8,15],[7,0,10,4,11],[9,10,0,15,5],[8,4,15,0,17],[15,11,5,17,0]]
#print(recorrido)

A=pn.read_csv('agviajero.txt',skiprows=0,usecols=[0,1,2,3,4])
recorrido=np.array(A)
print(recorrido)
def evaluaPosicion(individual):
    longitud=len(individual)
    suma=0
    for i in range(1,longitud):
        suma=suma+recorrido[i-1][i]
    suma=suma+recorrido[individual[i]][individual[0]]
    return suma,

def evalNQueens(individual):
    """Evaluation function for the n-queens problem.
    The problem is to determine a configuration of n queens
    on a nxn chessboard such that no queen can be taken by
    one another. In this version, each queens is assigned
    to one column, and only one queen can be on each line.
    The evaluation function therefore only counts the number
    of conflicts along the diagonals.
    """
    size = len(individual)
    #Count the number of conflicts with other queens.
    #The conflicts can only be diagonal, count on each diagonal line
    left_diagonal = [0] * (2*size-1)
    right_diagonal = [0] * (2*size-1)

    #Sum the number of queens on each diagonal:
    for i in range(size):
        left_diagonal[i+individual[i]] += 1
        right_diagonal[size-1-i+individual[i]] += 1

    #Count the number of conflicts on each diagonal
    sum_ = 0
    for i in range(2*size-1):
        if left_diagonal[i] > 1:
            sum_ += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            sum_ += right_diagonal[i] - 1
    return sum_,


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Since there is only one queen per line, 
#individual are represented by a permutation
toolbox = base.Toolbox()
toolbox.register("permutation", random.sample, range(NB_QUEENS), NB_QUEENS)

#Structure initializers
#An individual is a list that represents the position of each queen.
#Only the line is stored, the column is the index of the number in the list.
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluaPosicion)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0/NB_QUEENS)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(seed=0):
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                        halloffame=hof, verbose=True)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof=main()    
    recorrido2=numpy.asarray(recorrido)
    print(recorrido[2][3])