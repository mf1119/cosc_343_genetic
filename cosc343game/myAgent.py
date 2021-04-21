import json
from datetime import datetime
import random
from pathlib import Path

import numpy as np

playerName = "myAgent"

# Training Tuples
trainingSchedule = [("random", 300)]

###
# Traits
###
CHROMOSOME_LENGTH = 9


class Chromosome:
    # Wall
    _INTROVERSION = 0  # Affinity towards walls

    # Strawberries
    _GATHERER = 1  # Affinity towards berries
    _HUNGRY = 2  # Likelihood of eating berries. Sometimes, might have to run if enemies near.

    # Enemies
    _CONFIDENCE = 3  # Affinity towards enemies generally
    _AGGRESSION = 4  # Affinity towards weaker enemies
    _FEARFUL = 5  # Affinity towards stronger enemies

    # Friendlies
    _CLUMP = 6  # Affinity towards all friendlies
    _SHELTER = 7  # Affinity towards stronger friendlies
    _PROTECT = 8  # Affinity towards weaker friendlies


###
# Weights - Default Values
###
# Generation related vars
SURVIVAL_RATE = 0.7
SURVIVAL_OFFSET = 0.5
MUTATION_RATE = 0.8
MUTATION_DEGREE = 15
MUTATION_DELETERIOUS = 0.5

# Fitness weights       # Optimal values for the impatient
WEIGHT_ALIVE = 30       # 30
WEIGHT_TURN = 0.5       # 0.5
WEIGHT_SIZE = 20        # 20
WEIGHT_STRAWB = 60      # 60
WEIGHT_ENEMY = 50       # 50
WEIGHT_TRAVEL = 1       # 1
WEIGHT_BOUNCE = 0.5     # 0.5


class Weight:
    _SURVIVAL_RATE = 0
    _SURVIVAL_OFFSET = 1
    _MUTATION_RATE = 2
    _MUTATION_DEGREE = 3
    _MUTATION_DELETERIOUS = 4

    _WEIGHT_ALIVE = 5
    _WEIGHT_TURN = 6
    _WEIGHT_SIZE = 7
    _WEIGHT_STRAWB = 8
    _WEIGHT_ENEMY = 9
    _WEIGHT_TRAVEL = 10
    _WEIGHT_BOUNCE = 11


# Chromosome values are randomly assigned a value between -INIT and +INIT
INIT_RANGE = 100


class MyCreature:
    def __init__(self):
        self.chromosome = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for x in range(CHROMOSOME_LENGTH):
            self.chromosome[x] = random.randrange(-INIT_RANGE, INIT_RANGE)

    def AgentFunction(self, percepts):
        # The 'actions' variable must be returned:
        # 0 - move left
        # 1 - move up
        # 2 - move right
        # 3 - move down
        # 4 - eat

        actions = np.array([0, 0, 0, 0, 0])

        creature_map = percepts[:, :, 0]
        food_map = percepts[:, :, 1]
        wall_map = percepts[:, :, 2]

        # +ve = right and down
        xDir = 0
        yDir = 0
        eatAction = 0

        # creature
        for x in range(5):
            for y in range(5):
                if creature_map[x, y] < 0:
                    # If all enemies
                    xDir += ignore_divide_zero(1, (x - 2)) * self.chromosome[Chromosome._CONFIDENCE]
                    yDir += ignore_divide_zero(1, (y - 2)) * self.chromosome[Chromosome._CONFIDENCE]
                    if abs(creature_map[x, y]) > creature_map[2, 2]:
                        # If bigger enemies
                        xDir += ignore_divide_zero(1, (x - 2)) * self.chromosome[Chromosome._FEARFUL] * (
                                abs(creature_map[x, y]) - creature_map[2, 2])
                        yDir += ignore_divide_zero(1, (y - 2)) * self.chromosome[Chromosome._FEARFUL] * (
                                abs(creature_map[x, y]) - creature_map[2, 2])
                    elif abs(creature_map[x, y]) < creature_map[2, 2]:
                        # If smaller enemies
                        xDir += ignore_divide_zero(1, (x - 2)) * self.chromosome[Chromosome._AGGRESSION] * (
                                creature_map[2, 2] - abs(creature_map[x, y]))
                        yDir += ignore_divide_zero(1, (y - 2)) * self.chromosome[Chromosome._AGGRESSION] * (
                                creature_map[2, 2] - abs(creature_map[x, y]))
                if creature_map[x, y] > 0:
                    # If friendly
                    if not ((x == 2) and (y == 2)):
                        xDir += ignore_divide_zero(1, (x - 2)) * self.chromosome[Chromosome._CLUMP]
                        yDir += ignore_divide_zero(1, (y - 2)) * self.chromosome[Chromosome._CLUMP]

                    if creature_map[x, y] > creature_map[2, 2]:
                        # If bigger friends
                        xDir += ignore_divide_zero(1, (x - 2)) * self.chromosome[Chromosome._SHELTER] * (
                                creature_map[x, y] - creature_map[2, 2])
                        yDir += ignore_divide_zero(1, (y - 2)) * self.chromosome[Chromosome._SHELTER] * (
                                creature_map[x, y] - creature_map[2, 2])
                    elif creature_map[x, y] < creature_map[2, 2]:
                        # If smaller friends
                        xDir += ignore_divide_zero(1, (x - 2)) * self.chromosome[Chromosome._PROTECT] * (
                                creature_map[2, 2] - creature_map[x, y])
                        yDir += ignore_divide_zero(1, (y - 2)) * self.chromosome[Chromosome._PROTECT] * (
                                creature_map[2, 2] - creature_map[x, y])

        # food
        if food_map[2, 2] > 0:
            # If currently on food
            # The *5 is cheating a bit to make it likelier to eat, but it's
            # competing with 10 other variables, so I hope this will be considered fair.
            # *5 isn't so huge that other more urgent impetus won't block it.
            eatAction = self.chromosome[Chromosome._HUNGRY]

        for x in range(5):
            for y in range(5):
                if food_map[x, y] > 0:
                    # If food nearby
                    xDir += ignore_divide_zero(10, (x - 2)) * self.chromosome[Chromosome._GATHERER]
                    yDir += ignore_divide_zero(10, (y - 2)) * self.chromosome[Chromosome._GATHERER]

        # wall
        for x in range(5):
            for y in range(5):
                if wall_map[x, y] > 0:
                    # If there is wall nearby
                    xDir += ignore_divide_zero(10, (x - 2)) * self.chromosome[Chromosome._INTROVERSION]
                    yDir += ignore_divide_zero(10, (y - 2)) * self.chromosome[Chromosome._INTROVERSION]

        if xDir > 0:
            # Go right
            actions[2] = xDir
        else:
            # Go left
            actions[0] = -xDir

        if yDir > 0:
            # Go down
            actions[3] = yDir
        else:
            # Go up
            actions[1] = -yDir

        actions[4] = eatAction
        # Performs the biggest of xDir, yDir, and eatAction.
        return actions


# Return 0 before dividing to avoid DivideByZero errors.
def ignore_divide_zero(a, b):
    if b == 0:
        return 0
    else:
        return a / b


# Convert bool to 0 or 1
def bool_to_num(a):
    if a:
        return 1
    else:
        return 0


class Generation:
    value = 0


def newGeneration(old_population):
    Generation.value += 1

    # Return list of new agents of length N
    N = len(old_population)

    # Should = 9, but done this way in case code changes.
    chromosomeLength = len(old_population[0].chromosome)

    # This function should also return average fitness of the old_population
    # Fitness for all agents
    # Instead of array of fitness values, it is an array of chromosomes
    # w/ fitness appended at newChromosome[last]
    fitnessTable = np.zeros(shape=(0, chromosomeLength + 1))

    # Loops for each member of old_pop
    for n, creature in enumerate(old_population):
        # Fitness calculated by predetermined weighting
        fitnessVal = 0
        fitnessVal += bool_to_num(creature.alive) * weight.values[Weight._WEIGHT_ALIVE]
        fitnessVal += creature.turn * weight.values[Weight._WEIGHT_TURN]
        fitnessVal += creature.strawb_eats * weight.values[Weight._WEIGHT_STRAWB]
        fitnessVal += creature.enemy_eats * weight.values[Weight._WEIGHT_ENEMY]
        fitnessVal += creature.squares_visited * weight.values[Weight._WEIGHT_TRAVEL]
        fitnessVal += creature.bounces + weight.values[Weight._WEIGHT_BOUNCE]

        # Append fitness as "tag" on to chromosome
        fitnessAdded = np.append(creature.chromosome, fitnessVal)
        fitnessTable = np.vstack([fitnessTable, fitnessAdded])

    # At this point you should sort the agent according to fitness and create new population
    sortedFitness = fitnessTable[np.argsort(fitnessTable[:, chromosomeLength])]

    # Uses roulette wheel for selecting parents
    # The chance of being picked as a parent = fitness of individual / fitness of all individuals

    # Offset is a proportion of the fitness of the least fit surviving individual
    offset = sortedFitness[len(sortedFitness)
                           - int(N * weight.values[Weight._SURVIVAL_RATE])][chromosomeLength] * weight.values[
                 Weight._SURVIVAL_OFFSET]

    # parentCandidates will only contain a proportion of the old_population, according to weight.values[_SURVIVAL_RATE]
    parentCandidates = list()
    tally = 0  # Records the total fitness of all individuals
    for x in range(int(N * weight.values[Weight._SURVIVAL_RATE])):
        # Adding sortedFitness[N - 1] is fittest individual
        tally += sortedFitness[N - x - 1][chromosomeLength] - offset
        sortedFitness[N - x - 1][chromosomeLength] = tally
        parentCandidates.append(sortedFitness[N - x - 1])

    # Populating new generation
    new_population = list()
    for n in range(N):
        # Create new creature
        new_creature = MyCreature()

        parentOne = 0
        randomPick = random.random() * tally
        # Picking first parent according to roulette wheel
        for x in range(len(parentCandidates)):
            if randomPick < parentCandidates[x][chromosomeLength]:
                parentOne = x
                break

        parentTwo = parentOne
        # Picking second parent until it is different from first parent
        while parentTwo == parentOne:
            for x in range(len(parentCandidates)):
                randomPick = random.random() * tally
                if randomPick < parentCandidates[x][chromosomeLength]:
                    parentTwo = x
                    break

        for x in range(chromosomeLength):
            # Calculating mutations
            randomMutation = random.random()

            # Mutate only if it hits mutation rate
            if randomMutation < weight.values[Weight._MUTATION_RATE]:
                # Mutation ranges from -DEGREE to +DEGREE
                randomMutation = (random.random() - 0.5) * weight.values[Weight._MUTATION_DEGREE] * 2
            else:
                randomMutation = 0

            new_val = parentCandidates[random.choice([parentOne, parentTwo])][x] + randomMutation

            new_creature.chromosome[x] = new_val

        # Add the new agent to the new population
        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitnessTable[:, chromosomeLength])

    return new_population, int(avg_fitness)


class WeightConfig:
    values = list()

    def __init__(self, surv_rate,
                 surv_offset,
                 mut_rate,
                 mut_deg,
                 mut_del,
                 we_alive,
                 we_turn,
                 we_size,
                 we_strawb,
                 we_enemy,
                 we_travel,
                 we_bounce):
        self.values = [surv_rate, surv_offset, mut_rate, mut_deg, mut_del,
                       we_alive, we_turn, we_size, we_strawb, we_enemy, we_travel, we_bounce]


print("Using default coded weights")
weight = WeightConfig(SURVIVAL_RATE,
                      SURVIVAL_OFFSET,
                      MUTATION_RATE,
                      MUTATION_DEGREE,
                      MUTATION_DELETERIOUS,
                      WEIGHT_ALIVE,
                      WEIGHT_TURN,
                      WEIGHT_SIZE,
                      WEIGHT_STRAWB,
                      WEIGHT_ENEMY,
                      WEIGHT_TRAVEL,
                      WEIGHT_BOUNCE)
