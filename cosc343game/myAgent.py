from datetime import datetime
import random
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np

playerName = "myAgent"
nPercepts = 75  # This is the number of percepts
nActions = 5  # This is the number of actions

# Train against random for 5 generations, then against self for 1 generations
# trainingSchedule = [("random", 140), ("hunter", 0), ("random", 0), ("random", 0)]
trainingSchedule = [("random", 50)]

###
# Traits
###
CHROMOSOME_LENGTH = 9

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
# Weights
###
# Generation related vars
SURVIVAL_RATE = 0.80
SURVIVAL_OFFSET = 0.2
MUTATION_RATE = 0.8
MUTATION_DEGREE = 30
MUTATION_DELETERIOUS = 0.5

# Fitness weights
WEIGHT_ALIVE = 20
WEIGHT_TURN = 1
WEIGHT_SIZE = 10
WEIGHT_STRAWB = 40
WEIGHT_ENEMY = 50
WEIGHT_TRAVEL = 5
WEIGHT_BOUNCE = 1

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

# Logging
path = "log/game_" + datetime.now().strftime("%Y%m%d%H%M%S") + "/"
Path(path).mkdir(parents=True, exist_ok=True)
logFile = open(path + "log.txt", "a")

INIT_RANGE = 100


class MyCreature:
    def __init__(self):
        # The ultimate munching machine
        # self.chromosome = [0, 2000, 3000000, -500, 50000, -500, 0, 1000, 0]
        self.chromosome = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for x in range(CHROMOSOME_LENGTH):
            self.chromosome[x] = random.randrange(-INIT_RANGE, INIT_RANGE)
            # print(self.chromosome)
            # pass
        # print(self.chromosome)

    def AgentFunction(self, percepts):
        # actions = np.zeros((nActions))
        # The 'actions' variable must be returned and it must be a 5-dim numpy vector or a
        # list with 5 numbers:
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
                    xDir += ignoreDivideZero(1, (x - 2)) * self.chromosome[_CONFIDENCE]
                    yDir += ignoreDivideZero(1, (y - 2)) * self.chromosome[_CONFIDENCE]
                    if abs(creature_map[x, y]) > creature_map[2, 2]:
                        # If bigger enemies
                        xDir += ignoreDivideZero(1, (x - 2)) * self.chromosome[_FEARFUL] * (
                                abs(creature_map[x, y]) - creature_map[2, 2])
                        yDir += ignoreDivideZero(1, (y - 2)) * self.chromosome[_FEARFUL] * (
                                abs(creature_map[x, y]) - creature_map[2, 2])
                    elif abs(creature_map[x, y]) < creature_map[2, 2]:
                        # If smaller enemies
                        xDir += ignoreDivideZero(1, (x - 2)) * self.chromosome[_AGGRESSION] * (
                                creature_map[2, 2] - abs(creature_map[x, y]))
                        yDir += ignoreDivideZero(1, (y - 2)) * self.chromosome[_AGGRESSION] * (
                                creature_map[2, 2] - abs(creature_map[x, y]))
                if creature_map[x, y] > 0:
                    # If friendly
                    if not ((x == 2) and (y == 2)):
                        xDir += ignoreDivideZero(1, (x - 2)) * self.chromosome[_CLUMP]
                        yDir += ignoreDivideZero(1, (y - 2)) * self.chromosome[_CLUMP]

                    if creature_map[x, y] > creature_map[2, 2]:
                        # If bigger friends
                        xDir += ignoreDivideZero(1, (x - 2)) * self.chromosome[_SHELTER] * (
                                creature_map[x, y] - creature_map[2, 2])
                        yDir += ignoreDivideZero(1, (y - 2)) * self.chromosome[_SHELTER] * (
                                creature_map[x, y] - creature_map[2, 2])
                    elif creature_map[x, y] < creature_map[2, 2]:
                        # If smaller friends
                        xDir += ignoreDivideZero(1, (x - 2)) * self.chromosome[_PROTECT] * (
                                creature_map[2, 2] - creature_map[x, y])
                        yDir += ignoreDivideZero(1, (y - 2)) * self.chromosome[_PROTECT] * (
                                creature_map[2, 2] - creature_map[x, y])

        # food
        if food_map[2, 2] > 0:
            # If currently on food
            eatAction = self.chromosome[_HUNGRY]

        for x in range(5):
            for y in range(5):
                if food_map[x, y] > 0:
                    # If food nearby
                    xDir += ignoreDivideZero(10, (x - 2)) * self.chromosome[_GATHERER]
                    yDir += ignoreDivideZero(10, (y - 2)) * self.chromosome[_GATHERER]

        # wall
        for x in range(5):
            for y in range(5):
                if wall_map[x, y] > 0:
                    # If there is wall nearby
                    xDir += ignoreDivideZero(10, (x - 2)) * self.chromosome[_INTROVERSION]
                    yDir += ignoreDivideZero(10, (y - 2)) * self.chromosome[_INTROVERSION]

        if (xDir > 0):
            actions[2] = xDir
        else:
            actions[0] = -xDir

        if (yDir > 0):
            actions[3] = yDir
        else:
            actions[1] = -yDir

        actions[4] = eatAction
        #
        #
        # if food_map[2, 2] > 0:
        #     print(np.array([xDir, yDir, eatAction]))

        return actions


def ignoreDivideZero(a, b):
    if b == 0:
        return 0
    else:
        return a / b


def boolToNum(a):
    if a == True:
        return 1
    else:
        return 0


def newGeneration(old_population):
    # Return list of new agents of length N
    N = len(old_population)

    # Logging related
    blueSurvivor = 0
    redSurvivor = N

    # This function should also return average fitness of the old_population
    # Fitness for all agents
    # Instead of array of fitness values, it is an array of chromosomes w/ fitness appended at newChromosome[last]

    chromosomeLength = len(old_population[0].chromosome)

    fitnessTable = np.zeros(shape=(0, chromosomeLength + 1))

    # Loops for each member of old_pop
    for n, creature in enumerate(old_population):
        # fitnessVal = 1
        # fitnessVal +=                       boolToNum(creature.alive) * weight.values[_WEIGHT_ALIVE]
        # if creature.turn > 0:               fitnessVal = fitnessVal * (1+(creature.turn / (100/weight.values[_WEIGHT_TURN])))
        # if creature.strawb_eats > 0:        fitnessVal = fitnessVal * (1+(creature.strawb_eats / (100/weight.values[_WEIGHT_STRAWB])))
        # if creature.enemy_eats > 0:         fitnessVal = fitnessVal * (1+(creature.enemy_eats / (100/weight.values[_WEIGHT_ENEMY])))
        # if creature.squares_visited > 0:    fitnessVal = fitnessVal * (1+(creature.squares_visited / (100/weight.values[_WEIGHT_TRAVEL])))
        # if creature.bounces > 0:            fitnessVal = fitnessVal * (1+(creature.bounces / (100/weight.values[_WEIGHT_BOUNCE])))

        fitnessVal = 0
        fitnessVal += boolToNum(creature.alive) * weight.values[_WEIGHT_ALIVE]
        fitnessVal += creature.turn * weight.values[_WEIGHT_TURN]
        fitnessVal += creature.strawb_eats * weight.values[_WEIGHT_STRAWB]
        fitnessVal += creature.enemy_eats * weight.values[_WEIGHT_ENEMY]
        fitnessVal += creature.squares_visited * weight.values[_WEIGHT_TRAVEL]
        fitnessVal += creature.bounces + weight.values[_WEIGHT_BOUNCE]

        # Append fitness as "tag" on to chromosome
        fitnessAdded = np.append(creature.chromosome, fitnessVal)

        fitnessTable = np.vstack([fitnessTable, fitnessAdded])

        # For logging
        if creature.alive:
            blueSurvivor += 1

        redSurvivor -= creature.enemy_eats

    # At this point you should sort the agent according to fitness and create new population
    sortedFitness = fitnessTable[np.argsort(fitnessTable[:, chromosomeLength])]
    offset = sortedFitness[len(sortedFitness) - int(N * weight.values[_SURVIVAL_RATE])][chromosomeLength] * weight.values[_SURVIVAL_OFFSET]
    parentCandidates = list()
    tally = 0
    for x in range(int(N * weight.values[_SURVIVAL_RATE])):
        tally += sortedFitness[N - x - 1][chromosomeLength] - offset
        sortedFitness[N - x - 1][chromosomeLength] = tally
        parentCandidates.append(sortedFitness[N - x - 1])

    new_population = list()
    for n in range(N):
        # Create new creature
        new_creature = MyCreature()

        parentOne = 0
        randomPick = random.random() * tally

        for x in range(len(parentCandidates)):
            if randomPick < parentCandidates[x][chromosomeLength]:
                parentOne = x
                break

        parentTwo = parentOne
        # for x in range(len(parentCandidates)):
        #     if randomPick < parentCandidates[x][chromosomeLength]:
        #         parentTwo = x
        #         break
        while parentTwo == parentOne:
            for x in range(len(parentCandidates)):
                randomPick = random.random() * tally
                if randomPick < parentCandidates[x][chromosomeLength]:
                    parentTwo = x
                    break

        for x in range(chromosomeLength):
            # totalChoice = parentCandidates[parentOne][chromosomeLength] \
            #               + parentCandidates[parentTwo][chromosomeLength]
            # randomPick = random.random() * totalChoice

            randomMutation = random.random()
            if randomMutation < weight.values[_MUTATION_RATE]:
                randomMutation = weight.values[_MUTATION_DEGREE] * (random.random() * (1 + weight.values[_MUTATION_DELETERIOUS]) - weight.values[_MUTATION_DELETERIOUS])
            else:
                randomMutation = 0

            # new_creature.chromosome[x] = ((parentCandidates[parentOne][x] + parentCandidates[parentTwo][x]) / 2) \
            #                              * randomMutation

            # if randomPick < parentCandidates[parentOne][chromosomeLength]:
            #     new_creature.chromosome[x] = parentCandidates[parentOne][x]
            # else:
            #     new_creature.chromosome[x] = parentCandidates[parentTwo][x]

            new_creature.chromosome[x] = parentCandidates[random.choice([parentOne, parentTwo])][x]

            if new_creature.chromosome[x] > 0:
                new_creature.chromosome[x] + randomMutation
            else:
                new_creature.chromosome[x] - randomMutation

        # Add the new agent to the new population
        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitnessTable[:, chromosomeLength])

    print("Int: " + str(int(parentCandidates[0][0]))
          + ", Gat: " + str(int(parentCandidates[0][1]))
          + ", Hun: " + str(int(parentCandidates[0][2]))
          + ", Con: " + str(int(parentCandidates[0][3]))
          + ", Agg: " + str(int(parentCandidates[0][4]))
          + ", Fea: " + str(int(parentCandidates[0][5]))
          + ", Clm: " + str(int(parentCandidates[0][6]))
          + ", Shl: " + str(int(parentCandidates[0][7]))
          + ", Pro: " + str(int(parentCandidates[0][8]))
          + ", Ftn: " + str(int(parentCandidates[0][9])))

    # For logging
    popInfo = PopInfo(weight, blueSurvivor, redSurvivor, avg_fitness)
    # print(popInfo)
    logFile.write(json.dumps(popInfo.toJson(), indent=4))
    logFile.write("\n")

    return (new_population, int(avg_fitness))


class PopInfo:
    def __init__(self, weights, blue_survivor, red_survivor, average):
        self.weights = weights
        self.blueSurvivor = blue_survivor
        self.redSurvivor = red_survivor
        self.averageFitness = average

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)



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

    def write(self):
        # self = WeightConfig(SURVIVAL_RATE, SURVIVAL_OFFSET, MUTATION_RATE, MUTATION_DEGREE, MUTATION_DELETERIOUS,
        #                       WEIGHT_ALIVE, WEIGHT_TURN, WEIGHT_SIZE, WEIGHT_STRAWB, WEIGHT_ENEMY, WEIGHT_TRAVEL,
        #                       WEIGHT_BOUNCE)
        file = open("weightconfig.txt", "w")

        print("writing")

        file.write(json.dumps(self.__dict__, indent=4))
        file.close()

    def read(self):
        file = open("weightconfig.txt", "r")
        fileWeight = json.loads(file.read(), object_hook=lambda d: SimpleNamespace(**d))
        self.values[0] = fileWeight.values[0]
        self.values[1] = fileWeight.values[1]
        self.values[2] = fileWeight.values[2]
        self.values[3] = fileWeight.values[3]
        self.values[4] = fileWeight.values[4]
        self.values[5] = fileWeight.values[5]
        self.values[6] = fileWeight.values[6]
        self.values[7] = fileWeight.values[7]
        self.values[8] = fileWeight.values[8]
        self.values[9] = fileWeight.values[9]
        self.values[10] = fileWeight.values[10]
        self.values[11] = fileWeight.values[11]
        file.close()

# weight = WeightConfig(SURVIVAL_RATE, SURVIVAL_OFFSET, MUTATION_RATE, MUTATION_DEGREE, MUTATION_DELETERIOUS,
#                               WEIGHT_ALIVE, WEIGHT_TURN, WEIGHT_SIZE, WEIGHT_STRAWB, WEIGHT_ENEMY, WEIGHT_TRAVEL,
#                               WEIGHT_BOUNCE)
# weight.write()

# Setup weights
weightPath = path + "weightconfig.txt"

if not Path(weightPath).is_file():
    print("No specific Weights: Using Default weight configs")
    weightPath = "weightconfig.txt"

weight = 0

if Path(weightPath).is_file():
    print("Using file weights")
    weightFile = open(weightPath, "r")
    fileWeight = json.loads(weightFile.read(), object_hook=lambda d: SimpleNamespace(**d))
    weight = WeightConfig(fileWeight.values[0],
                          fileWeight.values[1],
                          fileWeight.values[2],
                          fileWeight.values[3],
                          fileWeight.values[4],
                          fileWeight.values[5],
                          fileWeight.values[6],
                          fileWeight.values[7],
                          fileWeight.values[8],
                          fileWeight.values[9],
                          fileWeight.values[10],
                          fileWeight.values[11])

else:
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

