import shutil
from datetime import datetime
import json
import os
import random
import sys
import multiprocessing
from pathlib import Path

import game
from cosc343game.myAgent import WeightConfig

POPULATION_COUNT = 20
MUTATION_RATE = 0.8
MUTATION_DEGREE = 5
MUTATION_DELETERIOUS = 0.5

STARTING_RANGE = 50

population = list()


if __name__ == '__main__':
    # game.main(sys.argv[1:])
    # files = os.listdir("log")
    # print(files)

    # for dir in files:
    #     if os.path.isfile("log/" + dir):
    #         print(dir)

    shutil.rmtree("log/", ignore_errors=True)

    weight = WeightConfig(random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE,
                          random.random() * STARTING_RANGE)

    path = "log/game_" + datetime.now().strftime("%Y%m%d%H%M%S") + "/"
    Path(path).mkdir(parents=True, exist_ok=True)
    file = open(path + "weightconfig.txt", "w")
    file.write(json.dumps(weight.__dict__))
    file.close()

    game.main(sys.argv[1:])







