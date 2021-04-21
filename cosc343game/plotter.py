from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fileName = "game_20210417233458_plot.txt"

    path = "log/"
    Path(path).mkdir(parents=True, exist_ok=True)
    filePath = path + fileName
    file = open(filePath, "r")
    listFromFile = file.read().splitlines()
    file.close()

    numberList = list()
    for x in range(len(listFromFile)):
        numberList.append([int(i) for i in listFromFile[x].split()])

    array = np.array(numberList)

    x = array[:, 0]
    yFitness = array[:, 10]

    yIntroversion = array[:, 1]
    yGatherer = array[:, 2]
    yHungry = array[:, 3]
    yConfidence = array[:, 3]
    yAggression = array[:, 5]
    yFearful = array[:, 6]
    yClump = array[:, 7]
    yShelter = array[:, 8]
    yProtect = array[:, 9]

    plt.xlabel("Turns")

    plt.plot(x, yFitness, color='red', label='Fitness')

    plt.plot(x, yIntroversion, color='peru', label='Introversion')
    plt.plot(x, yGatherer, color='peachpuff', label='Gatherer')
    plt.plot(x, yHungry, color='coral', label='Hungry')

    plt.plot(x, yConfidence, color='fuchsia', label='Confidence')
    plt.plot(x, yAggression, color='orchid', label='Aggression')
    plt.plot(x, yFearful, color='mediumvioletred', label='Fearful')

    plt.plot(x, yClump, color='blue', label='Clump')
    plt.plot(x, yShelter, color='darkblue', label='Shelter')
    plt.plot(x, yProtect, color='slateblue', label='Protect')
    plt.legend()
    plt.show()
