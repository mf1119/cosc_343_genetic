from types import SimpleNamespace

import numpy as np
import json


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class Joke:
    def __init__(self, q, a):
        self.question = q
        self.answer = a


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    funnyJoke = Joke("What do you call a stick?", "A stick, dumbass.")
    print(funnyJoke.question)
    print(json.dumps(funnyJoke.__dict__))
    file = open("JokeFile", "w")
    file.write(json.dumps(funnyJoke.__dict__))
    file.close()

    file = open("JokeFile", "r")
    copiedJoke = json.loads(file.read(), object_hook=lambda d: SimpleNamespace(**d))
    print(copiedJoke.answer)
    file.close()


    # print_hi('PyCharm')

    arr = np.array([[[1, 2, 3], [4, 5, 6]],
                    [[3, 7, 2], [1, 8, 5]],
                    [[2, 7, 9], [3, 4, 8]]])

    # print(arr)

    unsort = np.array([[1, 6, 3],
                       [5, 9, 2],
                       [3, 8, 5],
                       [8, 1, 4],
                       [9, 1, 2],
                       [2, 7, 9]])

    # print(unsort)
    # print(unsort[0][2])
    # print("-----")
    # print(unsort[np.argsort(unsort[:, 1])])

    #
    # arra = np.array([1, 2, 3, 4])
    # appa = np.append(arra, 5)
    # print(appa[4])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
