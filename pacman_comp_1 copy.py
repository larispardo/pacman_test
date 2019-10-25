## run with
# >> python pacman_comp_1.py -n 10 -g DirectionalGhost

import csv
import importlib
import numpy as np

from pacman import *
import textDisplay

students = {
    'check1':np.nan
}

if __name__ == '__main__':
    args = readCommand( sys.argv[1:] ) # Get game components based on input
    args['display'] = textDisplay.NullGraphics()
    args['numGames'] = 20
    w = csv.writer(open("student_scores.csv", "a"))

    for key in students:
        try:
            gens = [25]
            mutes = [0.3]
            paths = [15]
            for gen in gens:
                for mute in mutes:
                    for path in paths:
                        module = importlib.import_module(key+'.CompetitionPacman')
                        args['pacman'] = module.CompAgent(path, gen, mute)
                        # exec('from key import CompetitionPacman')
                        out = runGames( **args)
                        scores = [o.state.getScore() for o in out]
                        wins = [o.state.isWin() for o in out]
                        winRate = wins.count(True) / float(len(wins))
                        row = [gen,mute,path, np.mean(scores), winRate]
                        w.writerow(row)
        except ImportError as e:
            print('Error with', key)
            print(e)

print('')
print('#'*50)
print('#'*50)
print('#'*50)
print('')

for key in students:
    print(key, students[key])

# Save results to a csv file

print('!!')