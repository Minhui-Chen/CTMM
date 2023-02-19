import re, time
import numpy as np

def merge_dicts(reps, out):
    for key, value in reps[0].items():
        if isinstance(value, dict):
            out[key] = {}
            merge_dicts([rep[key] for rep in reps], out[key])
        else:
            out[key] = [rep[key] for rep in reps]

## list to array
def list2array(dic):
    for key, value in dic.items():
        if isinstance(value, dict):
            list2array(value)
        elif isinstance(value, list):
            dic[key] = np.array(value)

def main():
    reps = []
    for f in snakemake.input.out:
        for line in open(f):
            reps.append( np.load(line.strip(), allow_pickle=True).item() )

    out={}
    merge_dicts(reps, out)
    list2array(out)

    np.save(snakemake.output.out, out)

if __name__ == '__main__':
    main()
