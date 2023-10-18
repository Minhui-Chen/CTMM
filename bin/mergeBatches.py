import re, os
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
    _, file_extension = os.path.splitext(snakemake.input.out[0])
    if file_extension == '.npy':
        for f in snakemake.input.out:
            reps += np.load(f, allow_pickle=True).tolist()
    else:
        for f in snakemake.input.out:
            for line in open(f):
                try:
                    reps.append(np.load(line.strip(), allow_pickle=True).item())
                except:
                    print(line.strip())
                    print(np.load(line.strip()))

    out = {}
    merge_dicts(reps, out)
    list2array(out)

    np.save(snakemake.output.out, out)

if __name__ == '__main__':
    main()
