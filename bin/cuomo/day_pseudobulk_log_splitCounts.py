import os, gzip
import numpy as np, pandas as pd

def main():
    # 
    line_no = int( os.popen(f'zcat {snakemake.input.counts}|wc -l').read().split()[0] )

    counts = gzip.open(snakemake.input.counts, 'rt')
    header = counts.readline()

    outputs = [gzip.open(f, 'wt') for f in snakemake.output.counts]
    for f in outputs:
        f.write(header)

    line_no_per_f = np.array_split(np.arange(line_no-1), len(outputs))
    line_no_per_f = [np.max(x) for x in line_no_per_f]

    # 
    i = 0
    k = 0
    while k < len(outputs):
        line = counts.readline()
        outputs[k].write(line)
        if i == line_no_per_f[k]:
            k += 1
        i += 1

    # 
    for f in outputs:
        f.close()

if __name__ == '__main__':
    main()
