import os, sys, re
import numpy as np
from scipy import linalg, optimize, stats
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import STAP
from ctmm import util, wald, ctp, log


def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output
    optim_by_R = params.get('optim_by_R', False)
    method = params.get('method', None)

    #
    batch = params.batch
    outs = [re.sub('/rep/', f'/rep{i}/', params.out) for i in batch]
    for y_f, P_f, nu_f, out_f in zip(
            [line.strip() for line in open(input.y)],
            [line.strip() for line in open(input.P)],
            [line.strip() for line in open(input.nu)],
            outs
    ):
        log.logger.info(f'{y_f}, {P_f}, {nu_f}')

        # cell type number
        C = np.loadtxt(y_f).shape[1]

        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        # Dictionary to store output
        out = {}

        ## REML
        free_reml, free_reml_wald = ctp.free_REML(y_f, P_f, nu_f, method=method,
                                                  jack_knife=True, optim_by_R=optim_by_R)
        out['reml'] = {'free': free_reml, 'wald': {'free': free_reml_wald}}

        # save
        np.save(out_f, out)

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))


if __name__ == '__main__':
    main()
