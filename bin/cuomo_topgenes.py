import numpy as np, pandas as pd
import math

op = np.load(snakemake.input.op, allow_pickle=True).item()
ctp = np.load(snakemake.input.ctp, allow_pickle=True).item()
remlJK = np.load(snakemake.input.remlJK, allow_pickle=True).item()

ctp_data = pd.DataFrame( {'gene': ctp['gene'] } )
for method in snakemake.params.ctp:
    if method in ['reml','he','ml']:
        data = pd.DataFrame( {  'gene':ctp['gene'], 
                                method + '_V_p': ctp[method]['wald']['free']['V'],
                                method + '_beta_p': ctp[method]['wald']['free']['ct_beta']
                                } )
        ctp_data = ctp_data.merge( data, on='gene' )
    elif method == 'remlJK':
        data = pd.DataFrame( {  'gene':remlJK['gene'],
                                method+'_V_p': remlJK['reml']['wald']['free']['V'],
                                method+'_beta_p': remlJK['reml']['wald']['free']['ct_beta']
                                } )
        ctp_data = ctp_data.merge( data, on='gene' )

ctp_data = pd.DataFrame(ctp_data)

f = open(snakemake.output.topgenes, 'w')
f.write( 'CTP\n' )
for method in snakemake.params.ctp:
    f.write( method + '\n' )
    ctp_data = ctp_data.sort_values(method+'_V_p', ascending=False)
    ctp_data.head(20).to_csv(f, index=False, sep='\t', mode='a')
    ctp_data.loc[ctp_data[method+'_beta_p'] > 0.01].head(20).to_csv(
            f, index=False, sep='\t', mode='a')

    ctp_data = ctp_data.sort_values(method+'_beta_p', ascending=False)
    ctp_data.head(20).to_csv(f, index=False, sep='\t', mode='a')
    ctp_data.loc[ctp_data[method+'_V_p'] > 0.01].head(20).to_csv(
            f, index=False, sep='\t', mode='a')


op_data = {'gene': op['gene'] }
for method in snakemake.params.op:
    if method == 'reml':
        op_data[method+'_V_p'] = op[method]['lrt']['free_hom']
        op_data[method+'_beta_p'] = op[method]['lrt']['free_hom']
    else:
        op_data[method+'_V_p'] = op[method]['wald']['free']['V']
        op_data[method+'_beta_p'] = op[method]['wald']['free']['ct_beta']

op_data = pd.DataFrame(op_data)

f.write( 'OP\n' )
for method in snakemake.params.op:
    f.write( method + '\n' )
    op_data = op_data.sort_values(method+'_V_p', ascending=False)
    op_data.head(30).to_csv(f, index=False, sep='\t', mode='a')
    #if method != 'reml':
    #    op_data.loc[op_data[method+'_beta_p'] < ((-1) *math.log10(0.01))].head(20).to_csv(f, index=False, sep='\t', mode='a')
f.close()
