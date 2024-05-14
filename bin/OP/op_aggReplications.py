import numpy as np

# s
s = []
with open(snakemake.output.s, 'w') as f:
	for f2 in snakemake.input.s:
		for line in open(f2):
			f.write(line.strip()+'\n')
			s.append(np.loadtxt(line.strip()))
# nu
nu = []
with open(snakemake.output.nu, 'w') as f:
	for f2 in snakemake.input.nu:
		for line in open(f2):
			f.write(line.strip()+'\n')
			nu.append(np.loadtxt(line.strip()))
cellspecific_var = np.mean(nu)

# P
P = []
for f in snakemake.input.P:
    for line in open(f):
        P.append( np.loadtxt(line.strip()) )
P2 = np.array(P)**2
cellspecific_var = [np.mean( np.sum(nu[i] * P2[i], axis=1) ) for i in range(len(nu))]

# est_pi
pi = []
with open(snakemake.output.pi, 'w') as f:
	for f2 in snakemake.input.pi:
		for line in open(f2):
			f.write(line.strip()+'\n')
			pi.append(np.loadtxt(line.strip()))

# waldNlrt out
def append_dicts(dic, dics):
    for key, value in dic.items():
        if isinstance(value, dict):
            if key not in dics.keys():
                dics[key] = {}
            append_dicts(value, dics[key])
        else:
            if key not in dics.keys():
                dics[key] = [value]
            else:
                dics[key].append(value)

out = {}
## aggregate
for f in snakemake.input.out:
    for line in open(f):
        npy = np.load(line.strip(), allow_pickle=True).item()
        append_dicts(npy, out)

## add components of variance in ML
if 'ml' in out.keys():
    d = out['ml']
    # hom
    if 'hom' in d.keys():
        d['hom']['celltype_main_var'] = [beta_ @ s_ @ beta_ for beta_, s_ in zip(d['hom']['beta']['ct_beta'], s)]
        d['hom']['cellspecific_var'] = cellspecific_var

    # iid
    if 'iid' in d.keys():
        d['iid']['celltype_main_var'] = [beta_ @ s_ @ beta_ for beta_, s_ in zip(d['iid']['beta']['ct_beta'], s)]
        d['iid']['interxn_var'] = [(np.trace(V_ @ s_)+pi_ @ V_ @ pi_) for V_,s_,pi_ in zip(d['iid']['V'],s,pi)]
        if 'W' in d['iid'].keys():
            d['iid']['celltype_noise_var'] = [(np.trace(W_ @ s_)+pi_ @ W_ @ pi_) for W_,s_,pi_ in zip(d['iid']['W'],s,pi)]
        d['iid']['cellspecific_var'] = cellspecific_var
    
    # free
    if 'free' in d.keys():
        d['free']['celltype_main_var'] = [beta_ @ s_ @ beta_ for beta_, s_ in zip(d['free']['beta']['ct_beta'], s)]
        d['free']['interxn_var'] = [(np.trace(V_ @ s_)+pi_ @ V_ @ pi_) for V_,s_,pi_ in zip(d['free']['V'],s,pi)]
        if 'W' in d['free'].keys():
            d['free']['celltype_noise_var']=[(np.trace(W_ @ s_)+pi_ @ W_ @ pi_) for W_,s_,pi_ in zip(d['free']['W'],s,pi)]
        d['free']['cellspecific_var'] = cellspecific_var

    # full
    if 'full' in d.keys():
        d['full']['celltype_main_var'] = [beta_ @ s_ @ beta_ for beta_, s_ in zip(d['full']['beta']['ct_beta'], s)]
        d['full']['interxn_var'] = [(np.trace(V_ @ s_)+pi_ @ V_ @ pi_) for V_,s_,pi_ in zip(d['full']['V'],s,pi)]
        if 'W' in d['full'].keys():
            d['full']['celltype_noise_var']=[(np.trace(W_ @ s_)+pi_ @ W_ @ pi_) for W_,s_,pi_ in zip(d['full']['W'],s,pi)]
        d['full']['cellspecific_var'] = cellspecific_var

## add components of variance in REML
if 'reml' in out.keys():
    d = out['reml']
    # hom
    if 'hom' in d.keys():
        d['hom']['cellspecific_var'] = cellspecific_var

    # iid
    if 'iid' in d.keys():
        d['iid']['interxn_var'] = [(np.trace(V_ @ s_)+pi_ @ V_ @ pi_) for V_,s_,pi_ in zip(d['iid']['V'],s,pi)]
        if 'W' in d['iid'].keys():
            d['iid']['celltype_noise_var'] = [(np.trace(W_ @ s_)+pi_ @ W_ @ pi_) for W_,s_,pi_ in zip(d['iid']['W'],s,pi)]
        d['iid']['cellspecific_var'] = cellspecific_var
    
    # free
    if 'free' in d.keys():
        d['free']['interxn_var'] = [(np.trace(V_ @ s_)+pi_ @ V_ @ pi_) for V_,s_,pi_ in zip(d['free']['V'],s,pi)]
        if 'W' in d['free'].keys():
            d['free']['celltype_noise_var']=[(np.trace(W_ @ s_)+pi_ @ W_ @ pi_) for W_,s_,pi_ in zip(d['free']['W'],s,pi)]
        d['free']['cellspecific_var'] = cellspecific_var

    # full
    if 'full' in d.keys():
        d['full']['interxn_var'] = [(np.trace(V_ @ s_)+pi_ @ V_ @ pi_) for V_,s_,pi_ in zip(d['full']['V'],s,pi)]
        if 'W' in d['full'].keys():
            d['full']['celltype_noise_var']=[(np.trace(W_ @ s_)+pi_ @ W_ @ pi_) for W_,s_,pi_ in zip(d['full']['W'],s,pi)]
        d['full']['cellspecific_var'] = cellspecific_var

## list to array
def list2array(dic):
    for key, value in dic.items():
        if isinstance(value, dict):
            list2array(value)
        elif isinstance(value, list):
            dic[key] = np.array(value)

list2array(out)
np.save(snakemake.output.out, out)
