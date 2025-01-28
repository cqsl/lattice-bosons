import jax
import numpy as np
import flax
import flax.linen as nn

import netket as nk
nk.config.netket_experimental_fft_autocorrelation = True

import bnqs
from bnqs.sampler import LocalRule
import bnqs.models as models

from jax import config
config.update("jax_enable_x64", True)
# Check that the default jax backend is 'gpu'
print(jax.default_backend())

import optax

import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--jobid', dest='jobid', help="Job id")
parser.add_argument('--parameters', dest='parameters', help="Python namespace containing simulation parameters")
args = parser.parse_args()

##

jobid = args.jobid
pars = json.load(open(args.parameters))
ld = os.path.dirname(args.parameters)

N = pars['N']	# Latteral dimension of the lattice (L in the main text)
n_dim = pars['n_dim']	# Number of spatial dimensions
extent = pars['extent']	# Latteral dimensions of the lattice along both axes, here (N, N)
n_sites = pars['n_sites']	# Number of lattice sites
pbc = pars['pbc']	# Whether to use PBCs, here (true, true)
n_particles = pars['n_particles']	# Number of particles (N in the main text)
U = pars['U']	# On-site interaction strength in units of J

kernel_size = pars['kernel_size']	# Size of the convolutional filters
features = pars['features']	# Number of channels per convolutional layer
depth = pars['depth']	# Number of layers

n_samples = pars['n_samples']	# Number of samples
chunk_size = pars['chunk_size']	# Chunk size
n_chains = pars['n_chains']	# Number of Markov chains
sweep_factor = pars['sweep_factor']	# sweep_size = sweep_factor * n_sweeps
n_sweeps = pars['n_sweeps']	# Number of Metropolis-Hastings steps per sample / sweep_factor
n_discard_per_chain = pars['n_discard_per_chain']	# Number of samples discarded at each step
n_burnin = pars['n_burnin']	# Number of burn-in samples for thermalizing the chains

n_iter_jastrow = pars['n_iter_jastrow']	# Number of optimization steps for bare Jastrow
lrate_jastrow = pars['lrate_jastrow']	# Learning rate for the Jastrow optimization
dshift_jastrow = pars['dshift_jastrow']	# Diagonal shift for the Jastrow optimization

n_iter = pars['n_iter']	# Number of optimization steps for the full network
lrate = pars['lrate']	# Learning rate for the full network optimization
dshift = pars['dshift']	# Diagonal shift for the full network optimization

ham_dtype = pars['ham_dtype']	# Data type of the Hamiltonian
sampler_dtype = pars['sampler_dtype']	# Data type of the configurations
model_dtype = pars['model_dtype']	# Data type of the parameters of the variatonal Ansatz

##

hi = nk.hilbert.Fock(n_particles=n_particles, N=n_sites)
g = nk.graph.Hypercube(N, n_dim=n_dim)
ha = nk.operator.BoseHubbard(hi, U=U, graph=g, dtype=ham_dtype)

model = models.SQJastrow(g, kernel_init=jax.nn.initializers.normal(np.sqrt(2/n_sites**3)), param_dtype=model_dtype)
rule = LocalRule.from_graph(g)
sampler = nk.sampler.MetropolisSampler(hi, rule, n_chains=n_chains, sweep_size=sweep_factor*n_sweeps, dtype=sampler_dtype)
vs = nk.vqs.MCState(sampler, model=model, n_samples=n_samples, seed=0, chunk_size=chunk_size, n_discard_per_chain=n_discard_per_chain)
print('Number of Jastrow parameters = ', vs.n_parameters)

prefix = 'Jastrow'
suffix = f'.{jobid}'
log_jastrow = nk.logging.JsonLog(os.path.join(ld, prefix+suffix), save_params_every=1)
model_parameters_fname = os.path.join(ld, 'vqs-'+prefix+suffix+'.mpack')
burnin = True

optimizer = optax.sgd(learning_rate=lrate_jastrow)
solver = nk.optimizer.solver.svd
preconditioner=nk.optimizer.SR(diag_shift=dshift_jastrow, solver=solver)
gs = nk.driver.VMC(ha, optimizer, variational_state=vs, preconditioner=preconditioner)


def cb(step, logged_data, driver):
    acceptance = float(driver.state.sampler_state.acceptance)
    logged_data["acceptance"] = acceptance
    with open(model_parameters_fname, 'wb') as file:
        file.write(flax.serialization.to_bytes(driver.state))
    return True

if burnin:
    print('Burn-in in progress...')
    for _ in range(n_burnin):
        vs.sample()
    print('Thermalised!')

print('Run the Jastrow optimisation problem.\n Logger: '
      f'jastrow.{jobid}'
)

gs.run(n_iter=n_iter_jastrow, out=log_jastrow, callback=cb)

e_stats = vs.expect(ha)
print('Jastrow: ', e_stats.mean, e_stats.error_of_mean)

jp = vs.parameters['Jastrow']

##

prefix = 'ResNetJastrow'
suffix = f'.{jobid}'
log = nk.logging.JsonLog(os.path.join(ld, prefix+suffix), save_params_every=1)
model_parameters_fname = os.path.join(ld, 'vqs-'+prefix+suffix+'.mpack')

def cb(step, logged_data, driver):
    acceptance = float(driver.state.sampler_state.acceptance)
    logged_data["acceptance"] = acceptance
    with open(model_parameters_fname, 'wb') as file:
        file.write(flax.serialization.to_bytes(driver.state))
    return True

model = models.ResNetJastrow(g, depth * (features,), n_dim * (kernel_size,), param_dtype=model_dtype, output_activation=nn.gelu, kernel_init=jax.nn.initializers.normal(np.sqrt(2/n_sites**3)))
vs = nk.vqs.MCState(sampler, model=model, n_samples=n_samples, seed=0, chunk_size=chunk_size, n_discard_per_chain=n_discard_per_chain)
print('Number of parameters = ', vs.n_parameters)

params = vs.parameters
params['Jastrow'] = jp
vs.parameters = params

if burnin:
    print('Burn-in in progress...')
    for _ in range(n_burnin):
        vs.sample()
    print('Thermalised!')

optimizer = optax.sgd(learning_rate=lrate)
preconditioner = nk.optimizer.SR(diag_shift=dshift, solver=solver)
gs = nk.driver.VMC(ha, optimizer, variational_state=vs, preconditioner=preconditioner)
gs.run(n_iter=n_iter, out=log, callback=cb)

e_stats = vs.expect(ha)
print('ResNetJastrow: ', e_stats.mean, e_stats.error_of_mean)
