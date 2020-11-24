import pickle
import numpy as np
import os
import sys
import itertools
import copy
import datetime
from desc.input_output import read_input


def make_bash_scripts(number, output_dir, ncpu, ngpu, req_mem, times, mode='traverse'):
    """Creates slurm scripts for batch submit jobs.

    Args:
        number (int): how many to create
        output_dir (str): where to create the scripts
        ncpu (int): how many CPU to request
        nhpu (int): how many GPU to request
        req_mem (int): how much memory to request (in GB)
        times (list): list of estimated runtimes for jobs, in minutes)
    """

    # make the directory
    os.makedirs(output_dir, exist_ok=True)

    for i in range(number):
        with open(os.path.join(output_dir, 'driver' + str(i) + '.sh'), 'w+') as f:
            f.write('#!/bin/bash \n')
            f.write('#SBATCH -N 1 \n')
            f.write('#SBATCH -c ' + str(ncpu) + '\n')
            f.write('#SBATCH --mem ' + str(req_mem) + 'G\n')
            if ngpu > 0:
                f.write('#SBATCH -G ' + str(ngpu) + '\n')
            f.write('#SBATCH -o ' +
                    os.path.join(output_dir, 'log' + str(i) + '.out \n'))
            f.write('#SBATCH -t ' +
                    str(datetime.timedelta(minutes=times[i])) + '\n')

            f.write('root_dir=$HOME/DESC \n')
            f.write('module load anaconda \n')
            f.write('conda activate jax \n')
            if mode == 'traverse':
                f.write('cd $root_dir \n')
                f.write('python hyperparam_scan.py ' + str(i) + '\n')
            f.write('exit')


def main(scenario_index=-2):

    checkpt_dir = os.path.expanduser(
        "/projects/EKOLEMEN/DESC/scan_results_10_29/")
    if not os.path.exists(checkpt_dir):
        os.makedirs(checkpt_dir)

    base_name = 'QAS'
    base_inputs = read_input('./examples/DESC/' + base_name)

    hyperparams = {'Mpol': [np.array([6, 8, 10, 12]),
                            np.array([6, 9, 12]),
                            np.array([6, 12])],
                   'Ntor': [np.array([2, 4, 6]),
                            np.array([3, 6]),
                            np.array([6])],
                   'Nstart': [2, 4],      # when to add non axisymmetric modes
                   'bdry_start': [2, 6],  # when to start perturbing bdry
                   'bdry_ratio': [np.array([0, .25, .5, .75, 1]),
                                  np.array([0, .5, 1]),
                                  np.array([0, 1])],
                   'zeta_ratio': [np.array([0, .25, .5, .75, 1]),
                                  np.array([0, .5, 1]),
                                  np.array([0, 1])],
                   'errr_ratio': [1e-4, 1e-5],
                   'gtol': [.1, 10],
                   'nfev': [1000, 5000],
                   #                'zern_mode': ['fringe', 'ansi'],
                   }
    # offset N and bdry perturbations to later in the sequence
    hyperparams['Ntor'] = [np.pad(
        Ntor, (pad, 0)) for Ntor in hyperparams['Ntor'] for pad in hyperparams['Nstart']]
    hyperparams['bdry_ratio'] = [np.pad(bdry_ratio, (bpad, 0)) for bdry_ratio in
                                 hyperparams['bdry_ratio'] for bpad in hyperparams['bdry_start']]
    hyperparams['zeta_ratio'] = [np.pad(bdry_ratio, (bpad, 0)) for bdry_ratio in
                                 hyperparams['zeta_ratio'] for bpad in hyperparams['bdry_start']]
    del hyperparams['Nstart']
    del hyperparams['bdry_start']

    # make it into a dict of lists of dicts, easier for parsing later
    hyperparams = {key: [{key: np.atleast_1d(
        vali)} for vali in hyperparams[key]] for key, val in hyperparams.items()}

    # don't want to combine zeta and bdry perturbations, just do them seperately
    hyperparams['bdry_ratio'] = [{'zeta_ratio': np.array(
        [1]), **foo} for foo in hyperparams['bdry_ratio']]
    hyperparams['zeta_ratio'] = [{'bdry_ratio': np.array(
        [1]), **foo} for foo in hyperparams['zeta_ratio']]
    hyperparams['bdry_zeta'] = hyperparams['bdry_ratio'] + \
        hyperparams['zeta_ratio']
    del hyperparams['zeta_ratio']
    del hyperparams['bdry_ratio']

    # create individial scenarios - list of dicts
    scenarios = []
    import itertools
    for scenario in itertools.product(*list(hyperparams.values())):
        foo = {k: v for d in scenario for k, v in d.items()}
        scenarios.append(foo)

    # make all continuation arrays the same length, and delete ones that are too long
    pops = []
    for i, scenario in enumerate(scenarios):
        maxlen = max([len(val) for val in scenario.values()])
        if maxlen > 10:
            pops.append(i)
        for key, val in scenario.items():
            scenario[key] = np.pad(val, (0, maxlen-len(val)), 'edge')
    for index in sorted(pops, reverse=True):
        del scenarios[index]

    # set nodes based on spectral resolution, set pressure to zero
    for scenario in scenarios:
        scenario['Mnodes'] = (1.5*scenario['Mpol']).astype(int)
        scenario['Nnodes'] = (1.5*scenario['Ntor']).astype(int)
        scenario['pres_ratio'] = np.array([0])
        scenario['verbose'] = 2

    arrs = ['Mpol', 'Ntor', 'Mnodes', 'Nnodes',
            'bdry_ratio', 'pres_ratio', 'zeta_ratio', 'errr_ratio', 'pert_order',
            'ftol', 'xtol', 'gtol', 'nfev']
    # populate non-scanned hyperparams and equilibrium specs
    for scenario in scenarios:
        for key, val in base_inputs.items():
            if key not in scenario:
                if key in arrs:
                    scenario[key] = np.atleast_1d(val[0])
                else:
                    scenario[key] = val


    for scenario in scenarios:
        arr_len = 0
        for a in arrs:
            arr_len = max(arr_len, len(scenario[a]))
        for a in arrs:
            if scenario[a].size == 1:
                scenario[a] = np.broadcast_to(scenario[a], arr_len, subok=True).copy()
            elif scenario[a].size != arr_len:
                raise Exception(
                    'Continuation parameter arrays are not proper lengths, got {}.size=={}, expected {}'.format(a,scenario[a].size,arr_len))


                
    num_scenarios = len(scenarios)
    num_cores = 16
    ngpu = 0
    req_mem = 48
    runtimes = 120*np.ones(num_scenarios)

    ###############
    # Batch Run
    ###############
    if scenario_index == -1:

        make_bash_scripts(num_scenarios, checkpt_dir,
                          num_cores, ngpu, req_mem, runtimes)
        print('Created Driver Scripts in ' + checkpt_dir)
        for i in range(num_scenarios):
            os.system('sbatch {}'.format(os.path.join(
                checkpt_dir, 'driver' + str(i) + '.sh')))
        print('Jobs submitted, exiting')
        return

    ###############
    # Load Scenario and Data
    ###############
    if scenario_index >= 0:
        verbose = 2
        print(datetime.datetime.today().strftime('%c'),
              ' Loading Scenario ' + str(scenario_index) + ':')
        scenario = scenarios[scenario_index]
        print(scenario)

    out_fname = checkpt_dir + base_name + '_scenario_' + str(scenario_index)
    from desc.continuation import solve_eq_continuation
    iterations = solve_eq_continuation(
        scenario, checkpoint_filename=out_fname, device=None)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
