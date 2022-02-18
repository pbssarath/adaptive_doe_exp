from datetime import timedelta
import os
import logging
import time
import numpy as np

from _submodules.paryopt.PARyOpt import paryopt
from _submodules.paryopt.PARyOpt.evaluators import async_parse_result_local
from _submodules.paryopt.PARyOpt.evaluators import paryopt_async

parameter_names = ['donor ratio', 'concentration', 'spin speed', 'annealing temperature', 'additive amount']
# lower limits
l_limit = np.asarray([0.3, 12., 1000., 80., 0.])
# upper limits
u_limit = np.asarray([0.5, 18., 5000., 120., 8.])


def scale_down_parameters(x: np.array) -> np.array:
    x_new = x.copy()
    for i, val in enumerate(x):
        x_new[i] = (val - l_limit[i]) / (u_limit[i] - l_limit[i])
    return x_new


def scale_up_parameters(x: np.array) -> np.array:
    x_new = x.copy()
    for i, val in enumerate(x):
        x_new[i] = l_limit[i] + val * (u_limit[i] - l_limit[i])
    return x_new


def folder_generator(directory, x) -> None:
    """
    prepares a given folder for performing the simulations. The cost function (out-of-script) will be executed
    in this directory for location x. Typically this involves writing a config file, generating/copying meshes and

    In our example, we are running a simple case and so does not require any files to be filled. We shall pass the
    location of cost function as a command line argument
    :param directory:
    :param x:
    :return:
    """
    with open(os.path.join(directory, 'x.txt'), 'w') as f:
        actual_param = scale_up_parameters(x)
        f.write(', '.join(parameter_names))
        f.write('\n')
        f.write(', '.join([str(i) for i in actual_param]))
    with open(os.path.join(directory, 'y.txt'), 'w') as f:
        f.write('\n')
    with open(os.path.join(directory, 'if_parse.txt'), 'w') as f:
        f.write('False')
    print('New folder created: {}'.format(directory))
    logger.info('Folder created: {}'.format(directory))


def result_parser(directory, x):
    """
    Parses the result from a file and returns the cost function.
    The file is written be the actual cost function. One can also do post processing in this function and return the
    subsequent value. Based on the construct of our cost function example3_evaluator.py, the generated result.txt
    will be in this 'directory'
    :param directory:
    :param x:
    :return:
    """
    with open(os.path.join(directory, 'if_parse.txt'), 'r') as f:
        if_parse = f.readline().strip()
    if if_parse == "False":
        # cost function evaluation not yet done
        return paryopt_async.ValueNotReady()
    else:
        # parse result and return
        val = 0.0
        with open(os.path.join(directory, 'y.txt')) as f:
            val = float(f.readline())
        print(x, val)
        logger.info('Folder completed: {}'.format(directory))
        print('Folder completed: {}!'.format(directory))
        return -1.0 * val


def user_defined_kappa(curr_iter, freq, t_const):
    """
    user defined kappa for multiple acquisition functions
    :param curr_iter:
    :param freq:
    :param t_const:
    :return:
    """
    kappa = 40.5 * (np.sin((curr_iter + 1) * np.pi / freq) + 1.5) * np.exp(-t_const * curr_iter)
    return kappa


def my_init_strategy(args):
    pass


if __name__ == "__main__":

    # logging setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # fh = logging.FileHandler('muri_1_log_{}.log'.format(time.strftime("%Y.%m.%d-%H%M%S")), mode='a')
    fh = logging.FileHandler('muri_1_log.log'.format(time.strftime("%Y.%m.%d-%H%M%S")), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('BayesOpt for OPV additive optimization')

    # define basic parameters
    ndim = len(parameter_names)
    # bounds
    l_bound = -0.2 * np.ones(ndim)
    u_bound = 1.2 * np.ones(ndim)
    # optima per iteration
    n_opt = 8
    # max number of iterations
    iter_max = 8

    jobs_dir = os.path.join(os.getcwd(), 'opt_jobs')
    # parallel, asynchronous, out-of-script
    evaluator = async_parse_result_local.AsyncLocalParseResultEvaluator(job_generator=folder_generator,
                                                                        jobs_dir=jobs_dir,
                                                                        wait_time=timedelta(minutes=1),
                                                                        max_pending=8,
                                                                        required_fraction=0.75)

    logger.info('Optimization evaluations are done in {} directory'.format(jobs_dir))

    # generate a list of kappa strategies (functions) that correspond to each acquisition function
    my_kappa_funcs = []
    for j in range(n_opt):
        my_kappa_funcs.append(lambda curr_iter_num, freq=10. * (j * j + 2), t_const=0.8 / (1. + j):
                              user_defined_kappa(curr_iter_num, freq=freq, t_const=t_const))

    b_opt = paryopt.BayesOpt(cost_function=evaluator,
                             n_dim=ndim, n_opt=n_opt, n_init=4,
                             u_bound=u_bound, l_bound=l_bound,
                             kern_function='matern_52',
                             acq_func='LCB', kappa_strategy=my_kappa_funcs,
                             if_restart=False)
    logger.info('BO initialized')

    for curr_iter in range(iter_max):
        b_opt.update_iter()
        if not curr_iter % 2:
            b_opt.estimate_best_kernel_parameters(theta_bounds=[[0.01, 10]])

    # export cost function evaluations to a CSV file
    b_opt.export_csv(os.path.join(os.getcwd(), 'all_data.csv'))

    # get current best evaluated value
    best_location, best_value = b_opt.get_current_best()

    result_txt = 'Optimization done for {} iterations, best evaluation is at {} with cost: {}'. \
        format(b_opt.get_current_iteration(), best_location, best_value)

    logger.info(result_txt)
    print(result_txt)
    logger.info('Asynchronous bayesian optimization completed!')
