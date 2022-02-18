{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datetime import timedelta\n",
    "import os\n",
    "import logging\n",
    "import time\n",
    "import numpy as np\n",
    "\n",
    "from PARyOpt import BayesOpt\n",
    "from PARyOpt.evaluators.async_parse_result_local import AsyncLocalParseResultEvaluator\n",
    "from PARyOpt.evaluators.paryopt_async import ValueNotReady"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# The user has to define the mapping from actual space to optimization space. One default usage is given here:\n",
    "\n",
    "# parameter names:\n",
    "parameter_names = ['donor ratio', 'concentration', 'spin speed', 'annealing temperature', 'additive amount']\n",
    "# lower limits\n",
    "l_limit = np.asarray([0, 0, 0., 0., 0.])\n",
    "# upper limits\n",
    "u_limit = np.asarray([1., 1., 1., 1., 1.])\n",
    "\n",
    "def scale_down_parameters(x: np.array) -> np.array:\n",
    "    x_new = x.copy()\n",
    "    for i,val in enumerate(x):\n",
    "        x_new[i] = (val - l_limit[i]) / (u_limit[i] - l_limit[i])\n",
    "    return x_new\n",
    "\n",
    "\n",
    "def scale_up_parameters(x: np.array) -> np.array:\n",
    "    x_new = x.copy()\n",
    "    for i,val in enumerate(x):\n",
    "        x_new[i] = l_limit[i] + val * (u_limit[i] - l_limit[i])\n",
    "    return x_new"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# Set up functions for running experiments\n",
    "\n",
    "def folder_generator(directory, x) -> None:\n",
    "    \"\"\"\n",
    "    prepares a given folder for performing the simulations. The cost function (out-of-script) will be executed\n",
    "    in this directory for location x. Typically this involves writing a config file, generating/copying meshes and\n",
    "\n",
    "    In our example, we are running a simple case and so does not require any files to be filled. We shall pass the\n",
    "    location of cost function as a command line argument\n",
    "    :param directory:\n",
    "    :param x:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    with open(os.path.join(directory, 'x.txt'), 'w') as f:\n",
    "        actual_param = scale_up_parameters(x)\n",
    "        f.write(', '.join(parameter_names))\n",
    "        f.write('\\n')\n",
    "        f.write(', '.join([str(i) for i in actual_param]))\n",
    "    with open(os.path.join(directory, 'y.txt'), 'w') as f:\n",
    "        f.write('\\n')\n",
    "    with open(os.path.join(directory, 'if_parse.txt'), 'w') as f:\n",
    "        f.write('False')\n",
    "    print('New folder created: {}'.format(directory))\n",
    "\n",
    "\n",
    "def result_parser(directory, x):\n",
    "    \"\"\"\n",
    "    Parses the result from a file and returns the cost function.\n",
    "    The file is written be the actual cost function. One can also do post processing in this function and return the\n",
    "    subsequent value. Based on the construct of our cost function example3_evaluator.py, the generated result.txt\n",
    "    will be in this 'directory'\n",
    "    :param directory:\n",
    "    :param x:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    with open(os.path.join(directory, 'if_parse.txt'), 'r') as f:\n",
    "        if_parse = f.readline().strip()\n",
    "    if if_parse == \"False\":\n",
    "        # cost function evaluation not yet done\n",
    "        return ValueNotReady()\n",
    "    else:\n",
    "        # parse result and return\n",
    "        val = 0.0\n",
    "        with open(os.path.join(directory, 'y.txt')) as f:\n",
    "            val = float(f.readline())\n",
    "        print('Folder completed: {}!'.format(directory))\n",
    "        print('{}\\n{}\\n'.format(parameter_names, ', '.join(scale_up_parameters(x))))\n",
    "        # return negative of value because we want maximization and PARyOpt does minimization\n",
    "        return -1.0*val\n",
    "\n",
    "\n",
    "def user_defined_kappa(iter_num, freq, t_const):\n",
    "    \"\"\"\n",
    "    user defined kappa for multiple acquisition functions\n",
    "    :param iter_num:\n",
    "    :param freq:\n",
    "    :param t_const:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    kappa = 40.5 * (np.sin((iter_num+1) * np.pi / freq) + 1.5) * np.exp(-t_const *iter_num)\n",
    "    return kappa\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "name": "#%% Main function\n"
    }
   },
   "outputs": [],
   "source": [
    "# logging setup\n",
    "logger = logging.getLogger()\n",
    "logger.setLevel(logging.INFO)\n",
    "fh = logging.FileHandler('muri_log_{}.log'.format(time.strftime(\"%Y.%m.%d-%H%M%S\")), mode='a')\n",
    "formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')\n",
    "fh.setFormatter(formatter)\n",
    "logger.addHandler(fh)\n",
    "logger.info('BayesOpt for OPV additive optimization')\n",
    "\n",
    "# define basic parameters\n",
    "ndim = len(parameter_names)\n",
    "# bounds in the normalized space\n",
    "l_bound = 0.0 * np.ones(ndim)\n",
    "u_bound = 1.0 * np.ones(ndim)\n",
    "\n",
    "# experiments per iteration\n",
    "n_opt = 8\n",
    "# max number of iterations\n",
    "iter_max = 8\n",
    "\n",
    "jobs_dir = os.path.join(os.getcwd(), 'opt_jobs')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# Create cost function evaluator\n",
    "\n",
    "# parallel, asynchronous, local, manual\n",
    "evaluator = AsyncLocalParseResultEvaluator(job_generator=folder_generator,\n",
    "                                           jobs_dir=jobs_dir,\n",
    "                                           # checks the folder every 1 minute..\n",
    "                                           wait_time=timedelta(minutes=1),\n",
    "                                           max_pending=8,\n",
    "                                           required_fraction=0.75)\n",
    "\n",
    "logger.info('Optimization evaluations are done in {} directory'.format(jobs_dir))\n",
    "\n",
    "# generate a list of kappa strategies (functions) that correspond to each acquisition function\n",
    "my_kappa_funcs = []\n",
    "for j in range(n_opt):\n",
    "    my_kappa_funcs.append(lambda curr_iter_num, freq=10. * (j * j + 2), t_const=0.8 / (1. + j):\n",
    "                          user_defined_kappa(curr_iter_num, freq=freq, t_const=t_const))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001B[0;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[0;31mKeyboardInterrupt\u001B[0m                         Traceback (most recent call last)",
      "\u001B[0;32m<ipython-input-8-6b94affce32f>\u001B[0m in \u001B[0;36m<module>\u001B[0;34m\u001B[0m\n\u001B[1;32m      5\u001B[0m                  \u001B[0mkern_function\u001B[0m\u001B[0;34m=\u001B[0m\u001B[0;34m'matern_52'\u001B[0m\u001B[0;34m,\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m      6\u001B[0m                  \u001B[0macq_func\u001B[0m\u001B[0;34m=\u001B[0m\u001B[0;34m'LCB'\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mkappa_strategy\u001B[0m\u001B[0;34m=\u001B[0m\u001B[0mmy_kappa_funcs\u001B[0m\u001B[0;34m,\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m----> 7\u001B[0;31m                  if_restart=False)\n\u001B[0m\u001B[1;32m      8\u001B[0m \u001B[0mlogger\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0minfo\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0;34m'BO initialized'\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m      9\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n",
      "\u001B[0;32m~/PROJECTS/machineLearning/bayesianOptimization/paryopt/PARyOpt/paryopt.py\u001B[0m in \u001B[0;36m__init__\u001B[0;34m(self, cost_function, l_bound, u_bound, n_dim, n_opt, n_init, init_strategy, do_init, kern_function, acq_func, acq_func_optimizer, kappa_strategy, constraints, if_restart, restart_filename)\u001B[0m\n\u001B[1;32m    332\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    333\u001B[0m         \u001B[0;32mif\u001B[0m \u001B[0mdo_init\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m--> 334\u001B[0;31m             \u001B[0mself\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0m_initialize\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0m\u001B[1;32m    335\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    336\u001B[0m         \u001B[0;32massert\u001B[0m \u001B[0mall\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mu_bound\u001B[0m \u001B[0;34m>=\u001B[0m \u001B[0ml_bound\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0;34m\"all the upper bound values are not greater than lower bound values\"\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n",
      "\u001B[0;32m~/PROJECTS/machineLearning/bayesianOptimization/paryopt/PARyOpt/paryopt.py\u001B[0m in \u001B[0;36m_initialize\u001B[0;34m(self)\u001B[0m\n\u001B[1;32m    349\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    350\u001B[0m             \u001B[0;31m# save pending and failed directly (since this is initialization)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m--> 351\u001B[0;31m             \u001B[0mcompleted\u001B[0m \u001B[0;34m=\u001B[0m \u001B[0mself\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0m_require_all_cost_function\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mpopulation\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0m\u001B[1;32m    352\u001B[0m             \u001B[0;32mfor\u001B[0m \u001B[0mx\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0my\u001B[0m \u001B[0;32min\u001B[0m \u001B[0mcompleted\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    353\u001B[0m                 \u001B[0;32massert\u001B[0m \u001B[0mlen\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mx\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mshape\u001B[0m\u001B[0;34m)\u001B[0m \u001B[0;34m==\u001B[0m \u001B[0;36m1\u001B[0m \u001B[0;32mand\u001B[0m \u001B[0mx\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mshape\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0;36m0\u001B[0m\u001B[0;34m]\u001B[0m \u001B[0;34m==\u001B[0m \u001B[0mself\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mn_dim\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n",
      "\u001B[0;32m~/PROJECTS/machineLearning/bayesianOptimization/paryopt/PARyOpt/paryopt.py\u001B[0m in \u001B[0;36m_require_all_cost_function\u001B[0;34m(self, xs)\u001B[0m\n\u001B[1;32m    938\u001B[0m         \u001B[0mfailed\u001B[0m \u001B[0;34m=\u001B[0m \u001B[0mlist\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    939\u001B[0m         \u001B[0;32mwhile\u001B[0m \u001B[0mlen\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mpending\u001B[0m\u001B[0;34m)\u001B[0m \u001B[0;34m>\u001B[0m \u001B[0;36m0\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m--> 940\u001B[0;31m             \u001B[0mnew_completed\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mpending\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mnew_failed\u001B[0m \u001B[0;34m=\u001B[0m \u001B[0mself\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mcost_function\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mpending\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0m\u001B[1;32m    941\u001B[0m             \u001B[0mcompleted\u001B[0m \u001B[0;34m+=\u001B[0m \u001B[0mnew_completed\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    942\u001B[0m             \u001B[0mfailed\u001B[0m \u001B[0;34m+=\u001B[0m \u001B[0mnew_failed\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n",
      "\u001B[0;32m~/PROJECTS/machineLearning/bayesianOptimization/paryopt/PARyOpt/evaluators/paryopt_async.py\u001B[0m in \u001B[0;36m__call__\u001B[0;34m(self, xs, old_xs)\u001B[0m\n\u001B[1;32m    129\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    130\u001B[0m     \u001B[0;32mdef\u001B[0m \u001B[0m__call__\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mself\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mxs\u001B[0m\u001B[0;34m:\u001B[0m \u001B[0mList\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0mnp\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0marray\u001B[0m\u001B[0;34m]\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mold_xs\u001B[0m\u001B[0;34m:\u001B[0m \u001B[0mList\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0mnp\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0marray\u001B[0m\u001B[0;34m]\u001B[0m \u001B[0;34m=\u001B[0m \u001B[0mlist\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m--> 131\u001B[0;31m         \u001B[0;32mreturn\u001B[0m \u001B[0mself\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mevaluate_population\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mxs\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mold_xs\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0m\u001B[1;32m    132\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    133\u001B[0m     \u001B[0;32mdef\u001B[0m \u001B[0mevaluate_population\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mself\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mxs\u001B[0m\u001B[0;34m:\u001B[0m \u001B[0mList\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0mnp\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0marray\u001B[0m\u001B[0;34m]\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mif_ready_xs\u001B[0m\u001B[0;34m:\u001B[0m \u001B[0mList\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0mnp\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0marray\u001B[0m\u001B[0;34m]\u001B[0m \u001B[0;34m=\u001B[0m \u001B[0mlist\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;31m\\\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n",
      "\u001B[0;32m~/PROJECTS/machineLearning/bayesianOptimization/paryopt/PARyOpt/evaluators/paryopt_async.py\u001B[0m in \u001B[0;36mevaluate_population\u001B[0;34m(self, xs, if_ready_xs)\u001B[0m\n\u001B[1;32m    185\u001B[0m             \u001B[0;31m# sleep time increases with check iteration and eventually asymptotes at 10 sec\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    186\u001B[0m             \u001B[0;32mif\u001B[0m \u001B[0mn_completed_new\u001B[0m \u001B[0;34m<\u001B[0m \u001B[0mn_req_pts\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m--> 187\u001B[0;31m                 \u001B[0mtime\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0msleep\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0;36m9\u001B[0m \u001B[0;34m*\u001B[0m \u001B[0mnp\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mtanh\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mcheck_iter\u001B[0m\u001B[0;34m)\u001B[0m \u001B[0;34m+\u001B[0m \u001B[0;36m1.\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0m\u001B[1;32m    188\u001B[0m             \u001B[0;32melse\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m    189\u001B[0m                 \u001B[0;32mbreak\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n",
      "\u001B[0;31mKeyboardInterrupt\u001B[0m: "
     ]
    }
   ],
   "source": [
    "# Initialize\n",
    "b_opt = BayesOpt(cost_function=evaluator,\n",
    "                 n_dim=ndim, n_opt=n_opt, n_init=4,\n",
    "                 u_bound=u_bound, l_bound=l_bound,\n",
    "                 kern_function='matern_52',\n",
    "                 acq_func='LCB', kappa_strategy=my_kappa_funcs,\n",
    "                 if_restart=False)\n",
    "logger.info('BO initialized')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/venv/lib/python3.6/site-packages/scipy/optimize/_minimize.py:518: RuntimeWarning: Method Powell cannot handle constraints nor bounds.\n",
      "  RuntimeWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_jm4taooz\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_7rljzi5j\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_ep_vimzw\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_0xwx6mw8\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_xdmps4ez\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_y52eb55a\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_g8l8mqn2\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_a6g2ka_7\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_2t1b1v13\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_vq8hkuoi\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_p3a7_4hq\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_la9v09zl\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_bpldqy0l\n",
      "New folder created: /home/balajip/PROJECTS/machineLearning/bayesianOptimization/paryopt/Projects/muri_adaptive_doe/opt_jobs/folder_mt896d3_\n"
     ]
    }
   ],
   "source": [
    "# Update iterations\n",
    "for curr_iter in range(iter_max):\n",
    "    b_opt.update_iter()\n",
    "    if not curr_iter % 2:\n",
    "        b_opt.estimate_best_kernel_parameters(theta_bounds=[[0.01, 10]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# Post process data\n",
    "\n",
    "# export cost function evaluations to a CSV file\n",
    "b_opt.export_csv(os.path.join(os.getcwd(), 'all_data.csv'))\n",
    "\n",
    "# get current best evaluated value\n",
    "best_location, best_value = b_opt.get_current_best()\n",
    "\n",
    "result_txt = 'Optimization done for {} iterations, best evaluation is at {} with cost: {}'.format(\n",
    "    b_opt.get_current_iteration(), scale_up_parameters(best_location), best_value)\n",
    "\n",
    "logger.info(result_txt)\n",
    "print(result_txt)\n",
    "logger.info('Asynchronous bayesian optimization completed!')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}