from __future__ import division
import pickle
import copy
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import autograd.numpy as ag_np
import autograd
from scipy.stats import norm
from autograd.scipy import stats as ag_stats

N = 200
D = 1
random_seed = 42

x_trains = np.asarray([-2., -1.8, -1., 1., 1.8, 2.])
y_trains = np.asarray([-3., 0.2224, 3., 3., 0.2224, -3.])

def prob_density_function(x):
    return np.exp(-1 * (x**2))/(np.sqrt(2 * np.pi))

def relu(x):
    return ag_np.maximum(x, 0.)

# def tan_h(x):
#     return ag_np.tanh(x)

def RBF(x):
    return ag_np.exp(-1 * (x**2.))

def calc_potential_energy(nn_params):
    all_weights_biases = get_all_weights_and_biases(nn_params)
    log_likelihoods_weights = []
    for i, weight in enumerate(all_weights_biases):
        log_likelihoods_weights.append(ag_stats.norm.logpdf(weight, 0, 1))
    sum_log_likelihood_weight_prior = ag_np.sum(log_likelihoods_weights)


    yhat_predictions = predict_y_given_x_with_NN(x_trains, nn_params, activation_func=ag_np.tanh)
    log_likelihoods = []
    for i, yhat_prediction in enumerate(yhat_predictions):
        log_likelihoods.append(ag_stats.norm.logpdf(y_trains[i], yhat_prediction, 0.1))
    sum_log_likelihood = ag_np.sum(log_likelihoods)
    return -1 * (sum_log_likelihood_weight_prior + sum_log_likelihood)


calc_gradient_potential_energy = autograd.grad(calc_potential_energy)


def calc_kinetic_energy(vector_p, scalar_m=1):
    sum = 0
    for p in vector_p:
        sum += p ** 2
    return sum/(2 * scalar_m)


def get_all_weights_and_biases(nn_param_list_of_dict):
    all_weights = []
    all_biases = []
    for ll, layer_dict in enumerate(nn_param_list_of_dict):
        all_weights.extend(layer_dict['w'].flatten())
        all_biases.extend(layer_dict['b'].flatten())
    return all_weights + all_biases


def convert_list_to_dict(list_weights_and_biases):
    dict_list = []
    dict_list.append({'w': np.array(list_weights_and_biases[0:10]).reshape(1, 10),
                      'b':np.array(list_weights_and_biases[20:30])})
    dict_list.append({'w': np.array(list_weights_and_biases[10:20]).reshape(10, 1),
                      'b': np.array([list_weights_and_biases[30]])})
    return dict_list


def make_nn_params_as_list_of_dicts(
        n_hiddens_per_layer_list=[5],
        n_dims_input=1,
        n_dims_output=1,
        weight_fill_func=np.zeros,
        bias_fill_func=np.zeros):
    nn_param_list = []
    n_hiddens_per_layer_list = [n_dims_input] + n_hiddens_per_layer_list + [n_dims_output]
    for n_in, n_out in zip(n_hiddens_per_layer_list[:-1], n_hiddens_per_layer_list[1:]):
        nn_param_list.append(
            dict(
                w=weight_fill_func((n_in, n_out)),
                b=bias_fill_func((n_out,)),
            ))
    return nn_param_list


def predict_y_given_x_with_NN(x=None, nn_param_list=None, activation_func=ag_np.tanh):
    for layer_id, layer_dict in enumerate(nn_param_list):
        if layer_id == 0:
            if x.ndim > 1:
                in_arr = x
            else:
                if x.size == nn_param_list[0]['w'].shape[0]:
                    in_arr = x[ag_np.newaxis ,:]
                else:
                    in_arr = x[: ,ag_np.newaxis]
        else:
            in_arr = activation_func(out_arr)
        out_arr = ag_np.dot(in_arr, layer_dict['w']) + layer_dict['b']
    return ag_np.squeeze(out_arr)


def make_proposal_via_leapfrog_steps(
        cur_bnn_params, cur_momentum_vec,
        n_leapfrog_steps=1,
        step_size=1.0,
        calc_gradient_potential_energy=None):

    # Initialize proposed variables as copies of current values
    prop_bnn_params = copy.deepcopy(cur_bnn_params)
    prop_momentum_vec = copy.deepcopy(cur_momentum_vec)

    gradient_potential_energy_vector = get_all_weights_and_biases(calc_gradient_potential_energy(prop_bnn_params))
    #print gradient_potential_energy_vector
    prop_momentum_vec = prop_momentum_vec - (step_size/2.0) * ag_np.array(gradient_potential_energy_vector)
    #print prop_momentum_vec
    #exit(2)
    for step_id in range(n_leapfrog_steps):
        # This will use the grad of kinetic energy (has simple closed form)
        prop_bnn_vector = get_all_weights_and_biases(prop_bnn_params)
        prop_bnn_vector = prop_bnn_vector + step_size * prop_momentum_vec
        prop_bnn_params = convert_list_to_dict(prop_bnn_vector)

        gradient_potential_energy_vector = get_all_weights_and_biases(calc_gradient_potential_energy(prop_bnn_params))

        if step_id < (n_leapfrog_steps - 1):
            # full step update of momentum
            prop_momentum_vec = prop_momentum_vec - step_size * ag_np.array(gradient_potential_energy_vector)
        else:
            # half step update of momentum
            prop_momentum_vec = prop_momentum_vec - step_size * ag_np.array(gradient_potential_energy_vector) / 2

    prop_momentum_vec = -1 * prop_momentum_vec
    return prop_bnn_params, prop_momentum_vec


def run_HMC_sampler(
        init_bnn_params=None,
        n_hmc_iters=100,
        n_leapfrog_steps=1,
        step_size=1.0,
        random_seed=42,
        calc_potential_energy=None,
        calc_kinetic_energy=None,
        calc_gradient_potential_energy=None,
        ):

    prng = np.random.RandomState(int(random_seed))
    # Set initial bnn params
    cur_bnn_params = init_bnn_params
    cur_potential_energy = calc_potential_energy(cur_bnn_params)

    bnn_samples = list()
    bnn_samples.append(cur_bnn_params)

    potential_energy_list = []

    n_accept = 0
    start_time_sec = time.time()
    for t in range(n_hmc_iters):
        # Draw momentum for CURRENT configuration
        cur_momentum_vec = prng.normal(0, 1, len(get_all_weights_and_biases(cur_bnn_params)))

        # Create PROPOSED configuration
        prop_bnn_params, prop_momentum_vec = make_proposal_via_leapfrog_steps(
            cur_bnn_params, cur_momentum_vec,
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=step_size,
            calc_gradient_potential_energy=calc_gradient_potential_energy)

        current_U = calc_potential_energy(cur_bnn_params)
        current_K = calc_kinetic_energy(cur_momentum_vec)
        proposed_U = calc_potential_energy(prop_bnn_params)
        proposed_K = calc_kinetic_energy(prop_momentum_vec)
        accept_proba = np.exp(current_U - proposed_U + current_K - proposed_K)

        # Draw random value from (0,1) to determine if we accept or not
        if np.random.uniform(0, 1) < accept_proba:
            # If here, we accepted the proposal
            n_accept += 1
            cur_bnn_params = prop_bnn_params

        # Update list of samples from "posterior"
        bnn_samples.append(prop_bnn_params)
        # TODO update energy tracking lists

        # Print some diagnostics every 50 iters
        if t < 5 or ((t+1) % 50 == 0) or (t+1) == n_hmc_iters:
            accept_rate = float(n_accept) / float(t+1)
            print("iter %6d/%d after %7.1f sec | accept_rate %.3f" % (
                t+1, n_hmc_iters, time.time() - start_time_sec, accept_rate))

        my_curr_potential_energy = calc_potential_energy(prop_bnn_params)
        potential_energy_list.append(my_curr_potential_energy)

    return (
        bnn_samples,
        dict(
            n_accept=n_accept,
            n_hmc_iters=n_hmc_iters,
            accept_rate=accept_rate), potential_energy_list
        )


def main():

    chain_potential_energy_list = []
    chain_bnn_samples = []
    burn_in = 800
    n_iterations = 2000
    x_ND = np.linspace(-20, 20, N)
    for i in range(3):
        n_hiddens_per_layer_list = [10]
        nn_params = make_nn_params_as_list_of_dicts(n_hiddens_per_layer_list=n_hiddens_per_layer_list ,
                                                    n_dims_input=1, n_dims_output=1)

        prng = np.random.RandomState(int(random_seed))

        all_weights_and_biases = get_all_weights_and_biases(nn_params)
        all_weights_and_biases = np.random.normal(0, 1, len(all_weights_and_biases))
        nn_params = convert_list_to_dict(all_weights_and_biases)

        init_bnn_params = nn_params
        bnn_samples, accept_rate_dict, potential_energy_list = \
            run_HMC_sampler(init_bnn_params, n_hmc_iters=2000, n_leapfrog_steps=10, step_size=0.001,random_seed=42,
            calc_potential_energy=calc_potential_energy, calc_kinetic_energy=calc_kinetic_energy,
                        calc_gradient_potential_energy=calc_gradient_potential_energy )

        chain_potential_energy_list.append(potential_energy_list)
        chain_bnn_samples.append(bnn_samples)

    for i, each_chain in enumerate(chain_potential_energy_list):
        plt.plot(range(len(each_chain)), each_chain, label = 'chain: {}'.format(i))
        plt.title('Potential Energy vs Iterations in different Chains')
        plt.xlabel('iterations')
        plt.ylabel('potential energy')
        plt.legend(loc='best')
        plt.savefig("partB.png")

    n_ten_samples = np.floor(np.linspace(800, 2000, num=1200)).astype(int)

    for i in range(3):
        predicted_y_in_each_chain = []
        for j, n_ten_sample in enumerate(n_ten_samples):
            predicted_y = predict_y_given_x_with_NN(x=x_ND, nn_param_list= chain_bnn_samples[i][n_ten_sample])
            predicted_y_in_each_chain.append(predicted_y)
            plt.plot(x_ND, predicted_y, label='sample: {}, chain: {}'.format(n_ten_sample,i))
            plt.title('10 samples in chain: {}'.format(i))
            plt.xlabel('x')
            plt.ylabel('predicted_y')
            plt.legend(loc='best')
            plt.savefig("partC {}.png".format(i))

        stds = np.std(predicted_y_in_each_chain, axis=0)
        means = np.mean(predicted_y_in_each_chain, axis=0)
        lower_bound = means - 2 * stds
        upper_bound = means + 2 * stds
        plt.figure(figsize=(8, 6))
        plt.xlabel('x')
        plt.ylabel('y_predictions')
        plt.title("empirical_mean, std deviation in chain {}".format(i))
        plt.fill_between(x_ND, lower_bound, upper_bound, facecolor='blue', alpha=0.5)
        plt.legend(loc='best')
        plt.savefig('partD {}.png'.format(i))


if __name__=="__main__":
    main()