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

"""
 # m_tilda = 0.0 # tilda corresponds to weights
 # s_tilda = 1.0
 # m_bar = 0.0  # bar corresponds to bias
 # s_bar = 1.0
 """


x_trains = np.asarray([-2.,    -1.8,   -1.,  1.,  1.8,     2.])
y_trains = np.asarray([-3.,  0.2224,    3.,  3.,  0.2224, -3.])

learning_rate = 1e-5
ITERATIONS = 100

L = 2
nn_hidden_layer = 10

def make_nn_params_as_list_of_dicts(
        n_hiddens_per_layer_list=[],
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
                      'b':np.array(list_weights_and_biases[20:30]).reshape(10)})
    dict_list.append({'w': np.array(list_weights_and_biases[10:20]).reshape(10, 1),
                      'b': np.array(list_weights_and_biases[30])})
    return dict_list


def estimate_variational_inference_loss_function(landa, nn_params):
    return -1 *(log_pdf_prior(nn_params) + log_pdf_likelihood(nn_params) - log_pdf_approximate_posterior(landa, nn_params))


def estimate_gradient_variational_loss_functtion(samples):
    pass

def log_pdf_approximate_posterior(landa, nn_params):
    all_weights_and_biases = get_all_weights_and_biases(nn_params)
    log_pdf_weights_biases = []
    for i in range(len(all_weights_and_biases)):
        log_pdf_weights_biases.append(ag_stats.norm.logpdf(all_weights_and_biases[i], landa[2*i], ag_np.exp(landa[2*i+1])))
    log_pdf_approximate_posterior = ag_np.sum(log_pdf_weights_biases)
    return log_pdf_approximate_posterior


calc_gradient_log_pdf_approximate_posterior = autograd.grad(log_pdf_approximate_posterior)


def log_pdf_prior(nn_params):
    all_weights_and_biases = get_all_weights_and_biases(nn_params)
    log_pdf_weights_biases_prior = ag_stats.norm.logpdf(all_weights_and_biases, 0.0, 1)
    log_pdf_prior = ag_np.sum(log_pdf_weights_biases_prior)
    #print "log_pdf_prior", log_pdf_prior
    return log_pdf_prior

def log_pdf_likelihood(nn_params):
    yhat_predictions = predict_y_given_x_with_NN(x_trains, nn_params, activation_func=ag_np.tanh)
    #print "yhat_predictions", yhat_predictions
    log_pdf_likelihood = ag_stats.norm.logpdf(y_trains, yhat_predictions, 0.1)
    log_pdf_likelihood_sum = ag_np.sum(log_pdf_likelihood)
    #print "log_pdf_likelihood_sum",log_pdf_likelihood_sum
    return log_pdf_likelihood_sum


def main():

    time_start = time.time()
    nn_params = make_nn_params_as_list_of_dicts(n_hiddens_per_layer_list=[10], n_dims_input=1, n_dims_output=1)
    all_weights_and_biases = get_all_weights_and_biases(nn_params)

    estimated_loss_mean_in_all_iteration = []

    n_samples = 150
    landa = np.zeros(62, dtype=float)

    for index in range(62):
        if index % 2 ==0:
            landa[index] = np.random.normal(0.0, 1.0, 1)
        else:
            landa[index] = 1

    for iteration in range(ITERATIONS):
        estimated_loss_list_each_iteration = []

        grad_loss = []
        my_samples = []

        for _ in range(n_samples):
            if iteration == 0:
                for i in range(0, len(all_weights_and_biases)):
                    all_weights_and_biases[i] = np.random.normal(landa[2*i], 1, 1)
                nn_params = convert_list_to_dict(all_weights_and_biases)
                my_samples.append(nn_params)
            else:
                for i in range(0, len(all_weights_and_biases)):
                    all_weights_and_biases[i] = np.random.normal(landa[2*i], ag_np.exp(landa[2*i+1]), 1)
                nn_params = convert_list_to_dict(all_weights_and_biases)
                my_samples.append(nn_params)

        for sample in my_samples:
            #print "sample", sample
            estimated_loss = estimate_variational_inference_loss_function(landa, sample)
            #print "estimated_loss", estimated_loss
            estimated_loss_list_each_iteration.append(estimated_loss)
            grad_with_respect_to_landa_diff_samples = calc_gradient_log_pdf_approximate_posterior(landa, sample)
            #print 'grad_with_respect_to_landa_diff_samples', grad_with_respect_to_landa_diff_samples
            grad_loss.append(grad_with_respect_to_landa_diff_samples * estimated_loss)


        grad_los_mean = np.mean(grad_loss, axis= 0)
        landa += learning_rate * grad_los_mean
        #print "grad_loss_mean", grad_los_mean
        estimated_loss_mean_in_all_iteration.append(np.mean(estimated_loss_list_each_iteration, axis=0))

    _, subplot_grid = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(6, 8), squeeze=False)
    subplot_grid[0, 0].plot([x for x in range(ITERATIONS)], estimated_loss_mean_in_all_iteration)
    subplot_grid[0, 0].set_xlabel('iterations')
    subplot_grid[0, 0].set_ylabel('estimated_loss')
    plt.title('estimated loss in 2000 iterations')
    plt.savefig('estimated_loss1.png')


    grad_with_respect_to_landa_diff_samples = -1 * np.array(grad_with_respect_to_landa_diff_samples).reshape(10, 20)
    grad_with_respect_to_landa_diff_samples = np.mean(grad_with_respect_to_landa_diff_samples, axis=0)


if __name__=="__main__":
    main()