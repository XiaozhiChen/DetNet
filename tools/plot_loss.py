#!/usr/bin/python

import numpy as np
import re
import click
from matplotlib import pylab as plt


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax1.set_ylim([0.0, 1.5])
    # ax2.set_ylabel('accuracy %')
    for i, log_file in enumerate(files):
        loss_iterations, losses, cls_loss_iterations, cls_losses = parse_log(log_file)
        # disp_results(fig, ax1, ax2, loss_iterations, losses, cls_loss_iterations, cls_losses)
        disp_results(fig, ax1, loss_iterations, losses)
    plt.show()


def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    cls_pattern = r"Train net output #1 (?P<iter_num>\d+), loss_cls = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    cls_losses = []
    cls_loss_iterations = []

    for r in re.findall(cls_pattern, log):
        cls_loss_iterations.append(int(r[0]))
        cls_losses.append(float(r[1]))

    cls_loss_iterations = np.array(cls_loss_iterations)
    cls_losses = np.array(cls_losses)

#     accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    # accuracies = []
    # accuracy_iterations = []
    # accuracies_iteration_checkpoints_ind = []

    # for r in re.findall(accuracy_pattern, log):
        # iteration = int(r[0])
        # accuracy = float(r[1]) * 100

        # if iteration % 10000 == 0 and iteration > 0:
            # accuracies_iteration_checkpoints_ind.append(len(accuracy_iterations))

        # accuracy_iterations.append(iteration)
        # accuracies.append(accuracy)

    # accuracy_iterations = np.array(accuracy_iterations)
    # accuracies = np.array(accuracies)

    return loss_iterations, losses, cls_loss_iterations, cls_losses


def disp_results(fig, ax1, loss_iterations, losses, color_ind = 0):
    modula = len(plt.rcParams['axes.color_cycle'])
#     loss_iterations = loss_iterations[:100]
#     losses = losses[:100]
    ax1.plot(loss_iterations, losses, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula])

def disp_results2(fig, ax1, ax2, loss_iterations, losses, cls_loss_iterations, cls_losses, color_ind = 0):
    modula = len(plt.rcParams['axes.color_cycle'])
#     loss_iterations = loss_iterations[:100]
#     losses = losses[:100]
    ax1.plot(loss_iterations, losses, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula])
    # ax2.plot(cls_loss_iterations, cls_losses, 'o', color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])
    # ax2.plot(accuracy_iterations, accuracies, plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])
    # ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[accuracies_iteration_checkpoints_ind], 'o', color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])



if __name__ == '__main__':
    main()
