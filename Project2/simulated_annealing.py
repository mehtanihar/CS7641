from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from util import init_weights, compute_loss, perturb_weights, plot_loss
import time
import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve, GridSearchCV, validation_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, roc_auc_score, confusion_matrix


if __name__ == '__main__':
    debug = False
    data1 = pd.read_csv("datasets/pulsar/pulsar_stars.csv", dtype=np.float32)
    Y1 = np.array(data1.pop("target_class"))
    X1 = np.array(data1)
    X_train, X_test, y_train, y_test = train_test_split( X1, Y1, test_size = 0.3)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    T_init = 1e5
    # decay = np.arange(0.25, 0.96, 0.1)
    # decay = np.arange(0.75, 0.76, 0.1)
    decay = np.arange(0.855, 0.865, 0.001)
    
    num_iters = 50000  # number of iterations for a given temperature
    step_size = 0.1
    losses = np.empty(decay.size)
    accuracies_train = np.empty(decay.size)
    accuracies_test = np.empty(decay.size) 
    test_accs = np.empty(decay.size)

    for idx, decay_rate in enumerate(decay):
        np.random.seed(7)  # seed NumPy's random number generator for reproducibility of results
        # Initialize neural network
        print("Dacay rate is:", decay_rate)
        nn = MLPClassifier(hidden_layer_sizes=(10, 3), random_state=7, max_iter=1, warm_start=True)
        nn.fit(X_train, y_train)

        # Initialize weights
        nn.coefs_, nn.intercepts_ = init_weights(X_train.shape[1], list(nn.hidden_layer_sizes))
        loss_next = compute_loss(X_train, y_train, nn)

        T = T_init
        loss = []
        train_acc = []
        test_acc = []
        start = time.time()
        for i in range(num_iters):
            # Save current parameters
            coefs_prev = nn.coefs_
            intercepts_prev = nn.intercepts_
            loss_prev = loss_next

            if debug:
                print('Iteration # %d' % i)
                print('Loss = ', loss_prev)

            # Update parameters
            nn.coefs_ = perturb_weights(nn.coefs_, step_size)
            nn.intercepts_ = perturb_weights(nn.intercepts_, step_size)
            loss_next = compute_loss(X_train, y_train, nn)

            # Metropolis criterion for updating weights
            prob = np.exp((loss_prev - loss_next) / T)
            rand = np.random.rand()
            if loss_next < loss_prev or prob >= rand:
                pass
            else:
                nn.coefs_ = coefs_prev
                nn.intercepts_ = intercepts_prev
                loss_next = loss_prev

            loss.append(loss_next)
            train_acc.append(accuracy_score(y_train, nn.predict(X_train)))
            test_acc.append(accuracy_score(y_test, nn.predict(X_test)))

            T *= decay_rate

        end = time.time()
        runtime = end - start

        losses[idx] = loss[-1]
        accuracies_train[idx] = train_acc[-1]
        accuracies_test[idx] = test_acc[-1]

        # # Plot loss
        plot_loss(loss, filename='./plots/nn_loss_sa_new.png', title='Training loss curve: simulated annealing')

        # Find accuracy on the test set
        y_pred = nn.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy on test data using simulated annealing is %.2f%%' % (test_accuracy * 100))
        test_accs[idx] = test_accuracy*100
        print(classification_report(y_test, y_pred))
        # # Timing information
        print('Time taken to complete simulated annealing is %f seconds' % runtime)



    plt.figure(figsize=(10, 10))
    plt.plot(train_acc, label='sa train', linestyle='--', color='g')
    plt.plot(test_acc, label='sa test', linestyle='-', color='b')
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.title("Accuracy curve with iteration")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('./plots/nn_acc_sa.png')

    plt.figure(figsize=(10, 10))
    plt.plot(accuracies_train, label='sa train', linestyle='--', color='g')
    plt.plot(accuracies_test, label='sa test', linestyle='-', color='b')
    plt.ylabel("Accuracy")
    plt.xlabel("Decay rates")
    plt.title('Accuracies for different decay rates')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('./plots/nn_decays_acc_sa.png')
    #------

    # Plot losses and test accuracies for different decay rates
    fig, ax = plt.subplots()
    plt.grid()
    ax.plot(decay, losses, color='tab:blue', label='Training loss')
    ax.set_xlabel('Temperature decay rate')
    ax.set_ylabel('Loss')
    ax2 = ax.twinx()
    ax2.plot(decay, test_accs, color='tab:red', label='Test accuracy (%)')
    ax2.set_ylabel('Accuracy (%)')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines])
    ax.set_title('SA performance for different decay rates')
    fig.tight_layout()
    plt.savefig('./plots/nn_sa_decay_rates.png')
    plt.figure()
    plt.plot(decay, losses)
    plt.title('SA performance for different decay rates')
    plt.xlabel('Temperature decay rate')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('./plots/nn_sa_decay_rates.png')