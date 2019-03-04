from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from util import init_weights, compute_loss, perturb_weights, plot_loss
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve, GridSearchCV, validation_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, roc_auc_score, confusion_matrix


if __name__ == '__main__':
    debug = False

    data1 = pd.read_csv("./datasets/pulsar/pulsar_stars.csv", dtype=np.float32)
    Y1 = np.array(data1.pop("target_class"))
    X1 = np.array(data1)
    X_train, X_test, y_train, y_test = train_test_split( X1, Y1, test_size = 0.3)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    # Initialize neural network
    nn = MLPClassifier(hidden_layer_sizes=(10, 3), random_state=0, max_iter=1, warm_start=True)
    nn.fit(X_train, y_train)

    # Randomized search
    step_sz = 0.5
    max_iters = 40000
    eps = 1e-5
    num_restarts = 10
    losses = np.empty(num_restarts)
    accuracies_train = np.empty(num_restarts)
    accuracies_test = np.empty(num_restarts) 
    times = np.empty(num_restarts)
    coefs = []
    intercepts = []
    np.random.seed(7)
    for run in range(num_restarts):
        print("Running restart",run)
        # Initialize weights
        nn.coefs_, nn.intercepts_ = init_weights(X_train.shape[1], list(nn.hidden_layer_sizes))
        loss_next = compute_loss(X_train, y_train, nn)

        train_loss = []
        train_acc = []
        test_acc = []

        start = time.time()
        for it in range(max_iters):
            # Save current parameters
            coefs_prev = nn.coefs_
            intercepts_prev = nn.intercepts_
            loss_prev = loss_next

            if debug:
                print('Iteration %d' % it)
                print('Loss = ', loss_prev)
                print()

            # Update parameters
            nn.coefs_ = perturb_weights(nn.coefs_, step_sz)
            nn.intercepts_ = perturb_weights(nn.intercepts_, step_sz)

            # Keep the updated parameters only if the loss using them decreases
            loss_next = compute_loss(X_train, y_train, nn)
            if loss_next >= loss_prev:
                nn.coefs_ = coefs_prev
                nn.intercepts_ = intercepts_prev
                loss_next = loss_prev

            # train_loss[it] = loss_next
            train_loss.append(loss_next)
            train_acc.append(accuracy_score(y_train, nn.predict(X_train)))
            test_acc.append(accuracy_score(y_test, nn.predict(X_test)))

            diff = loss_prev - loss_next
            if diff > 0 and diff < eps:
                break

        end = time.time()
        times[run] = end - start

        losses[run] = train_loss[-1]
        accuracies_train[run] = train_acc[-1]
        accuracies_test[run] = test_acc[-1]
        coefs.append(nn.coefs_)
        intercepts.append(nn.intercepts_)

    # # Plot loss curve
    plot_loss(train_loss, filename='./plots/nn_loss_rhc.png', title='Training loss curve: randomized hill climbing')
    ## Plot accuracy

    plt.figure(figsize=(10, 10))
    plt.plot(train_acc, label='rhc train', linestyle='--', color='g')
    plt.plot(test_acc, label='rhc test', linestyle='-', color='b')
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.title("Accuracy curve with iteration")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('./plots/nn_acc_rhc.png')

    # Find best weights
    idx = np.argmin(losses)
    nn.coefs_ = coefs[idx]
    nn.intercepts_ = intercepts[idx]

    # Plot losses across runs
    plot_loss(losses, filename='./plots/nn_runs_rhc.png', title='Losses for different runs with RHC', xlabel='Run',
              ylabel='Loss')

    plt.figure(figsize=(10, 10))
    plt.plot(accuracies_train, label='rhc train', linestyle='--', color='g')
    plt.plot(accuracies_test, label='rhc test', linestyle='-', color='b')
    plt.ylabel("Accuracy")
    plt.xlabel("Run")
    plt.title('Accuracies for different runs with RHC')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('./plots/nn_runs_acc_rhc.png')

    # Find accuracy on the test set
    y_pred = nn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy on test data using randomized hill climbing is %.2f%%' % (test_accuracy * 100))
    print(classification_report(y_test, y_pred))
    # Timing information
    print('Average time per RHC run is %f seconds' % np.mean(times))
