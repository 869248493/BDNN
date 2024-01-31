import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


# ---------------------------------- pre-processing ----------------------------------------
data = pd.read_csv('SFEW.csv')
data = data.fillna(0)
# Drop movie_name column
data.drop(data.columns[0], axis=1, inplace=True)

# Rename columns
col_name_list = ["LPQ_1", "LPQ_2", "LPQ_3", "LPQ_4", "LPQ_5", "PHOG_1", "PHOG_2", "PHOG_3", "PHOG_4", "PHOG_5"]
for i in range(len(col_name_list)):
    data.rename(columns={data.columns[i + 1]: col_name_list[i]}, inplace=True)

# Make Label 0 Indexed
data['label'] -= 1

# Adding new column of pattern on the first column for one to one relation
pattern = []

element_holder = 0
base_number = 0.1
monotonic_pattern_index = 1
rate = 0.01
for label in data['label']:
    if element_holder == label:
        tmp = base_number + rate * monotonic_pattern_index
        monotonic_pattern_index += 1
        pattern.append(tmp)
    else:
        element_holder = label
        base_number += 0.1
        monotonic_pattern_index = 1
        tmp = base_number + rate * monotonic_pattern_index
        pattern.append(tmp)

data.insert(0,"new_label",pattern)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]
n_features = train_data.shape[1]

# split training data into input and target
train_input = train_data.iloc[:, 2:n_features + 1]
train_target = train_data.iloc[:, 1]
# new pattern output column for reverse input
reverse_train_target = train_data.iloc[:, 0]

# normalise training and testing data by columns
for column in train_input:
    train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# define test set
test_input = test_data.iloc[:, 2:n_features + 1]
test_target = test_data.iloc[:, 1]

for column in test_input:
    test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# Create Tensors to hold inputs and outputs
X = torch.Tensor(train_input.values).float()
Y = torch.Tensor(train_target.values).long()

# Restructure the input and output for reverse training
X_reverse = torch.Tensor(train_target.values).float()

X_reverse.resize_((X_reverse.shape[0],1))
fill = torch.zeros(X_reverse.shape[0],7)

torch.set_printoptions(profile="full")
row_index = 0

for x, y in zip(X_reverse,reverse_train_target):
    fill[row_index, int(x.data)] = y
    row_index += 1

X_reverse = fill.float()
Y_reverse = torch.Tensor(train_input.values).float()


# ---------------------------------- DEFINE NETWORK ----------------------------------------
input_neurons = X.shape[1]
hidden_neurons = 25
output_neurons = 7
learning_rate = 0.05
# True: backward | False: forward
global reverse_flag
reverse_flag = False


class Two_Layer_Network(torch.nn.Module):
    def __init__(self, input_n, hidden_n, output_n):
        # Forward network layers
        super(Two_Layer_Network, self).__init__()
        self.hidden_1 = torch.nn.Linear(input_n, hidden_n)
        self.out = torch.nn.Linear(hidden_n, output_n)

        # Reversed network layers
        self.hidden_reverse = torch.nn.Linear(output_n,hidden_n)
        self.out_reverse = torch.nn.Linear(hidden_n,input_n)

    def forward(self, x):
        if not reverse_flag:
            input_to_hidden = self.hidden_1(x)
            hidden_to_activation = torch.sigmoid(input_to_hidden)
            activation_to_output = self.out(hidden_to_activation)
            return activation_to_output
        else:
            input_to_hidden_reverse = self.hidden_reverse(x)
            hidden_to_activation_reverse = torch.sigmoid(input_to_hidden_reverse)
            activation_to_output_reverse = self.out_reverse(hidden_to_activation_reverse)
            return activation_to_output_reverse


net = Two_Layer_Network(input_neurons, hidden_neurons, output_neurons)

# forward loss function
loss_function = torch.nn.CrossEntropyLoss()
# backward loss function
loss_function_reverse = torch.nn.MSELoss()

# forward optimizer
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
# backward optimizer
optimiser_reverse = torch.optim.SGD(net.parameters(), lr=learning_rate)


# --------------------------------- Train -----------------------------------------
# store all losses for visualisation
all_losses = []


def train():
    # Define how many cycles(forward + reverse)
    num_epochs = 50
    accuracy = 0

    # Define number of epoch per cycle(forward/reverse training)
    iteration_epochs = 100
    max_iteration_epochs = 200

    for epoch in range(num_epochs):
        while (accuracy < 70) and (iteration_epochs < max_iteration_epochs):

            # Perform forward pass: compute predicted y by passing x to the model.
            if reverse_flag:
                Y_pred = net(X_reverse)
            else:
                Y_pred = net(X.float())

            # calculate loss
            loss = calculate_loss(Y_pred)
            all_losses.append(loss.item())

            # convert three-column predicted Y values to one column for comparison
            _, predicted = torch.max(Y_pred, 1)
            # calculate and print accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()
            accuracy = (100 * sum(correct) / total)

            iteration_epochs += 1

            # Clear the gradients before running the backward pass.
            net.zero_grad()

            # Perform backward pass
            loss = loss.to(dtype=torch.float)
            loss.backward()

            if reverse_flag:
                optimiser_reverse.step()
            else:
                optimiser.step()

        # print progress
        progress_log(num_epochs, epoch, loss, accuracy)

        # Export and Reshape the weight for reverse
        reverse()

        accuracy = 0
        iteration_epochs = 0


def calculate_loss(y_pred):
    if reverse_flag:
        loss = loss_function_reverse(y_pred, Y_reverse)
    else:
        loss = loss_function(y_pred, Y)
    return loss


def progress_log(num_epochs,epoch,loss,accuracy):
    # print progress
    if num_epochs % 1 == 0:
        if reverse_flag:
            print('Epoch [%d/%d] Loss: %.4f  backward'
                  % (epoch + 1, num_epochs, loss.item()))
        else:
            print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %% forward'
                  % (epoch + 1, num_epochs, loss.item(), accuracy))


def reverse():
    global reverse_flag
    reverse_flag = not reverse_flag
    if reverse_flag:
        # Transpose the backward weight matrix
        layer_a_to_b = net.out.weight.data
        transpose_a_b = torch.transpose(layer_a_to_b, 0, 1)
        layer_b_to_c = net.hidden_1.weight.data
        transpose_b_c = torch.transpose(layer_b_to_c, 0, 1)
        # Apply to the forward weight matrix
        net.hidden_reverse.weight.data = transpose_a_b.float()
        net.out_reverse.weight.data = transpose_b_c.float()

    else:
        # Transpose the forward weight matrix
        layer_a_to_b = net.out_reverse.weight.data
        transpose_a_b = torch.transpose(layer_a_to_b, 0, 1)
        layer_b_to_c = net.hidden_reverse.weight.data
        transpose_b_c = torch.transpose(layer_b_to_c, 0, 1)
        # Apply to the backward weight matrix
        net.hidden_1.weight.data = transpose_a_b.float()
        net.out.weight.data = transpose_b_c.float()


# --------------------------------- Confusion Matrix of Training -----------------------------------------
def train_matrix():
    confusion = torch.zeros(output_neurons, output_neurons)

    Y_pred = net(X)

    _, predicted = torch.max(Y_pred, 1)

    for i in range(train_data.shape[0]):
        actual_class = Y.data[i]
        predicted_class = predicted.data[i]

        confusion[actual_class][predicted_class] += 1

    print('')
    print('Confusion matrix for training:')
    print(confusion)


# --------------------------------- Test -----------------------------------------
def create_test():
    # create Tensors to hold inputs and outputs
    X_test = torch.Tensor(test_input.values).float()
    Y_test = torch.Tensor(test_target.values).long()

    Y_pred_test = net(X_test)

    # get prediction
    # convert three-column predicted Y values to one column for comparison
    _, predicted_test = torch.max(Y_pred_test, 1)

    # calculate accuracy
    total_test = predicted_test.size(0)
    correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
    print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

    return(Y_test,predicted_test)


def test_matrix(Y_test,predicted_test):
    confusion_test = torch.zeros(output_neurons, output_neurons)

    for i in range(test_data.shape[0]):
        actual_class = Y_test.data[i]
        predicted_class = predicted_test.data[i]
        confusion_test[actual_class][predicted_class] += 1

    print('')
    print('Confusion matrix for testing:')
    print(confusion_test)

# --------------------------------- Graph loss -----------------------------------------
def graph():
    plt.figure()
    plt.plot(all_losses)
    plt.show()


# --------------------------------- Main -----------------------------------------
if __name__ == '__main__':
    train()
    train_matrix()
    (Y_test, predicted_test) = create_test()
    test_matrix(Y_test, predicted_test)
    graph()
