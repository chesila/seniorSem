import time
import copy
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


data = pd.read_csv('ClaMP_Raw-5184.csv', skipinitialspace=True)

data.head()


cl = data['benign'].value_counts()
cl2 = len(data['benign'])
# class is classifying malicious and benign


feature_col = ["e_lfanew", "NumberOfSections", "Characteristics", "SizeOfCode", "AddressOfEntryPoint", "SizeOfImage"]
X = data[feature_col]
y = data.benign

# print(X.shape)
# print(y.shape)
# print(X.describe())
# print(y.describe())

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# print(X.shape())



X_train = normalize(X_train, norm = 'l1')
X_test = normalize(X_test, norm = 'l1')



# print(y_test.value_counts())


# column creations for ML to read from (PANDAS)
column_names = ["e_lfanew", "NumberOfSections", "Characteristics", "SizeOfCode", "AddressOfEntryPoint", "SizeOfImage"
                ]

X_train = pd.DataFrame(X_train, columns=column_names)
X_test = pd.DataFrame(X_test, columns=column_names)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

class Environment1:

    def __init__(self, data, label, history_t=8):
        self.label = label
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.position = []
        self.history = []
        self.history = [0 for _ in range(self.history_t)]
        return self.history  # obs

    def step(self, act):
        reward = abs(act - self.label.iloc[self.t, :]['benign'])
        if reward > 0:
            reward = -1
        else:
            reward = 1

        self.t += 1

        self.history = []
        self.history.append(self.data.iloc[self.t, :]['e_lfanew'])
        self.history.append(self.data.iloc[self.t, :]['NumberOfSections'])
        self.history.append(self.data.iloc[self.t, :]['Characteristics'])
        self.history.append(self.data.iloc[self.t, :]['SizeOfCode'])
        self.history.append(self.data.iloc[self.t, :]['AddressOfEntryPoint'])
        self.history.append(self.data.iloc[self.t, :]['SizeOfImage'])



        if (self.t == len(self.data) - 1):
            self.done = True

        return self.history, reward, self.done


env = Environment1(X_train, y_train)


# return Q, total_losses, total_rewards

# class Q-network
class Q_Network(nn.Module):
    def __init__(self, obs_len, h_s, actions_n):
        super(Q_Network, self).__init__()

        self.fc_val = nn.Sequential(
            #showcases 8x32
            nn.Linear(obs_len, h_s),
            nn.ReLU(),
            nn.Linear(h_s, actions_n),


        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        h = self.fc_val(x)
        return h



input_size = 8
output_size = 2
hidden_size = 32
USE_CUDA = False
LR = 0.001
#6, 32, 2
Q = Q_Network(input_size, hidden_size, output_size)

Q_ast = copy.deepcopy(Q)

if USE_CUDA:
    Q = Q.cuda()
loss_function = nn.MSELoss()

# defining the optimizer
optimizer = optim.Adam(list(Q.parameters()), lr=LR)


epoch_num = 30
step_max = len(env.data)
memory_size = 32
batch_size = 8
gamma = 0.97

memory = []  # Replay memory
total_step = 0
total_rewards = []
total_losses = []
epsilon = 1.0  # exploration rate
epsilon_decrease = 1e-3
epsilon_min = 0.1
start_reduce_epsilon = 200
train_freq = 8
update_q_freq = 20
gamma = 0.97  # discount rate
show_log_freq = 5

accuracy_per_epoch = []


def testing():
    # testing: sets the environment's integration from dataset to train, execution part of training
    test_env = Environment1(X_test, y_test)
    pobs = test_env.reset()
    test_acts = []
    test_rewards = []
    for _ in range(len(test_env.data) - 1):
        pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32)))
        #pact = 1x10? should be 1x8
        pact = np.argmax(pact.data)
        test_acts.append(pact.item())

        obs, reward, done = test_env.step(np.ndarray(pact))
        obs.append(0)
        obs.append(0)
        pobs = obs
        test_rewards.append(reward)
    test_acts.append(0)

    accuracy_per_epoch.append(metrics.accuracy_score(y_test, test_acts))
    fpr, tpr, _ = roc_curve(y_test, test_acts)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('AUC.png')
    plt.show()


# accuracy between 49-50%

# epoch is the number of learning stages for DQN to go through, 30 iterations to be exact
start = time.time()
for epoch in range(epoch_num):
    pobs = env.reset()
    step = 0
    done = False
    total_reward = 0
    total_loss = 0
    if not done and step < step_max:

        # go through steps in training
        pact = np.random.randint(2)
        if np.random.rand() > epsilon:
            pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
            pact = np.argmax(pact.data)

            obs, reward, done = env.step(pact)

            # adding memory
            memory.append((pobs, pact, reward, obs, done))
            memory = np.asarray(memory, dtype='object')
        if len(memory) > memory_size:
            memory.pop(0)

            # training/updating Q
        if len(memory) == memory_size:
            if total_step % train_freq == 0:
                shuffled_memory = np.random.permutation(memory)

                memory_idx = range(len(shuffled_memory))
                for i in memory_idx[::batch_size]:
                    batch = np.array(shuffled_memory[i:i + batch_size])
                    b_problty = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                    b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                    b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                    q = Q(torch.from_numpy(b_problty))
                    q_ = Q_ast(torch.from_numpy(b_obs))
                    maximum_q = np.max(q_.data.numpy(), axis=1)
                    target = copy.deepcopy(q.data)
                    for j in range(batch_size):
                        # Bellman equation
                        target[j, b_pact[j]] = b_reward[j] + gamma * maximum_q[j] * (not b_done[j])
                    # clearing environment
                    Q.zero_grad()
                    # computing loss
                    loss = loss_function(q, target)
                    total_loss += loss.data.item()
                    # compute gradients
                    loss.backward()
                    # adjusting weights
                    optimizer.step()

            if total_step % update_q_freq == 0:
                Q_ast = copy.deepcopy(Q)
            # updating epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            #next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1
        # appending rewards/loss
        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_loss, elapsed_time])))
            # epsilon: agent randomly explores action space at a higher rate
            # loss function used in logistic regression

            # 30	1.0	0	0.0	0.0	3.4301397800445557


def annot_max(x, y, ax=None):
    x_max = x[np.argmax(y)]
    y_max = max(y)
    text = "x={:}, y={:.3f}".format(x_max, y_max)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrow_props = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrow_props, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(x_max, y_max), xytext=(0.94, 0.96), **kw)


def main():

    testing()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    y = accuracy_per_epoch * 30

    plt.plot(x, y, color='green', linewidth=3)
    plt.xlim(1, 31)
    plt.xlabel('speed')
    plt.ylabel('accuracy score')
    plt.grid(True)
    annot_max(x, y)
    plt.savefig('accuracy_graph.png')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Final execution time: ", end - start)
