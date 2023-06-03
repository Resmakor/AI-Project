# 197 instancji, 23 cechy, 2 klasy (zdrowy 0, chory 1) dane nieuporządkowane, brak danych nieokreslonych, 24 kolumny z czego jedna odrzucamy calkiem (name)
import matplotlib.ticker as ticker
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Przygotowanie danych z ręcznym usunięciem pierwszej linijki (nazwy cech)"""
filename = "parkinsons.txt"
data = np.loadtxt(filename, delimiter=",", dtype=str)
x = np.concatenate((data[:, 1:17], data[:, 18:]), axis=1).astype(float).T
y_t = data[:, 17].astype(float)
y_t = y_t.reshape(1, y_t.shape[0])
np.transpose([np.array(range(x.shape[0])), x.min(axis=1), x.max(axis=1)])

# Normalizacja
x_min = x.min(axis=1)
x_max = x.max(axis=1)
x_norm_max = 1
x_norm_min = -1
x_norm = np.zeros(x.shape)
for i in range(x.shape[0]):
    x_norm[i, :] = (x_norm_max - x_norm_min) / (x_max[i] - x_min[i]) * (
        x[i, :] - x_min[i]
    ) + x_norm_min
np.transpose([np.array(range(x.shape[0])),
             x_norm.min(axis=1), x_norm.max(axis=1)])

# Przed posortowaniem
plt.plot(y_t[0])
plt.show()

y_t_s_ind = np.argsort(y_t)
x_n_s = np.zeros(x.shape)
y_t_s = np.zeros(y_t.shape)
for i in range(x.shape[1]):
    y_t_s[0, i] = y_t[0, y_t_s_ind[0, i]]
    x_n_s[:, i] = x_norm[:, y_t_s_ind[0, i]]

# Po posortowaniu
plt.plot(y_t_s[0])
plt.show()

hkl.dump([x, y_t, x_norm, x_n_s, y_t_s], "parkinsons.hkl")
x, y_t, x_norm, x_n_s, y_t_s = hkl.load("parkinsons.hkl")
if min(y_t.T)[0] > 0:
    y = y_t.squeeze() - 1  # index of first class should equal to 0
else:
    y = y_t.squeeze()
X = x.T

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2
)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, K):
        super(Model, self).__init__()
        layers = [nn.Linear(input_dim, K[0])]
        for i in range(len(K) - 1):
            layers.append(nn.Linear(K[i], K[i+1]))
        layers.append(nn.Linear(K[-1], output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = F.sigmoid(self.layers[-1](x))
        return x
    
lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
K1 = 6
K2 = 4
max_layers = 10
PK_3D_K = np.zeros((len(lr_vec), max_layers))
max_epoch = 100
PK_max = 0
lr_max_ind = 0
layers_for_max_PK = 0

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

for lr_ind in range(len(lr_vec)):
    for i in range(3, max_layers + 1):
        layers = []
        for j in range(i):
            if j % 2 == 0:
                layers.append(K1)
            else:
                layers.append(K2)
        model = Model(X_train.shape[1], int(max(y) + 1), layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_vec[lr_ind])
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(max_epoch):
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            PK = correct.mean().item() * 100
            print("LR {} | K1 {} | K2 {} | PK {} | layers {} | layers_count {} ".format(lr_vec[lr_ind], K1, K2, PK, layers, len(layers)))
            PK_3D_K[lr_ind, i - 3] = PK

        if PK > PK_max:
            PK_max = PK
            lr_max_ind = lr_ind
            layers_for_max_PK = len(layers)


print("Max PK: {} for LR: {} | K1: {} | K2: {} | layers_count {} ".format(PK_max, lr_vec[lr_max_ind], K1, K2, layers_for_max_PK))

PK_3D_K = PK_3D_K[:, :max_layers-2]
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
Y, X = np.meshgrid(np.log10(lr_vec), np.array([range(3, max_layers + 1)]))
ax.plot_surface(X, Y, PK_3D_K.T, cmap='viridis')
ax.set_xlabel('Liczba warstw')
ax.set_ylabel('Współczynnik uczenia (10^x)')
ax.set_zlabel('PK')
plt.savefig("Fig.1_PK_layers_lr_pytorch_parkinsons.png", bbox_inches="tight")
plt.show()
