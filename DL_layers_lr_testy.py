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
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self.layers[-1](x), dim=1)
        return x


lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
num_layers = np.arange(2, 11)  # Liczba warstw od 2 do 10
PK_2D_lr_layers = np.zeros((len(lr_vec), len(num_layers)))
max_epoch = 100
max_pk = 0
best_lr_index = 0

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

for lr_ind in range(len(lr_vec)):
    for layer_ind in range(len(num_layers)):
        model = Model(X_train.shape[1], int(
            max(y) + 1), num_layers[layer_ind], hidden_dim=1)
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
            correct = (torch.argmax(y_pred, dim=1) ==
                       y_test).type(torch.FloatTensor)
            PK = correct.mean().item() * 100
            print("LR {} | Layers {} | PK {}".format(
                lr_vec[lr_ind], num_layers[layer_ind], PK))
            PK_2D_lr_layers[lr_ind, layer_ind] = PK

            if PK > max_pk:
                max_pk = PK
                best_lr_index = lr_ind

# Rysowanie wykresu
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
X, Y = np.meshgrid(lr_vec, num_layers)
surf = ax.plot_surface(X, Y, PK_2D_lr_layers.T, cmap="viridis")
ax.set_xlabel("Współczynnik uczenia")
ax.set_ylabel("Liczba warstw")
ax.set_zlabel("PK")
ax.view_init(30, 200)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.xaxis.offsetText.set_fontsize(10)
plt.savefig("Fig.1_PK_layers_learning_rate_pytorch_parkinsons.png",
            bbox_inches="tight")
