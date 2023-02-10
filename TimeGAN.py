"""TimeGAN.

Adapted from Stefan Jansen's book
[Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/21_gans_for_synthetic_time_series/02_TimeGAN_TF2.ipynb).
*TensorFlow2 implemention*
*stock prices*

Which in turn was based Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar's paper
[Time-series Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf),
Neural Information Processing Systems (NeurIPS), 2019.
*TensorFlow1 implementation*
*Google stock data -- Open, High, Low, Close, Adj_Close, Volume*

Last updated: Feb 1, 2023
[Original code](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/timegan) by Jinsung Yoon (jsyoon0823@gmail.com)
"""  # noqa: E501


# 1. Imports & Settings

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model

# from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam  # solve tensorflow issue on M1 Mac

# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")

sns.set_style("white")

RANDOM_SEED = 42

# 2. Experiment Path

results_path = Path("time_gan")
if not results_path.exists():
    results_path.mkdir()

experiment = 2

log_dir = results_path / f"experiment_{experiment:02}"
if not log_dir.exists():
    log_dir.mkdir(parents=True)

hdf_store = results_path / "TimeSeriesGAN.h5"

# 3. Prepare Data

# 3.1. Parameters

seq_len = 24  # number of timepoints in rolling window
batch_size = 128
n_seq = 6  # number of features

features = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]


def select_data():
    df = pd.read_csv("data/GOOGLE_BIG.csv").dropna()
    df.to_hdf(hdf_store, "data/real")


select_data()

# 3.2. Plot Series

df = pd.read_hdf(hdf_store, "data/real")

df.shape

df.head()

axes = df.div(df.iloc[0]).plot(
    subplots=True,
    figsize=(14, 6),
    layout=(3, 2),
    title=features,
    legend=False,
    rot=0,
    lw=1,
    color="k",
)
for ax in axes.flatten():
    ax.set_xlabel("")

plt.suptitle("Price Series")
plt.gcf().tight_layout()
sns.despine()

# 3.3. Correlation

sns.clustermap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(h_neg=20, h_pos=220),
    center=0,
)

# 3.4. Normalize Data

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df).astype(np.float32)

# 3.5. Create rolling window sequences

data = []
for i in range(len(df) - seq_len):
    data.append(scaled_data[i : i + seq_len])

n_windows = len(data)

# 3.6. Create tf.data.Dataset

real_series = (
    tf.data.Dataset.from_tensor_slices(data)
    .shuffle(  # mix the datasets (to make it similar to i.i.d.)
        buffer_size=n_windows, seed=RANDOM_SEED
    )
    .batch(batch_size)
)
real_series_iter = iter(real_series.repeat())

# 3.7. Set up random series generator


def make_random_data():
    while True:
        yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))


random_series = iter(
    tf.data.Dataset.from_generator(make_random_data, output_types=tf.float32)
    .batch(batch_size)
    .repeat()
)

# 4. TimeGAN Components

# 4.1. Network Parameters

hidden_dim = n_seq * 4
num_layers = 3

# 4.2. Set up logger

writer = tf.summary.create_file_writer(log_dir.as_posix())

# 4.3. Input place holders

X = Input(shape=[seq_len, n_seq], name="RealData")
Z = Input(shape=[seq_len, n_seq], name="RandomData")

# 4.4. RNN block generator


def make_rnn(n_layers, hidden_units, output_units, name):
    return Sequential(
        [
            GRU(units=hidden_units, return_sequences=True, name=f"GRU_{i + 1}")
            for i in range(n_layers)
        ]
        + [Dense(units=output_units, activation="sigmoid", name="OUT")],
        name=name,
    )


# 4.5. Embedder & Recovery

embedder = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=hidden_dim, name="Embedder"
)
recovery = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=n_seq, name="Recovery"
)

# 4.6. Generator & Discriminator

generator = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=hidden_dim, name="Generator"
)
discriminator = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=1, name="Discriminator"
)
supervisor = make_rnn(
    n_layers=2, hidden_units=hidden_dim, output_units=hidden_dim, name="Supervisor"
)

# 5. TimeGAN Training

# 5.1. Settings

# iterations = 10000 in blog post, took ~2.5 hours
# iterations = 50000 in original paper
train_steps = 50000
gamma = 1

# 5.2. Generic Loss Functions

mse = MeanSquaredError()
bce = BinaryCrossentropy()

# 6. Phase 1: Autoencoder Training

# 6.1. Architecture

H = embedder(X)
X_tilde = recovery(H)

autoencoder = Model(inputs=X, outputs=X_tilde, name="Autoencoder")

autoencoder.summary()

plot_model(
    autoencoder, to_file=(results_path / "autoencoder.png").as_posix(), show_shapes=True
)

# 6.2. Autoencoder Optimizer

autoencoder_optimizer = Adam()

# 6.3. Autoencoder Training Step


@tf.function
def train_autoencoder_init(x):
    with tf.GradientTape() as tape:
        x_tilde = autoencoder(x)
        embedding_loss_t0 = mse(x, x_tilde)
        e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

    var_list = embedder.trainable_variables + recovery.trainable_variables
    gradients = tape.gradient(e_loss_0, var_list)
    autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
    return tf.sqrt(embedding_loss_t0)


# 6.4. Autoencoder Training Loop

for step in tqdm(range(train_steps)):
    X_ = next(real_series_iter)
    step_e_loss_t0 = train_autoencoder_init(X_)
    with writer.as_default():
        tf.summary.scalar("Loss Autoencoder Init", step_e_loss_t0, step=step)

# 6.5. Persist model

autoencoder.save(log_dir / "autoencoder")

# 7. Phase 2: Supervised training

# 7.1. Define Optimizer

supervisor_optimizer = Adam()

# 7.2. Train Step


@tf.function
def train_supervisor(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        h_hat_supervised = supervisor(h)
        # g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :]) # from blog post
        g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

    var_list = supervisor.trainable_variables
    gradients = tape.gradient(g_loss_s, var_list)
    supervisor_optimizer.apply_gradients(zip(gradients, var_list))
    return g_loss_s


# 7.3. Training Loop

for step in tqdm(range(train_steps)):
    X_ = next(real_series_iter)
    step_g_loss_s = train_supervisor(X_)
    with writer.as_default():
        tf.summary.scalar("Loss Generator Supervised Init", step_g_loss_s, step=step)

# 7.4. Persist Model

supervisor.save(log_dir / "supervisor")

# 8. Joint Training

# 8.1. Generator

# 8.1.1. Adversarial Architecture - Supervised

E_hat = generator(Z)
H_hat = supervisor(E_hat)
Y_fake = discriminator(H_hat)

adversarial_supervised = Model(
    inputs=Z, outputs=Y_fake, name="AdversarialNetSupervised"
)

adversarial_supervised.summary()

plot_model(adversarial_supervised, show_shapes=True)

# 8.1.2. Adversarial Architecture in Latent Space

Y_fake_e = discriminator(E_hat)

adversarial_emb = Model(inputs=Z, outputs=Y_fake_e, name="AdversarialNet")

adversarial_emb.summary()

plot_model(adversarial_emb, show_shapes=True)

# 8.1.3. Mean & Variance Loss

X_hat = recovery(H_hat)
synthetic_data = Model(inputs=Z, outputs=X_hat, name="SyntheticData")

synthetic_data.summary()

plot_model(synthetic_data, show_shapes=True)


def get_generator_moment_loss(y_true, y_pred):
    y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
    y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
    g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    g_loss_var = tf.reduce_mean(
        tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6))
    )
    return g_loss_mean + g_loss_var


# 8.2. Discriminator

# 8.2.1. Architecture: Real Data

Y_real = discriminator(H)
discriminator_model = Model(inputs=X, outputs=Y_real, name="DiscriminatorReal")

discriminator_model.summary()

plot_model(discriminator_model, show_shapes=True)

# 8.3. Optimizers

generator_optimizer = Adam()
discriminator_optimizer = Adam()
embedding_optimizer = Adam()

# 8.4. Generator Train Steps


@tf.function
def train_generator(x, z):
    with tf.GradientTape() as tape:
        y_fake = adversarial_supervised(z)
        generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake), y_pred=y_fake)

        y_fake_e = adversarial_emb(z)
        generator_loss_unsupervised_e = bce(
            y_true=tf.ones_like(y_fake_e), y_pred=y_fake_e
        )
        h = embedder(x)
        h_hat_supervised = supervisor(h)
        generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_hat = synthetic_data(z)
        generator_moment_loss = get_generator_moment_loss(x, x_hat)

        generator_loss = (
            generator_loss_unsupervised
            + generator_loss_unsupervised_e
            + 100 * tf.sqrt(generator_loss_supervised)
            + 100 * generator_moment_loss
        )

    var_list = generator.trainable_variables + supervisor.trainable_variables
    gradients = tape.gradient(generator_loss, var_list)
    generator_optimizer.apply_gradients(zip(gradients, var_list))
    return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss


# 8.5. Embedding Train Step


@tf.function
def train_embedder(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        h_hat_supervised = supervisor(h)
        g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_tilde = autoencoder(x)
        embedding_loss_t0 = mse(x, x_tilde)
        e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)
        e_loss = e_loss_0 + 0.1 * g_loss_s

    var_list = embedder.trainable_variables + recovery.trainable_variables
    gradients = tape.gradient(e_loss, var_list)
    embedding_optimizer.apply_gradients(zip(gradients, var_list))
    return tf.sqrt(embedding_loss_t0)


# 8.6. Discriminator Train Step


@tf.function
def get_discriminator_loss(x, z):
    y_real = discriminator_model(x)
    discriminator_loss_real = bce(y_true=tf.ones_like(y_real), y_pred=y_real)

    y_fake = adversarial_supervised(z)
    discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake), y_pred=y_fake)

    y_fake_e = adversarial_emb(z)
    discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e), y_pred=y_fake_e)

    return (
        discriminator_loss_real
        + discriminator_loss_fake
        + gamma * discriminator_loss_fake_e
    )


@tf.function
def train_discriminator(x, z):
    with tf.GradientTape() as tape:
        discriminator_loss = get_discriminator_loss(x, z)

    var_list = discriminator.trainable_variables
    gradients = tape.gradient(discriminator_loss, var_list)
    discriminator_optimizer.apply_gradients(zip(gradients, var_list))
    return discriminator_loss


# 8.7. Training Loop

step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
for step in range(train_steps):
    # Train generator (twice as often as discriminator)
    for kk in range(2):
        X_ = next(real_series_iter)
        Z_ = next(random_series)

        # Train generator
        step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
        # Train embedder
        step_e_loss_t0 = train_embedder(X_)

    X_ = next(real_series_iter)
    Z_ = next(random_series)
    step_d_loss = get_discriminator_loss(X_, Z_)
    if step_d_loss > 0.15:
        step_d_loss = train_discriminator(X_, Z_)

    if step % 1000 == 0:
        print(
            f"{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | "
            f"g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}"
        )

    with writer.as_default():
        tf.summary.scalar("G Loss S", step_g_loss_s, step=step)
        tf.summary.scalar("G Loss U", step_g_loss_u, step=step)
        tf.summary.scalar("G Loss V", step_g_loss_v, step=step)
        tf.summary.scalar("E Loss T0", step_e_loss_t0, step=step)
        tf.summary.scalar("D Loss", step_d_loss, step=step)

# 8.8. Persist Synthetic Data Generator

synthetic_data.save(log_dir / "synthetic_data")

# 9. Generate Synthetic Data

# 9.1. Generate Data From Random

generated_data = []
for i in range(int(n_windows / batch_size)):
    Z_ = next(random_series)
    d = synthetic_data(Z_)
    generated_data.append(d)

len(generated_data)

generated_data = np.array(np.vstack(generated_data))
generated_data.shape

np.save(log_dir / "generated_data.npy", generated_data)

# 9.2. Rescale

generated_data = scaler.inverse_transform(generated_data.reshape(-1, n_seq)).reshape(
    -1, seq_len, n_seq
)

generated_data.shape

# 9.3. Persist Data

with pd.HDFStore(hdf_store) as store:
    store.put(
        "data/synthetic",
        pd.DataFrame(generated_data.reshape(-1, n_seq), columns=features),
    )

# 9.4. Plot sample Series

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 7))
axes = axes.flatten()

index = list(range(1, 25))
synthetic = generated_data[np.random.randint(n_windows)]

idx = np.random.randint(len(df) - seq_len)
real = df.iloc[idx : idx + seq_len]

for j, feature in enumerate(features):
    (
        pd.DataFrame(
            {"Real": real.iloc[:, j].values, "Synthetic": synthetic[:, j]}
        ).plot(
            ax=axes[j], title=feature, secondary_y="Synthetic", style=["-", "--"], lw=1
        )
    )
sns.despine()
fig.tight_layout()
plt.savefig(results_path / "plot.png")
