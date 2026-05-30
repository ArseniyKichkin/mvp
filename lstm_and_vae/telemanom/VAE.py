from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from collections import deque
import math

class VAE(nn.Module):
    def __init__(self, errors, config):
        super(VAE, self).__init__()

        self.batch_size = 70
        self.window_size = 50
        self.fc1 = nn.Linear(self.window_size, 128)
        self.fc21 = nn.Linear(128, 8)
        self.fc22 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 128)
        self.fc4 = nn.Linear(128, self.window_size)
        self.train_loss = 0
        self.quantile = None
        self.mu_ref_norm = None
        self.mu_train_mean = None
        self.mu_train_std = None
        self.max_normal_score = 0
        self.anomaly_scores = []
        self.test_anomaly_scores = []
        self.mu_refs = []
        self.test_losses = []
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.anomaly_sequences = []
        self.buffer = 2*self.window_size
        self.config = config
        self.p = 0.16

        self.percentile = 55
        self.args = {
            "batch_size": 64,
            "epochs": 10,
            "no_accel": True,
            "seed": 1,
            "log_interval": 10
        }


        self.train_model(errors=errors, epochs=10)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def score_window(self, window_data, window_recon, mu, logvar, beta, prev_score=None):
        point_errors = torch.abs(window_data - window_recon)
        recon_error = torch.mean(point_errors)
        latent_dev = torch.norm(
            (mu - self.mu_train_mean) / self.mu_train_std
        )
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        score = recon_error + beta * kl + 0.01 * latent_dev
        return score

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + KLD

    def train_model(self, epochs, errors):
        self.train()

        n_windows = len(errors) - self.window_size + 1
        window_indices = list(range(n_windows))
        for epoch in range(1, epochs + 1):
            self.train_loss = 0
            total_samples = 0

            np.random.shuffle(window_indices)
            for batch_start in range(0, n_windows, self.batch_size):
                batch_indices = window_indices[batch_start:batch_start + self.batch_size]

                batch_data = []
                for idx in batch_indices:
                    window = errors[idx:idx + self.window_size]
                    batch_data.append(window)
                data_batch = torch.FloatTensor(np.array(batch_data))

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self(data_batch)

                loss = self.loss_function(recon_batch, data_batch, mu, logvar)
                loss.backward()
                self.train_loss += loss.item()
                self.optimizer.step()
                total_samples += len(data_batch)

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, self.train_loss/total_samples))

        with torch.no_grad():
            window_indices = list(range(n_windows))
            for batch_start in range(0, n_windows, self.batch_size):
                batch_indices = window_indices[batch_start:batch_start + self.batch_size]

                batch_data = []
                for idx in batch_indices:
                    window = errors[idx:idx + self.window_size]
                    batch_data.append(window)
                data_batch = torch.FloatTensor(np.array(batch_data))
                recon_batch, mu, logvar = self(data_batch)
                for i in range(len(recon_batch)):
                    self.mu_refs.append(mu[i])
                mu_stack = torch.stack(self.mu_refs)
                self.mu_train_mean = torch.mean(torch.stack(self.mu_refs), dim=0)
                self.mu_train_std = torch.std(mu_stack, dim=0) + 1e-6
                for i in range(len(recon_batch)):
                    score = self.score_window(window_recon=recon_batch[i], window_data=data_batch[i], mu=mu[i], logvar=logvar[i], beta=1.)
                    self.anomaly_scores.append(score.item())

            self.quantile = 2 * np.quantile(self.anomaly_scores,  1.)
            print("Quantiles: 0.9995 = {} and 1.0 = {}".format(self.quantile, np.quantile(self.anomaly_scores, 1.)))


            print('Epoch = {}, score = {} '.format(epoch, self.quantile))

    def mitigate_fp(self):
        e_max = []
        anomaly_seqs = [i for i in self.anomaly_sequences]
        print('anomaly_seqs', " = ", anomaly_seqs)
        num_seqs_removed = 0
        for i in range(len(self.anomaly_sequences)):
            e_max.append((i, self.test_anomaly_scores[i]))
        e_max = sorted(e_max, key=lambda x: -x[1])
        e_max.append((-1, torch.tensor(self.max_normal_score)))
        start_of_mitigate = len(self.anomaly_sequences) - 1
        for i in range(1, len(e_max)):
            e_max_prev = e_max[i - 1][-1].item()
            e_max_cur = e_max[i][-1].item()
            if (e_max_prev - e_max_cur) / e_max_prev > self.p:
                start_of_mitigate = i
        for i in range(start_of_mitigate, len(e_max) - 1):
            idx = e_max[i][0]

            if idx >= len(anomaly_seqs) or idx < 0:
                continue

            if anomaly_seqs[idx] in self.anomaly_sequences:
                self.anomaly_sequences.remove(anomaly_seqs[idx])
                num_seqs_removed += 1
        print('Mitigated {} sequences'.format(num_seqs_removed))



    def test_model(self, errors, is_inference=False):
        self.eval()
        test_loss = 0
        n_windows = len(errors) - self.window_size + 1
        window_indices = list(range(n_windows))
        prev_scores = deque()
        with torch.no_grad():
            for batch_start in range(0, n_windows, self.batch_size):
                batch_indices = window_indices[batch_start:batch_start + self.batch_size]
                for idx in batch_indices:
                    window = errors[idx:idx + self.window_size]
                    data = torch.tensor(window, dtype=torch.float32)

                    recon_batch, mu, logvar = self(data)
                    loss = self.loss_function(recon_batch, data, mu, logvar).item() / n_windows
                    test_loss += loss
                    score = self.score_window(window_recon=recon_batch, window_data=data, mu=mu, logvar=logvar, beta=1.)
                    prev_scores.append(score)
                    if self.quantile >= score > self.max_normal_score and not math.isnan(score) and not math.isinf(score):
                        self.max_normal_score = score
                    if self.window_size * len(prev_scores) > self.buffer:
                        prev_scores.popleft()

                    if (score > self.quantile ) or math.isnan(score):
                        if len(self.anomaly_sequences) != 0 and (idx - self.anomaly_sequences[-1][-1] + self.window_size - 1) <= self.buffer:
                            self.anomaly_sequences[-1][-1] = idx + self.window_size - 1 + self.config.l_s
                            if score > self.test_anomaly_scores[-1]:
                                self.test_anomaly_scores[-1] = score
                        else:
                            self.anomaly_sequences.append([idx + self.config.l_s , idx + self.window_size + self.config.l_s ])
                            self.test_anomaly_scores.append(score)
                        if is_inference:
                            self.cnt+=1
                            self.inference_scores.append(score.item())
                            print("Inference called!!!")
                    self.test_losses.append(test_loss)


            print('====> Test set loss: {:.4f}'.format(test_loss))