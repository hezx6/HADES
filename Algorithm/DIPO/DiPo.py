import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from Algorithm.DIPO.vae import VAE
from Algorithm.DIPO.helpers import EMA, cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, extract, Losses, SinusoidalPosEmb, init_weights


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, 1))

        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Predict_Model(nn.Module):  # 类似于Actor的核心网络，用来预测噪声或者函数值，预测网络的推理是Actor forward的一个步骤
    def __init__(self, state_dim, action_dim, hidden_size=256, time_dim=32):
        super(Predict_Model, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )

        input_dim = state_dim + action_dim + time_dim
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.Mish(), nn.Linear(hidden_size, hidden_size), nn.Mish(), nn.Linear(hidden_size, action_dim))
        self.apply(init_weights)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        out = torch.cat([x, t, state], dim=-1)
        out = self.layer(out)

        return out


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, noise_ratio, beta_schedule="vp", n_timesteps=1000, loss_type="l2", clip_denoised=True, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = Predict_Model(state_dim, action_dim)

        self.max_noise_ratio = noise_ratio
        self.noise_ratio = noise_ratio

        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio

    @torch.no_grad()
    def p_sample_loop(self, state, shape):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

        return x

    @torch.no_grad()
    def sample(self, state, eval=False):
        self.noise_ratio = 0 if eval else self.max_noise_ratio

        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape)
        return action.clamp_(-1.0, 1.0)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, eval=False):
        return self.sample(state, eval)


class DiPo(object):
    def __init__(
        self,
        args,
        state_dim,
        action_dim,
        memory,
        diffusion_memory,
        device,
        writer,
    ):

        self.policy_type = args.policy_type

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio, beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.memory = memory
        self.diffusion_memory = diffusion_memory
        self.action_gradient_steps = args.action_gradient_steps

        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm
        self.cri_grad_norm = args.cri_grad_norm

        self.step = 0
        self.tau = args.tau
        self.actor_target = copy.deepcopy(self.actor)
        self.update_actor_target_every = args.update_actor_target_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim
        self.action_lr = args.action_lr
        self.device = device
        self.threshod = args.threshod

        action_space = {"high": 1, "low": 0}

        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0
        else:
            self.action_scale = (action_space["high"] - action_space["low"]) / 2.0
            self.action_bias = (action_space["high"] + action_space["low"]) / 2.0

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale

        self.memory.append(state, action, reward, next_state, mask)
        self.diffusion_memory.append(state, action)

    def sample_action(self, state, steps, eval, log_writer):

        if steps < self.threshod and eval == False:
            action = np.random.uniform(0, 1, self.action_dim)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state, eval).cpu().data.numpy().flatten()
            action = action * self.action_scale + self.action_bias

        if eval == False and steps % 500 == 0:
            for i in range(self.action_dim):
                log_writer.add_scalar(f"action/action{i+1}", action[i], global_step=steps)

        return action

    def action_gradient(self, batch_size, log_writer):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1.0, 1.0)

        if self.step % 400 == 0:
            log_writer.add_scalar("gradiant/Action_Grad_Norm", actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())

        return states, best_actions

    def train(self, iterations, batch_size=256, log_writer=None):
        for _ in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(states, actions)

            next_actions = self.actor_target(next_states, eval=False)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.cri_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.cri_grad_norm, norm_type=2)
                if self.step % 400 == 0:
                    log_writer.add_scalar("gradiant/Critic_Grad_Norm", critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()

            """ Policy Training """
            states, best_actions = self.action_gradient(batch_size, log_writer)

            actor_loss = self.actor.loss(best_actions, states)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.ac_grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                if self.step % 400 == 0:
                    log_writer.add_scalar("gradiant/Actor_Grad_Norm", actor_grad_norms.max().item(), self.step)
            self.actor_optimizer.step()

            """ Step Target network """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.step % self.update_actor_target_every == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{id}.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic_{id}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic.pth")

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{id}.pth"))
            self.critic.load_state_dict(torch.load(f"{dir}/critic_{id}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pth"))
            self.critic.load_state_dict(torch.load(f"{dir}/critic.pth"))
