import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Optional
from torch.optim import Adam
from torch.distributions import Normal

from Algorithm.SDAC.helpers import EMA, cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, extract, Losses, SinusoidalPosEmb, init_weights


class QNet(nn.Module):
    """Q-network for SDAC algorithm"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], activation=nn.ReLU()):
        super().__init__()
        layers = []
        input_dim = obs_dim + act_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(activation)
            input_dim = size
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class NoisePredictor(nn.Module):
    """Core noise prediction network for diffusion policy"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], activation=nn.ReLU(), hidden_dim=256):
        super().__init__()
        self.act_dim = act_dim

        # Time embedding
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(32), nn.Linear(32, hidden_dim), activation, nn.Linear(hidden_dim, hidden_dim))

        # State embedding
        self.state_mlp = nn.Sequential(nn.Linear(obs_dim, hidden_dim), activation, nn.Linear(hidden_dim, hidden_dim))

        # Main network
        layers = []
        input_dim = hidden_dim + hidden_dim + act_dim  # time + state + action
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(activation)
            input_dim = size

        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, act_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, noisy_action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Predict noise for given noisy action
        Args:
            state: (batch, obs_dim)
            noisy_action: (batch, act_dim)
            time: (batch,)
        Returns:
            predicted noise: (batch, act_dim)
        """
        time_embed = self.time_mlp(time)
        state_embed = self.state_mlp(state)
        x = torch.cat([state_embed, time_embed, noisy_action], dim=-1)
        x = self.net(x)
        return self.output_layer(x)


class GaussianDiffusion(nn.Module):
    """Core diffusion process implementation"""

    def __init__(self, num_timesteps: int, beta_schedule: str = "linear"):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))

    def get_recon(self, t: int, x: torch.Tensor, noise: torch.Tensor):
        """
        Reconstruct the original sample x_0 from x_t and noise using:
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        # 取出对应时间步的 alphas_cumprod 值并添加维度以支持广播
        alpha_bar_t = self.alphas_cumprod[t]  # scalar
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).unsqueeze(-1)  # (batch_size, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t).unsqueeze(-1)  # (batch_size, 1)

        # 重建原始样本
        x_recon = (x - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
        return x_recon

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process (add noise to data)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, noise_predictor: nn.Module, x_start: torch.Tensor, state: torch.Tensor, t: torch.Tensor, weights: torch.Tensor = 1.0, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate loss for noise prediction"""
        if noise is None:
            noise = torch.randn_like(x_start)

        # Forward diffusion
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict noise
        predicted_noise = noise_predictor(state, x_noisy, t)

        # Weighted loss
        loss = F.mse_loss(predicted_noise, noise, reduction="none")
        return (loss * weights).mean()

    def p_sample(self, noise_predictor: nn.Module, x: torch.Tensor, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion sampling step"""
        predicted_noise = noise_predictor(state, x, t)

        # Calculate mean and variance
        alpha_t = extract(self.alphas, t, x.shape)
        alpha_cumprod_t = extract(self.alphas_cumprod, t, x.shape)

        beta_t = extract(self.betas, t, x.shape)
        sqrt_recip_alpha_t = extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        # Different predictions for epsilon (noise) vs x0 (direct prediction)
        model_mean = sqrt_recip_alpha_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)

        # Add noise for next step
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.sqrt(beta_t) * noise

    def p_sample_loop(self, noise_predictor: nn.Module, state: torch.Tensor, shape: Tuple[int], noise_scale: float = 1.0) -> torch.Tensor:
        """Full reverse diffusion sampling loop"""
        device = next(noise_predictor.parameters()).device
        # device = self.betas.device
        x = noise_scale * torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(noise_predictor, x, state, t)

        return x.clamp(-1.0, 1.0)

        # ---------- 2. 反向采样加权 p-loss ----------

    def reverse_sampling_weighted_p_loss(self, noise: torch.Tensor, weights: torch.Tensor, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)

        assert t.ndim == 1 and t.shape[0] == x_t.shape[0]

        noise_pred = model(x_t, t)
        loss = weights * F.mse_loss(noise_pred, noise, reduction="none")
        return loss.mean()


# ==================== SDAC Algorithm ====================


class DiffusionPolicy(nn.Module):
    """Wrapper for diffusion policy with exploration noise"""

    def __init__(self, obs_dim: int, act_dim: int, noise_predictor: nn.Module, diffusion: GaussianDiffusion, num_particles: int = 4, noise_scale: float = 0.1):
        super().__init__()
        self.noise_predictor = noise_predictor
        self.diffusion = diffusion
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.num_particles = num_particles
        self.noise_scale = noise_scale

    def forward(self, state: torch.Tensor, qnet1, qnet2, eval_mode: bool = False) -> torch.Tensor:
        """Sample actions using diffusion process"""
        batch_size = state.shape[0]
        shape = (batch_size, self.act_dim)
        current_noise_scale = 0.0 if eval_mode else self.noise_scale

        if self.num_particles == 1:
            action = self.diffusion.p_sample_loop(self.noise_predictor, state, shape, current_noise_scale)
        else:
            # Parallel sampling with multiple particles
            state_repeat = state.repeat(self.num_particles, 1)
            actions = self.diffusion.p_sample_loop(self.noise_predictor, state_repeat, (batch_size * self.num_particles, self.act_dim), current_noise_scale)
            actions = actions.view(batch_size, self.num_particles, self.act_dim)

            # 计算每个粒子的Q值
            state_expand = state.unsqueeze(1).expand(-1, self.num_particles, -1).reshape(-1, state.shape[-1])
            actions_flat = actions.reshape(-1, self.act_dim)
            q1 = qnet1(state_expand, actions_flat)
            q2 = qnet2(state_expand, actions_flat)
            q = torch.minimum(q1, q2).view(batch_size, self.num_particles)

            # 选择Q值最大的动作
            best_idx = torch.argmax(q, dim=1)
            action = actions[torch.arange(batch_size), best_idx, :]

        return action


from Algorithm.SDAC.replay_memory import list_ReplayMemory


class SDAC_Agent:

    def __init__(self, state_dim, action_dim, args, device, writer):
        self.device = device
        self.act_dim = action_dim
        self.diffusion_steps = args.diffusion_steps
        self.gamma = args.gamma
        self.tau = args.tau
        self.delay_alpha_update = args.delay_alpha_update
        self.num_particles = args.num_particles
        self.replay_buffer = list_ReplayMemory(capacity=50_000, device=device)

        # Set target entropy (auto if None)
        if args.target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = args.target_entropy

        # Create Q networks
        self.q1 = QNet(state_dim, action_dim, args.sdac_q_hidden_sizes).to(device)
        self.q2 = QNet(state_dim, action_dim, args.sdac_q_hidden_sizes).to(device)
        self.target_q1 = QNet(state_dim, action_dim, args.sdac_q_hidden_sizes).to(device)
        self.target_q2 = QNet(state_dim, action_dim, args.sdac_q_hidden_sizes).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Create diffusion policy
        self.noise_predictor = NoisePredictor(state_dim, action_dim, args.sdac_policy_hidden_sizes).to(device)
        self.diffusion = GaussianDiffusion(args.diffusion_steps, args.sdac_beta_schedule).to(device)
        self.policy = DiffusionPolicy(state_dim, action_dim, self.noise_predictor, self.diffusion, args.num_particles, args.sdac_noise_scale).to(device)

        # Entropy temperature
        self.log_alpha = torch.tensor(np.log(5), dtype=torch.float32, requires_grad=True, device=device)

        # Optimizers
        self.q_optimizer = Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.sdac_lr)
        self.policy_optimizer = Adam(self.noise_predictor.parameters(), lr=args.sdac_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=args.alpha_lr)

        self.writer = writer
        # Training state
        self.step = 0

    @torch.inference_mode()  # NOTE: runtime optimization
    def select_action(self, obs, eval=False):
        """Get action for given observation"""
        with torch.no_grad():
            action = self.policy(obs, self.q1, self.q2, eval_mode=eval)
            action = action.squeeze(dim=0).unsqueeze(dim=-1)
            scaled_action = (action + 1) / 2
            return scaled_action

    def append_memory(self, state, action, reward, next_state):
        orin_action = action * 2 - 1

        self.replay_buffer.append(state, orin_action, reward, next_state)

    def train(self, batch_size, log_writer):
        """Perform a single update step"""
        obs, actions, rewards, next_obs = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        obs = torch.cat(obs)
        actions = torch.stack(actions)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.cat(next_obs)
        dones = 0

        # ------------------- Update Q Networks ------------------- #
        with torch.no_grad():
            # Sample actions from policy
            next_actions = self.policy(next_obs, self.q1, self.q2)

            # Add exploration noise
            noise = torch.randn_like(next_actions) * torch.exp(self.log_alpha) * self.policy.noise_scale
            next_actions = next_actions + noise

            # Target Q values
            q1_target = self.target_q1(next_obs, next_actions)
            q2_target = self.target_q2(next_obs, next_actions)
            q_target = torch.minimum(q1_target, q2_target)
            q_backup = rewards + (1 - dones) * self.gamma * q_target

        # Current Q estimates
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)

        # MSE loss against Bellman backup
        q1_loss = F.mse_loss(q1, q_backup)
        q2_loss = F.mse_loss(q2, q_backup)

        # Optimize Q networks
        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()

        # ------------------- Update Policy ------------------- #
        # Sample new actions using current policy
        new_actions = self.policy(obs, self.q1, self.q2)

        # Diffusion process
        t = torch.randint(0, self.diffusion_steps, (obs.shape[0],), device=self.device).long()
        noise = torch.randn_like(new_actions)
        tilde_at = self.diffusion.q_sample(new_actions, t, noise)

        # Repeat samples for Monte Carlo estimation
        reverse_mc_num = 64
        tilde_at = tilde_at.repeat(reverse_mc_num, 1)
        t = t.repeat(reverse_mc_num)
        wide_obs = obs.repeat(reverse_mc_num, 1)

        # Compute policy loss
        def policy_loss_fn():
            noise2 = torch.randn_like(tilde_at)
            recon = self.diffusion.get_recon(t, tilde_at, noise2).clamp(-1, 1)

            with torch.no_grad():
                q_min = torch.minimum(self.q1(wide_obs, recon), self.q2(wide_obs, recon)) * 5.0 / torch.exp(self.log_alpha)

                q_mean, q_std = q_min.mean(), q_min.std()
                q_reshape = q_min.view(-1, reverse_mc_num)
                Z = torch.logsumexp(q_reshape, dim=1, keepdim=True)
                q_weights = torch.exp(q_reshape - Z).flatten()

            loss = self.diffusion.reverse_sampling_weighted_p_loss(noise2, q_weights, lambda x, t: self.noise_predictor(wide_obs, x, t), tilde_at, t)
            return loss, (q_weights, q_min, q_mean, q_std, recon)

        policy_loss, (q_weights, scaled_q, q_mean, q_std, recon) = policy_loss_fn()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ------------------- Update Alpha ------------------- #
        # Alpha loss (adjust entropy temperature)

        if self.step % self.delay_alpha_update == 0:
            approx_entropy = 0.5 * self.act_dim * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * (0.1 * torch.exp(self.log_alpha)) ** 2)
            log_alpha_loss = -self.log_alpha * (-approx_entropy.detach() + self.target_entropy)

            self.alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.alpha_optimizer.step()

            log_writer.add_scalar("algorithm/alpha_loss", log_alpha_loss.item(), self.step)
            log_writer.add_scalar("algorithm/approx_entropy", approx_entropy.mean().item(), self.step)

        # ------------------- Update Target Networks ------------------- #
        with torch.no_grad():
            # Soft update target Q networks
            for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

        self.step += 1

        if self.step % 50 == 0:
            log_writer.add_scalar("algorithm/q1_loss", q1_loss.item(), self.step)
            log_writer.add_scalar("algorithm/q2_loss", q2_loss.item(), self.step)
            log_writer.add_scalar("algorithm/policy_loss", policy_loss.max().item(), self.step)
            log_writer.add_scalar("algorithm/alpha", torch.exp(self.log_alpha).item(), self.step)

        # return {
        #     "q1_loss": q1_loss.item(),
        #     "q2_loss": q2_loss.item(),
        #     "policy_loss": policy_loss.item(),
        #     "alpha_loss": log_alpha_loss.item(),
        #     "alpha": torch.exp(self.log_alpha).item(),
        #     "approx_entropy": approx_entropy.mean().item(),
        # }

    def save_model(self, dir, remark):
        """Save model parameters"""

        if remark is not None:
            path = f"{dir}/actor_{remark}.pt"
        else:
            path = f"{dir}/actor.pt"

        torch.save(
            {
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "noise_predictor_state_dict": self.noise_predictor.state_dict(),
                "target_q1_state_dict": self.target_q1.state_dict(),
                "target_q2_state_dict": self.target_q2.state_dict(),
                "log_alpha": self.log_alpha,
                "optimizers": {
                    "q": self.q_optimizer.state_dict(),
                    "policy": self.policy_optimizer.state_dict(),
                    "alpha": self.alpha_optimizer.state_dict(),
                },
                "step": self.step,
            },
            path,
        )

    def load_model(self, dir, remark):
        """Load model parameters"""

        if remark is not None:
            path = f"{dir}/actor_{remark}.pt"
        else:
            path = f"{dir}/actor.pt"

        checkpoint = torch.load(path, map_location=self.device)

        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.noise_predictor.load_state_dict(checkpoint["noise_predictor_state_dict"])
        self.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
        self.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])

        self.log_alpha = checkpoint["log_alpha"]
        self.q_optimizer.load_state_dict(checkpoint["optimizers"]["q"])
        self.policy_optimizer.load_state_dict(checkpoint["optimizers"]["policy"])
        self.alpha_optimizer.load_state_dict(checkpoint["optimizers"]["alpha"])

        self.step = checkpoint["step"]
