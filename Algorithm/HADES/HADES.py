import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch


from Algorithm.HADES.helpers import EMA, cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, extract, Losses, SinusoidalPosEmb, init_weights
from torch_geometric.nn import GCNConv, global_mean_pool
from Algorithm.HADES.replay_memory import ReplayMemory, DiffusionMemory, list_ReplayMemory


class GCN_Predict_Model(nn.Module):
    def __init__(self, net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, num_net_gcn_layers, num_dag_gcn_layers, action_len, action_dim, normalize, hidden_size=1024, time_dim=32):
        super(GCN_Predict_Model, self).__init__()

        # ===============  GCN ==============
        self.normalize = normalize
        self.net_embedding_dim = net_embedding_dim
        self.dag_embedding_dim = dag_embedding_dim

        self.net_gcn_layers = nn.ModuleList([GCNConv(net_node_feat_dim if i == 0 else net_embedding_dim, net_embedding_dim) for i in range(num_net_gcn_layers)])

        self.dag_gcn_layers = nn.ModuleList([GCNConv(dag_node_feat_dim if i == 0 else dag_embedding_dim, dag_embedding_dim) for i in range(num_dag_gcn_layers)])

        self.dag_nodes_num = action_len

        # ========== time encoder  ============
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )

        self.fusion_model = nn.Sequential(nn.Linear(dag_embedding_dim + time_dim + action_len * action_dim, hidden_size), nn.Mish(), nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, action_len * action_dim))
        self.apply(init_weights)

    def forward(self, action, time, state, batch_size):
        time_embedding = self.time_mlp(time)

        if len(state) == 2:  # batch forward (traing)
            net_batch, dag_batch = state

            for i, gcn in enumerate(self.net_gcn_layers):
                net_embedding = F.relu(gcn(net_batch.x, net_batch.edge_index, edge_weight=net_batch.edge_attr)) if i == 0 else F.relu(gcn(net_embedding, net_batch.edge_index, edge_weight=net_batch.edge_attr))

            for i, gcn in enumerate(self.dag_gcn_layers):
                dag_embedding = F.relu(gcn(dag_batch.x, dag_batch.edge_index, edge_weight=dag_batch.edge_attr)) if i == 0 else F.relu(gcn(dag_embedding, dag_batch.edge_index, edge_weight=dag_batch.edge_attr))

            if self.normalize == True:
                net_embedding = torch.nn.functional.normalize(net_embedding, dim=-1)
                dag_embedding = torch.nn.functional.normalize(dag_embedding, dim=-1)

            net_embedding = net_embedding.reshape(batch_size, -1, self.net_embedding_dim)
            dag_embedding = dag_embedding.reshape(batch_size, -1, self.dag_embedding_dim)

            net_node_num = net_embedding.size(1)
            dag_node_num = dag_embedding.size(1)
            alpha = torch.ones(batch_size, net_node_num, self.dag_nodes_num).to(net_embedding.device)

            hyb_embedding_net = net_embedding + torch.matmul(alpha, dag_embedding) / dag_node_num
            hyb_embedding_dag = dag_embedding + torch.matmul(alpha.transpose(1, 2), net_embedding) / net_node_num

            hyb_embedding = (hyb_embedding_net.mean(dim=1) + hyb_embedding_dag.mean(dim=1)) / 2

            out = self.fusion_model(torch.cat([hyb_embedding, time_embedding, action], dim=-1))

            return out

        elif len(state) == 6 and batch_size == 1:  # single forward (interpaly with env)
            net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights = state

            for i, gcn in enumerate(self.net_gcn_layers):
                net_embedding = F.relu(gcn(net_feature, net_edge_index, edge_weight=net_edge_weights)) if i == 0 else F.relu(gcn(net_embedding, net_edge_index, edge_weight=net_edge_weights))

            for i, gcn in enumerate(self.dag_gcn_layers):
                dag_embedding = F.relu(gcn(dag_feature, dag_edge_index, edge_weight=dag_edge_weights)) if i == 0 else F.relu(gcn(dag_embedding, dag_edge_index, edge_weight=dag_edge_weights))

            if self.normalize == True:
                net_embedding = torch.nn.functional.normalize(net_embedding, dim=-1)
                dag_embedding = torch.nn.functional.normalize(dag_embedding, dim=-1)

            net_node_num = net_embedding.size(0)
            dag_node_num = dag_embedding.size(0)
            alpha = torch.ones(net_node_num, dag_node_num).to(net_embedding.device)

            hyb_embedding_net = net_embedding + torch.matmul(alpha, dag_embedding) / dag_node_num
            hyb_embedding_dag = dag_embedding + torch.matmul(alpha.t(), net_embedding) / net_node_num

            hyb_embedding = (hyb_embedding_net.mean(dim=0) + hyb_embedding_dag.mean(dim=0)) / 2

            out = self.fusion_model(torch.cat([hyb_embedding.unsqueeze(0), time_embedding, action], dim=-1))

            return out


class GCN_Critic(nn.Module):

    def __init__(self, net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, action_dim, max_action_len, normalize=False):
        super(GCN_Critic, self).__init__()

        self.normalize = normalize

        # GCN for network and DAG
        self.net_encoder = GCNConv(net_node_feat_dim, net_embedding_dim)
        self.dag_encoder = GCNConv(dag_node_feat_dim, dag_embedding_dim)

        self.action_encoder = nn.Sequential(nn.Linear(action_dim * max_action_len, net_embedding_dim), nn.Mish(), nn.Linear(net_embedding_dim, net_embedding_dim))  # action的embedding_dim与网络图的embedding_dim一致

        # Concatenation and processing
        self.fusion_model = nn.Sequential(nn.Linear(dag_embedding_dim + net_embedding_dim + net_embedding_dim, net_embedding_dim), nn.ReLU(), nn.Linear(net_embedding_dim, 1))

    def forward(self, net_feat, net_edge_index, net_edge_weights, dag_feat, dag_edge_index, dag_edge_weights, action, net_batch=None, dag_batch=None):

        if isinstance(action, torch.Tensor):
            batch_action = action
        else:
            batch_action = torch.stack(action)
        action_embedding = self.action_encoder(batch_action.squeeze(-1))

        # Apply GCN to network and DAG features
        net_embedding = self.net_encoder(net_feat, net_edge_index, edge_weight=net_edge_weights)
        dag_embedding = self.dag_encoder(dag_feat, dag_edge_index, edge_weight=dag_edge_weights)

        net_embedding = F.relu(net_embedding)
        dag_embedding = F.relu(dag_embedding)

        if self.normalize == True:
            net_embedding = torch.nn.functional.normalize(net_embedding, dim=-1)
            dag_embedding = torch.nn.functional.normalize(dag_embedding, dim=-1)

        if net_batch != None and dag_batch != None:  # batch gru processing when updating

            net_embedding = global_mean_pool(net_embedding, net_batch)  # mean pooling based on batch
            dag_embedding = global_mean_pool(dag_embedding, dag_batch)  # mean pooling based on batch

            combined_embedding = torch.cat((net_embedding, dag_embedding, action_embedding), dim=-1)  # [batch_size,embedding_dim]
            state_vals = self.fusion_model(combined_embedding)
            return state_vals.squeeze(1)
        else:
            # Assuming we concatenate embeddings by their mean or max
            net_embedding = net_embedding.mean(dim=0)
            dag_embedding = dag_embedding.mean(dim=0)

            # Concatenate and apply fully connected layers
            combined_embedding = torch.cat((net_embedding, dag_embedding, action_embedding), dim=-1)
            state_val = self.fusion_model(combined_embedding)

            return state_val


class GCN_Diffusion(nn.Module):
    def __init__(
        self,
        net_node_feat_dim,
        dag_node_feat_dim,
        dag_embedding_dim,
        net_embedding_dim,
        num_net_gcn_layers,
        num_dag_gcn_layers,
        action_len,
        action_dim,
        normalize,
        noise_ratio,
        beta_schedule="vp",
        n_timesteps=1000,
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
    ):
        super(GCN_Diffusion, self).__init__()

        self.action_dim = action_dim
        self.action_len = action_len
        self.predict_model = GCN_Predict_Model(
            net_node_feat_dim,
            dag_node_feat_dim,
            dag_embedding_dim,
            net_embedding_dim,
            num_net_gcn_layers,
            num_dag_gcn_layers,
            action_len,
            action_dim,
            normalize,
        )

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

    def p_mean_variance(self, x, t, s, batch_size):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.predict_model(x, t, s, batch_size))

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s, bs):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, batch_size=bs)

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
            x = self.p_sample(x, timesteps, state, batch_size)

        return x

    @torch.no_grad()
    def sample(self, state, batch_size, eval=False):
        self.noise_ratio = 0 if eval else self.max_noise_ratio

        shape = (batch_size, self.action_len * self.action_dim)
        action = self.p_sample_loop(state, shape)
        return action.clamp_(-1.0, 1.0)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        return sample

    def p_losses(self, x_start, state, t, weights, batch_size):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.predict_model(x_noisy, t, state, batch_size)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, best_action, state, weights=1.0):
        batch_size = len(best_action)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=best_action.device).long()
        return self.p_losses(best_action, state, t, weights, batch_size)

    def forward(self, state, batch_size, eval=False):
        return self.sample(state, batch_size, eval)


class HADES_Agent(object):
    def __init__(
        self,
        args,
        net_node_feat_dim,
        dag_node_feat_dim,
        dag_embedding_dim,
        net_embedding_dim,
        num_net_gcn_layers,
        num_dag_gcn_layers,
        action_dim,
        action_len,
        device,
        writer,
    ):

        self.actor = GCN_Diffusion(
            net_node_feat_dim=net_node_feat_dim,
            dag_node_feat_dim=dag_node_feat_dim,
            dag_embedding_dim=dag_embedding_dim,
            net_embedding_dim=net_embedding_dim,
            num_net_gcn_layers=num_net_gcn_layers,
            num_dag_gcn_layers=num_dag_gcn_layers,
            action_dim=action_dim,
            action_len=action_len,
            normalize=args.normalize,
            noise_ratio=args.noise_ratio,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.diffusion_steps,
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.memory = list_ReplayMemory(args.memory_size, device)
        self.diffusion_memory = DiffusionMemory(args.memory_size, device)

        self.action_update_epochs = args.action_update_epochs

        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm
        self.cri_grad_norm = args.cri_grad_norm

        self.step = 0
        self.tau = args.tau
        self.gamma = args.gamma
        self.actor_target = copy.deepcopy(self.actor)
        self.update_actor_target_every = args.update_actor_target_every

        self.critic = GCN_Critic(net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, action_dim, action_len, normalize=False).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_len = action_len
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

    def append_memory(self, state, action, reward, next_state):
        action = (action - self.action_bias) / self.action_scale

        self.memory.append(state, action, reward, next_state)
        self.diffusion_memory.append(state, action)

    # @torch.inference_mode()  # NOTE: runtime optimization
    def sample_action(self, net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights, steps, eval, writer):

        if steps < self.threshod and eval == False:
            action = torch.rand(self.action_dim * self.action_len).to(self.device)
            action = action.reshape(self.action_len, self.action_dim)

        else:
            state = (net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights)
            action = self.actor(state, batch_size=1, eval=eval).squeeze(0).unsqueeze(-1)
            action = action * self.action_scale + self.action_bias

        # if eval == False and steps % 500 == 0:
        #     for i in range(self.action_dim):
        #         writer.add_scalar(f"action/action{i+1}", action[i], global_step=steps)

        return action

    def action_gradient(self, batch_size, log_writer):  # 采用Q loss更新动作，然后替换动作
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)

        state_net_list = [
            Data(
                x=state["net_feature"],
                edge_index=state["net_edge_index"],
                edge_attr=state["net_edge_weights"],
            )
            for state in states
        ]
        state_dag_list = [
            Data(
                x=state["dag_feature"],
                edge_index=state["dag_edge_index"],
                edge_attr=state["dag_edge_weights"],
            )
            for state in states
        ]
        net_batch = Batch.from_data_list(state_net_list).to(self.device)
        dag_batch = Batch.from_data_list(state_dag_list).to(self.device)

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for _ in range(self.action_update_epochs):
            best_actions.requires_grad_(True)
            q = self.critic(net_batch.x, net_batch.edge_index, net_batch.edge_attr, dag_batch.x, dag_batch.edge_index, dag_batch.edge_attr, best_actions, net_batch.batch, dag_batch.batch)
            loss = -q

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1.0, 1.0)

        if self.step % 10 == 0:
            log_writer.add_scalar("algorithm/Action_Grad_Norm", actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        self.diffusion_memory.replace(idxs, best_actions)

        return states, best_actions

    def train(self, iterations, batch_size, log_writer):
        for _ in range(iterations):
            states, actions, rewards, next_states = self.memory.sample(batch_size)

            state_net_list = [
                Data(
                    x=state["net_feature"],
                    edge_index=state["net_edge_index"],
                    edge_attr=state["net_edge_weights"],
                )
                for state in states
            ]
            state_dag_list = [
                Data(
                    x=state["dag_feature"],
                    edge_index=state["dag_edge_index"],
                    edge_attr=state["dag_edge_weights"],
                )
                for state in states
            ]
            net_batch = Batch.from_data_list(state_net_list).to(self.device)
            dag_batch = Batch.from_data_list(state_dag_list).to(self.device)
            """ Q Training """
            current_q = self.critic(net_batch.x, net_batch.edge_index, net_batch.edge_attr, dag_batch.x, dag_batch.edge_index, dag_batch.edge_attr, actions, net_batch.batch, dag_batch.batch)

            next_state_net_list = [
                Data(
                    x=state["net_feature"],
                    edge_index=state["net_edge_index"],
                    edge_attr=state["net_edge_weights"],
                )
                for state in next_states
            ]
            next_state_dag_list = [
                Data(
                    x=state["dag_feature"],
                    edge_index=state["dag_edge_index"],
                    edge_attr=state["dag_edge_weights"],
                )
                for state in next_states
            ]
            next_net_batch = Batch.from_data_list(next_state_net_list).to(self.device)
            next_dag_batch = Batch.from_data_list(next_state_dag_list).to(self.device)

            state_dir = (next_net_batch, next_dag_batch)
            next_actions = self.actor_target(state_dir, batch_size=batch_size, eval=False)

            target_q = self.critic_target(
                next_net_batch.x, next_net_batch.edge_index, next_net_batch.edge_attr, next_dag_batch.x, next_dag_batch.edge_index, next_dag_batch.edge_attr, next_actions, next_net_batch.batch, next_dag_batch.batch
            )

            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            target_q = (rewards + self.gamma * target_q).detach()

            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.cri_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.cri_grad_norm, norm_type=2)
                if self.step % 10 == 0:
                    log_writer.add_scalar("algorithm/Critic_Grad_Norm", critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()

            """ Policy Training """
            """ 1. Actions Update """
            states, best_actions = self.action_gradient(batch_size, log_writer)

            """ 2. Policy Network Update """
            state_net_list = [
                Data(
                    x=state["net_feature"],
                    edge_index=state["net_edge_index"],
                    edge_attr=state["net_edge_weights"],
                )
                for state in states
            ]
            state_dag_list = [
                Data(
                    x=state["dag_feature"],
                    edge_index=state["dag_edge_index"],
                    edge_attr=state["dag_edge_weights"],
                )
                for state in states
            ]
            net_batch = Batch.from_data_list(state_net_list).to(self.device)
            dag_batch = Batch.from_data_list(state_dag_list).to(self.device)

            state_dir = (net_batch, dag_batch)
            actor_loss = self.actor.loss(best_actions.squeeze(-1), state_dir)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.ac_grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                if self.step % 10 == 0:
                    log_writer.add_scalar("algorithm/Actor_Grad_Norm", actor_grad_norms.max().item(), self.step)
            self.actor_optimizer.step()

            """ Step Target network """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.step % self.update_actor_target_every == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

    def save_model(self, dir, remark=None):
        if remark is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{remark}.pt")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pt")

    def load_model(self, dir, remark=None):
        if remark is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{remark}.pt"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pt"))
