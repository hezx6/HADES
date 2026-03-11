import numpy as np
import torch
import sys
import random
import time

from Algorithm.DIPO.DiPo import DiPo
from Algorithm.DIPO.DiPo_GCN import DiPo_GCN
from Algorithm.DIPO.DiPo_MLP import DiPo_MLP_Agent
from Algorithm.DIPO.replay_memory import ReplayMemory, DiffusionMemory, list_ReplayMemory
from Algorithm.GCN_PPO import GCN_PPO
from Algorithm.TD3 import TD3_Agent
from Algorithm.SAC import SAC_Agent
from Algorithm.SDAC.SDAC import SDAC_Agent
from Algorithm.QVPO.QVPO import QVPO_Agent
from Algorithm.DDPG import DDPG_Agent
import copy
import math
from Dataset_DAG_task import DatasetDAGTaskBuilder

from Env import Env, calculate_VaR_CVaR
from tensorboardX import SummaryWriter
import os
import constants as cn
import logging

from colorama import init, Fore, Style

from setup import setup, readParser


def main(args, config):
    writer = config["writer"]
    data_save_dir = config["data_save_dir"]
    model_save_dir = config["model_save_dir"]
    figure_save_dir = config["figure_save_dir"]
    device = config["device"]
    logger = config["logger"]

    #  # Initial environment
    slot_length = args.slot_length
    state_dim = args.es_num * (cn.resource_type + 4) + args.dag_node_num * (1 + 4 + args.es_num)

    action_dim = 1
    action_len = args.dag_node_num

    net_node_feat_dim = cn.resource_type + 4
    dag_node_feat_dim = 1 + 4 + args.es_num
    dag_embedding_dim = 1024  # !!!
    net_embedding_dim = 1024  #!!!这三个保持一致
    head_in_dim = 1024  #!!!
    num_net_gcn_layers = args.gcn_layer_num
    num_dag_gcn_layers = args.gcn_layer_num

    max_ep_len = args.max_ep_len
    batch_size = args.batch_size
    algorithm = args.algorithm

    dataset_file = args.dag_dataset_file
    if dataset_file == "":
        if args.dag_source == "huawei":
            dataset_file = os.path.join(cn.ROOT_DIR, "dataset", "Huawei-Network-AI-Challenge", "task_table.csv")
        elif args.dag_source == "cluster":
            dataset_file = os.path.join(cn.ROOT_DIR, "dataset", "cluster-trace-v2018", "batch_task.csv")

    dag_builder = DatasetDAGTaskBuilder(seed=args.seed)
    DAG_tasks_set = dag_builder.build(
        dataset_name=args.dag_source,
        file_path=dataset_file,
        dag_num=args.dag_num,
        target_total_nodes=args.dag_node_num,
        shuffle=args.dag_dataset_shuffle,
    )

    if len(DAG_tasks_set) < args.dag_num:
        raise ValueError("Not enough DAGs were built from dataset. " f"source={args.dag_source}, file={dataset_file}, expected={args.dag_num}, got={len(DAG_tasks_set)}, dag_node_num={args.dag_node_num}")

    env = Env(args, DAG_tasks_set)

    if args.mode == "train":

        rewards = ["" for _ in range(args.num_episode)]
        success_rate = ["" for _ in range(args.num_episode)]
        avg_trans_delay = ["" for _ in range(args.num_episode)]
        avg_queue_delay = ["" for _ in range(args.num_episode)]
        avg_comp_delay = ["" for _ in range(args.num_episode)]
        avg_total_delay = ["" for _ in range(args.num_episode)]
        avg_energy = ["" for _ in range(args.num_episode)]
        item = ["reward", "avg_total_delay", "avg_energy", "success_rate", "avg_trans_delay", "avg_queue_delay", "avg_comp_delay"]
        item_lists = [rewards, avg_total_delay, avg_energy, success_rate, avg_trans_delay, avg_queue_delay, avg_comp_delay]

        if algorithm == "HADES":

            agent = DiPo_GCN(args, net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, num_net_gcn_layers, num_dag_gcn_layers, action_dim, action_len, device, writer)

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                next_state = None
                current_state = {
                    "net_edge_index": None,
                    "net_edge_weights": None,
                    "net_feature": None,
                    "dag_edge_index": None,
                    "dag_edge_weights": None,
                    "dag_feature": None,
                }
                ue = random.choice(env.UEs)
                new_task = ue.generate_dag_task(t, env)
                while t <= max_ep_len * slot_length:

                    if next_state == None:
                        current_state["net_edge_index"], current_state["net_edge_weights"], current_state["net_feature"] = env.edge_network.get_state(device)
                        current_state["dag_edge_index"], current_state["dag_edge_weights"], current_state["dag_feature"] = env.get_dag_task_status(new_task, device)
                    else:
                        current_state = next_state

                    action = agent.sample_action(
                        current_state["net_feature"],
                        current_state["net_edge_index"],
                        current_state["net_edge_weights"],
                        current_state["dag_feature"],
                        current_state["dag_edge_index"],
                        current_state["dag_edge_weights"],
                        steps,
                        eval=False,
                        writer=writer,
                    )

                    env.step(t, episode, new_task, action, algorithm)

                    next_state = {
                        "net_edge_index": None,
                        "net_edge_weights": None,
                        "net_feature": None,
                        "dag_edge_index": None,
                        "dag_edge_weights": None,
                        "dag_feature": None,
                    }
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(round((t + slot_length) * 100) / 100, env)

                    next_state["net_edge_index"], next_state["net_edge_weights"], next_state["net_feature"] = env.edge_network.get_state(device)
                    next_state["dag_edge_index"], next_state["dag_edge_weights"], next_state["dag_feature"] = env.get_dag_task_status(new_task, device)

                    agent.append_memory(current_state, action, reward.get_value(), next_state)

                    if steps % int(max_ep_len / 10) == 0:
                        writer.add_scalar("reward/delay_reward", reward.delay_reward, global_step=steps)
                        # writer.add_scalar("reward/queue_reward", reward.queue_penalty, global_step=steps)
                        writer.add_scalar("reward/load_reward", reward.load_penalty, global_step=steps)
                        writer.add_scalar("reward/energy_reward", reward.energy_reward, global_step=steps)
                        writer.add_scalar("reward/SortinoRatio", reward.current_SortinoRatio, global_step=steps)
                        writer.add_scalar("reward/SharpeRatio", reward.current_SharpeRatio, global_step=steps)
                        writer.add_scalar("reward/average_delay", reward.current_average_delay, global_step=steps)
                        writer.add_scalar("reward/variance", reward.variance, global_step=steps)
                        writer.add_scalar("reward/down_variance", reward.down_variance, global_step=steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    if steps >= args.threshod and steps % args.train_interval == 0:
                        agent.train(args.epochs_num, batch_size=batch_size, log_writer=writer)

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()
                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

        elif algorithm == "SDAC":

            agent = SDAC_Agent(
                state_dim=state_dim,
                action_dim=action_len,
                args=args,
                device=device,
                writer=writer,
            )

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False
                next_state = None
                current_state = None
                ue = random.choice(env.UEs)
                new_task = ue.generate_dag_task(t, env)

                while t <= max_ep_len * slot_length:

                    if next_state == None:
                        _, _, net_feature = env.edge_network.get_state(device)
                        _, _, dag_feature = env.get_dag_task_status(new_task, device)
                        net_feature_flatten = net_feature.view(-1)
                        dag_feature_flatten = dag_feature.view(-1)
                        current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0).unsqueeze(dim=0)
                    else:
                        current_state = next_state

                    action = agent.select_action(current_state)

                    env.step(t, episode, new_task, action, algorithm)

                    next_state = None
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(round((t + slot_length) * 100) / 100, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    next_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0).unsqueeze(dim=0)

                    agent.replay_buffer.append(current_state, action.squeeze(dim=-1), reward.get_value(), next_state)

                    if steps % int(max_ep_len / 10) == 0:
                        writer.add_scalar("reward/delay_reward", reward.delay_reward, global_step=steps)
                        # writer.add_scalar("reward/queue_reward", reward.queue_penalty, global_step=steps)
                        writer.add_scalar("reward/load_reward", reward.load_penalty, global_step=steps)
                        writer.add_scalar("reward/energy_reward", reward.energy_reward, global_step=steps)
                        writer.add_scalar("reward/SortinoRatio", reward.current_SortinoRatio, global_step=steps)
                        writer.add_scalar("reward/SharpeRatio", reward.current_SharpeRatio, global_step=steps)
                        writer.add_scalar("reward/average_delay", reward.current_average_delay, global_step=steps)
                        writer.add_scalar("reward/variance", reward.variance, global_step=steps)
                        writer.add_scalar("reward/down_variance", reward.down_variance, global_step=steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    if steps > max_ep_len and steps % args.policy_update_delay == 0:
                        agent.train(batch_size=batch_size, log_writer=writer)

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

        elif algorithm == "QVPO":

            agent = QVPO_Agent(
                state_dim=state_dim,
                action_dim=action_len,
                args=args,
                device=device,
                writer=writer,
            )

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False
                next_state = None
                current_state = None
                ue = random.choice(env.UEs)
                new_task = ue.generate_dag_task(t, env)

                while t <= max_ep_len * slot_length:

                    if next_state == None:
                        _, _, net_feature = env.edge_network.get_state(device)
                        _, _, dag_feature = env.get_dag_task_status(new_task, device)
                        net_feature_flatten = net_feature.view(-1)
                        dag_feature_flatten = dag_feature.view(-1)
                        current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0).unsqueeze(dim=0)
                    else:
                        current_state = next_state

                    action = agent.select_action(current_state)

                    env.step(t, episode, new_task, action, algorithm)

                    next_state = None
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(round((t + slot_length) * 100) / 100, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    next_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0).unsqueeze(dim=0)

                    agent.replay_buffer.append(current_state, action.squeeze(dim=-1), reward.get_value(), next_state)

                    if steps % int(max_ep_len / 10) == 0:
                        writer.add_scalar("reward/delay_reward", reward.delay_reward, global_step=steps)
                        # writer.add_scalar("reward/queue_reward", reward.queue_penalty, global_step=steps)
                        writer.add_scalar("reward/load_reward", reward.load_penalty, global_step=steps)
                        writer.add_scalar("reward/energy_reward", reward.energy_reward, global_step=steps)
                        writer.add_scalar("reward/SortinoRatio", reward.current_SortinoRatio, global_step=steps)
                        writer.add_scalar("reward/SharpeRatio", reward.current_SharpeRatio, global_step=steps)
                        writer.add_scalar("reward/average_delay", reward.current_average_delay, global_step=steps)
                        writer.add_scalar("reward/variance", reward.variance, global_step=steps)
                        writer.add_scalar("reward/down_variance", reward.down_variance, global_step=steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    if steps > args.threshod and steps % args.policy_update_delay == 0:
                        agent.train(batch_size)

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

        elif algorithm == "ACDE":

            coef_entropy = args.coef_entropy

            agent = GCN_PPO(
                net_node_feat_dim=net_node_feat_dim,
                dag_node_feat_dim=dag_node_feat_dim,
                dag_embedding_dim=dag_embedding_dim,
                net_embedding_dim=net_embedding_dim,
                num_net_gcn_layers=num_net_gcn_layers,
                num_dag_gcn_layers=num_dag_gcn_layers,
                head_in_dim=head_in_dim,
                action_dim=action_dim,
                action_len=action_len,
                lr_actor=args.ppo_lr_actor,
                lr_critic=args.ppo_lr_critic,
                gamma=args.gamma,
                epochs_num=args.epochs_num,
                batch_size=args.batch_size,
                eps_clip=args.eps_clip,
                coef_entropy=coef_entropy,
                action_std_init=args.action_std_init,
                is_attention=args.attention,
                normalize=args.normalize,
                device=device,
                writer=writer,
            )

            update_timestep = max_ep_len * args.update_coef
            action_std_decay_freq = math.floor(args.num_episode / (math.ceil((args.action_std_init - args.min_action_std) / args.action_std_decay_rate) + 1))

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False

                while t <= max_ep_len * slot_length:

                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    net_edge_index, net_edge_weights, net_feature = env.edge_network.get_state(device)
                    dag_edge_index, dag_edge_weights, dag_feature = env.get_dag_task_status(new_task, device)

                    action, action_logprob, state_val = agent.select_action(net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights)

                    env.step(t, episode, new_task, action, algorithm)

                    state = {
                        "net_edge_index": net_edge_index.to(device),
                        "net_edge_weights": net_edge_weights.to(device),
                        "net_feature": net_feature.to(device),
                        "dag_edge_index": dag_edge_index.to(device),
                        "dag_edge_weights": dag_edge_weights.to(device),
                        "dag_feature": dag_feature.to(device),
                    }
                    agent.append_memory(state, action, reward.get_value(), done, action_logprob, state_val)

                    if steps % int(max_ep_len / 10) == 0:
                        writer.add_scalar("reward/delay_reward", reward.delay_reward, global_step=steps)
                        # writer.add_scalar("reward/queue_reward", reward.queue_penalty, global_step=steps)
                        writer.add_scalar("reward/load_reward", reward.load_penalty, global_step=steps)
                        writer.add_scalar("reward/energy_reward", reward.energy_reward, global_step=steps)
                        writer.add_scalar("reward/SortinoRatio", reward.current_SortinoRatio, global_step=steps)
                        writer.add_scalar("reward/SharpeRatio", reward.current_SharpeRatio, global_step=steps)
                        writer.add_scalar("reward/average_delay", reward.current_average_delay, global_step=steps)
                        writer.add_scalar("reward/variance", reward.variance, global_step=steps)
                        writer.add_scalar("reward/down_variance", reward.down_variance, global_step=steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    # update PPO agent
                    if steps % update_timestep == 0:
                        agent.update()

                    if episode % action_std_decay_freq == 0:
                        agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

        elif algorithm == "MESON":
            action_std_decay_freq = math.floor(args.num_episode / (math.ceil((args.action_std_init - args.min_action_std) / args.action_std_decay_rate) + 1))

            agent = DDPG_Agent(
                state_dim=state_dim,
                action_dim=action_len,
                gamma=args.gamma,
                tau=args.tau,
                actor_lr=args.ddpg_actor_lr,
                critic_lr=args.ddpg_critic_lr,
                init_noise_std=args.ddpg_init_noise_std,
                device=device,
            )

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False
                next_state = None
                current_state = None
                ue = random.choice(env.UEs)
                new_task = ue.generate_dag_task(t, env)

                while t <= max_ep_len * slot_length:

                    if next_state == None:
                        _, _, net_feature = env.edge_network.get_state(device)
                        _, _, dag_feature = env.get_dag_task_status(new_task, device)
                        net_feature_flatten = net_feature.view(-1)
                        dag_feature_flatten = dag_feature.view(-1)
                        current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0)
                    else:
                        current_state = next_state

                    action = agent.select_action(current_state)
                    action = action.unsqueeze(dim=-1)
                    env.step(t, episode, new_task, action, algorithm)

                    next_state = None
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(round((t + slot_length) * 100) / 100, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    next_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0)

                    agent.replay_buffer.append((current_state, action.squeeze(dim=-1), reward.get_value(), next_state, int(done)))

                    if steps % int(max_ep_len / 10) == 0:
                        writer.add_scalar("reward/delay_reward", reward.delay_reward, global_step=steps)
                        # writer.add_scalar("reward/queue_reward", reward.queue_penalty, global_step=steps)
                        writer.add_scalar("reward/load_reward", reward.load_penalty, global_step=steps)
                        writer.add_scalar("reward/energy_reward", reward.energy_reward, global_step=steps)
                        writer.add_scalar("reward/SortinoRatio", reward.current_SortinoRatio, global_step=steps)
                        writer.add_scalar("reward/SharpeRatio", reward.current_SharpeRatio, global_step=steps)
                        writer.add_scalar("reward/average_delay", reward.current_average_delay, global_step=steps)
                        writer.add_scalar("reward/variance", reward.variance, global_step=steps)
                        writer.add_scalar("reward/down_variance", reward.down_variance, global_step=steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    if len(agent.replay_buffer) > max_ep_len:
                        agent.train(256)

                    if episode % action_std_decay_freq == 0:
                        agent.decay_action_std(args.ddpg_action_std_decay_rate, args.ddpg_min_action_std)

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

        elif algorithm == "BSAC":

            agent = SAC_Agent(
                state_dim=state_dim,
                action_dim=action_len,
                gamma=args.gamma,
                tau=args.tau,
                alpha=args.sac_alpha,
                actor_lr=args.sac_actor_lr,
                critic_lr=args.sac_critic_lr,
                alpha_lr=args.sac_alpha_lr,
                target_entropy=args.sac_target_entropy,
                device=device,
                writer=writer,
            )

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False
                next_state = None
                current_state = None
                ue = random.choice(env.UEs)
                new_task = ue.generate_dag_task(t, env)

                while t <= max_ep_len * slot_length:

                    if next_state == None:
                        _, _, net_feature = env.edge_network.get_state(device)
                        _, _, dag_feature = env.get_dag_task_status(new_task, device)
                        net_feature_flatten = net_feature.view(-1)
                        dag_feature_flatten = dag_feature.view(-1)
                        current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0)
                    else:
                        current_state = next_state

                    action = agent.select_action(current_state)
                    action = action.squeeze(dim=0).unsqueeze(dim=-1)
                    scaled_action = (action + 1) / 2
                    env.step(t, episode, new_task, scaled_action)

                    next_state = None
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(round((t + slot_length) * 100) / 100, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    next_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0)

                    agent.replay_buffer.append((current_state, action.squeeze(dim=-1), reward.get_value(), next_state, int(done)))

                    if steps % int(max_ep_len / 10) == 0:
                        writer.add_scalar("reward/delay_reward", reward.delay_reward, global_step=steps)
                        # writer.add_scalar("reward/queue_reward", reward.queue_penalty, global_step=steps)
                        writer.add_scalar("reward/load_reward", reward.load_penalty, global_step=steps)
                        writer.add_scalar("reward/energy_reward", reward.energy_reward, global_step=steps)
                        writer.add_scalar("reward/SortinoRatio", reward.current_SortinoRatio, global_step=steps)
                        writer.add_scalar("reward/SharpeRatio", reward.current_SharpeRatio, global_step=steps)
                        writer.add_scalar("reward/average_delay", reward.current_average_delay, global_step=steps)
                        writer.add_scalar("reward/variance", reward.variance, global_step=steps)
                        writer.add_scalar("reward/down_variance", reward.down_variance, global_step=steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    if len(agent.replay_buffer) > max_ep_len:
                        agent.train(256)

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

        elif algorithm == "Random":

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False

                while t <= max_ep_len * slot_length:

                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    action = torch.rand(action_len, 1)

                    env.step(t, episode, new_task, action, algorithm)

                    if steps % int(max_ep_len / 10) == 0:
                        writer.add_scalar("reward/delay_reward", reward.delay_reward, global_step=steps)
                        # writer.add_scalar("reward/queue_reward", reward.queue_penalty, global_step=steps)
                        writer.add_scalar("reward/load_reward", reward.load_penalty, global_step=steps)
                        writer.add_scalar("reward/energy_reward", reward.energy_reward, global_step=steps)
                        writer.add_scalar("reward/SortinoRatio", reward.current_SortinoRatio, global_step=steps)
                        writer.add_scalar("reward/SharpeRatio", reward.current_SharpeRatio, global_step=steps)
                        writer.add_scalar("reward/average_delay", reward.current_average_delay, global_step=steps)
                        writer.add_scalar("reward/variance", reward.variance, global_step=steps)
                        writer.add_scalar("reward/down_variance", reward.down_variance, global_step=steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

        else:
            raise NotImplementedError

    if args.mode == "eval":

        rewards = ["" for _ in range(args.num_episode)]
        success_rate = ["" for _ in range(args.num_episode)]
        avg_trans_delay = ["" for _ in range(args.num_episode)]
        avg_queue_delay = ["" for _ in range(args.num_episode)]
        avg_comp_delay = ["" for _ in range(args.num_episode)]
        avg_total_delay = ["" for _ in range(args.num_episode)]
        avg_energy = ["" for _ in range(args.num_episode)]
        item = ["reward", "avg_total_delay", "avg_energy", "success_rate", "avg_trans_delay", "avg_queue_delay", "avg_comp_delay"]
        item_lists = [rewards, avg_total_delay, avg_energy, success_rate, avg_trans_delay, avg_queue_delay, avg_comp_delay]

        episode_item = [
            "total_delay_record",
            "energy_record",
        ]
        episode_total_delay_each_step = []
        episode_energy_each_step = []

        if algorithm == "HADES":

            agent = DiPo_GCN(args, net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, num_net_gcn_layers, num_dag_gcn_layers, action_dim, action_len, device, writer)
            agent.load_model(dir=model_save_dir, remark=args.remark)

            steps = 0
            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                current_state = {
                    "net_edge_index": None,
                    "net_edge_weights": None,
                    "net_feature": None,
                    "dag_edge_index": None,
                    "dag_edge_weights": None,
                    "dag_feature": None,
                }

                while t <= max_ep_len * slot_length:

                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)
                    current_state["net_edge_index"], current_state["net_edge_weights"], current_state["net_feature"] = env.edge_network.get_state(device)
                    current_state["dag_edge_index"], current_state["dag_edge_weights"], current_state["dag_feature"] = env.get_dag_task_status(new_task, device)

                    start = time.perf_counter()
                    action = agent.sample_action(
                        current_state["net_feature"],
                        current_state["net_edge_index"],
                        current_state["net_edge_weights"],
                        current_state["dag_feature"],
                        current_state["dag_edge_index"],
                        current_state["dag_edge_weights"],
                        steps,
                        eval=True,
                        writer=writer,
                    )
                    end = time.perf_counter()
                    writer.add_scalar(f"eval/inference_latency", end - start, global_step=steps)

                    env.step(t, episode, new_task, action, algorithm)

                    writer.add_scalar(f"eval/episode_{episode}/reward/delay_reward", reward.delay_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/load_reward", reward.load_penalty, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/energy_reward", reward.energy_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SortinoRatio", reward.current_SortinoRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SharpeRatio", reward.current_SharpeRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/average_delay", reward.current_average_delay, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/variance", reward.variance, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/down_variance", reward.down_variance, global_step=episode_steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    reward.reset()
                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("eval/metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("eval/metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("eval/metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("eval/metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("eval/metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("eval/metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

                mode = "overwrite" if episode == 1 else "overwrite_last"

                episode_total_delay_each_step = env.episode_delay_record
                episode_energy_each_step = env.episode_energy_record
                episode_each_step_records = [episode_total_delay_each_step, episode_energy_each_step]

        elif algorithm == "SDAC":

            agent = SDAC_Agent(state_dim=state_dim, action_dim=action_len, args=args, device=device, writer=writer)

            agent.load_model(dir=model_save_dir, remark=args.remark)

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False

                while t <= max_ep_len * slot_length:
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0).unsqueeze(0)

                    start = time.perf_counter()
                    action = agent.select_action(current_state)
                    end = time.perf_counter()
                    writer.add_scalar(f"eval/inference_latency", end - start, global_step=steps)
                    action = action.unsqueeze(dim=-1)

                    env.step(t, episode, new_task, action, algorithm)

                    writer.add_scalar(f"eval/episode_{episode}/reward/delay_reward", reward.delay_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/load_reward", reward.load_penalty, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/energy_reward", reward.energy_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SortinoRatio", reward.current_SortinoRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SharpeRatio", reward.current_SharpeRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/average_delay", reward.current_average_delay, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/variance", reward.variance, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/down_variance", reward.down_variance, global_step=episode_steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("eval/metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("eval/metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("eval/metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("eval/metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("eval/metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("eval/metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

                mode = "overwrite" if episode == 1 else "overwrite_last"

                episode_total_delay_each_step = env.episode_delay_record
                episode_energy_each_step = env.episode_energy_record
                episode_each_step_records = [episode_total_delay_each_step, episode_energy_each_step]

        elif algorithm == "QVPO":

            agent = QVPO_Agent(state_dim=state_dim, action_dim=action_len, args=args, device=device, writer=writer)

            agent.load_model(dir=model_save_dir, remark=args.remark)

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False

                while t <= max_ep_len * slot_length:
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0).unsqueeze(0)

                    start = time.perf_counter()
                    action = agent.select_action(current_state)
                    end = time.perf_counter()
                    writer.add_scalar(f"eval/inference_latency", end - start, global_step=steps)
                    action = action.unsqueeze(dim=-1)

                    env.step(t, episode, new_task, action, algorithm)

                    writer.add_scalar(f"eval/episode_{episode}/reward/delay_reward", reward.delay_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/load_reward", reward.load_penalty, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/energy_reward", reward.energy_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SortinoRatio", reward.current_SortinoRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SharpeRatio", reward.current_SharpeRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/average_delay", reward.current_average_delay, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/variance", reward.variance, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/down_variance", reward.down_variance, global_step=episode_steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("eval/metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("eval/metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("eval/metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("eval/metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("eval/metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("eval/metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

                mode = "overwrite" if episode == 1 else "overwrite_last"

                episode_total_delay_each_step = env.episode_delay_record
                episode_energy_each_step = env.episode_energy_record
                episode_each_step_records = [episode_total_delay_each_step, episode_energy_each_step]

        elif algorithm == "ACDE":

            coef_entropy = args.coef_entropy

            agent = GCN_PPO(
                net_node_feat_dim=net_node_feat_dim,
                dag_node_feat_dim=dag_node_feat_dim,
                dag_embedding_dim=dag_embedding_dim,
                net_embedding_dim=net_embedding_dim,
                num_net_gcn_layers=num_net_gcn_layers,
                num_dag_gcn_layers=num_dag_gcn_layers,
                head_in_dim=head_in_dim,
                action_dim=action_dim,
                action_len=action_len,
                lr_actor=args.ppo_lr_actor,
                lr_critic=args.ppo_lr_critic,
                gamma=args.gamma,
                epochs_num=args.epochs_num,
                batch_size=args.batch_size,
                eps_clip=args.eps_clip,
                coef_entropy=coef_entropy,
                action_std_init=args.min_action_std,
                is_attention=args.attention,
                normalize=args.normalize,
                device=device,
                writer=writer,
            )

            agent.load_model(dir=model_save_dir, remark=args.remark)

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False

                while t <= max_ep_len * slot_length:

                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    net_edge_index, net_edge_weights, net_feature = env.edge_network.get_state(device)
                    dag_edge_index, dag_edge_weights, dag_feature = env.get_dag_task_status(new_task, device)

                    start = time.perf_counter()
                    action, action_logprob, state_val = agent.select_action(net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights)
                    end = time.perf_counter()
                    writer.add_scalar(f"eval/inference_latency", end - start, global_step=steps)

                    env.step(t, episode, new_task, action, algorithm)

                    writer.add_scalar(f"eval/episode_{episode}/reward/delay_reward", reward.delay_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/load_reward", reward.load_penalty, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/energy_reward", reward.energy_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SortinoRatio", reward.current_SortinoRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SharpeRatio", reward.current_SharpeRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/average_delay", reward.current_average_delay, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/variance", reward.variance, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/down_variance", reward.down_variance, global_step=episode_steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("eval/metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("eval/metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("eval/metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("eval/metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("eval/metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("eval/metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

                mode = "overwrite" if episode == 1 else "overwrite_last"

                episode_total_delay_each_step = env.episode_delay_record
                episode_energy_each_step = env.episode_energy_record
                episode_each_step_records = [episode_total_delay_each_step, episode_energy_each_step]

        elif algorithm == "MESON":

            agent = DDPG_Agent(state_dim=state_dim, action_dim=action_len, gamma=args.gamma, tau=args.tau, actor_lr=args.ddpg_actor_lr, critic_lr=args.ddpg_critic_lr, init_noise_std=args.ddpg_min_action_std, device=device)

            agent.load_model(dir=model_save_dir, remark=args.remark)

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward
                done = False

                while t <= max_ep_len * slot_length:
                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0)

                    start = time.perf_counter()
                    action = agent.select_action(current_state)
                    end = time.perf_counter()
                    writer.add_scalar(f"eval/inference_latency", end - start, global_step=steps)
                    action = action.unsqueeze(dim=-1)

                    env.step(t, episode, new_task, action, algorithm)

                    writer.add_scalar(f"eval/episode_{episode}/reward/delay_reward", reward.delay_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/load_reward", reward.load_penalty, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/energy_reward", reward.energy_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SortinoRatio", reward.current_SortinoRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SharpeRatio", reward.current_SharpeRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/average_delay", reward.current_average_delay, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/variance", reward.variance, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/down_variance", reward.down_variance, global_step=episode_steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("eval/metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("eval/metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("eval/metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("eval/metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("eval/metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("eval/metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

                mode = "overwrite" if episode == 1 else "overwrite_last"

                episode_total_delay_each_step = env.episode_delay_record
                episode_energy_each_step = env.episode_energy_record
                episode_each_step_records = [episode_total_delay_each_step, episode_energy_each_step]

        elif algorithm == "DSAC":

            agent = SAC_Agent(
                state_dim=state_dim,
                action_dim=action_len,
                gamma=args.gamma,
                tau=args.tau,
                alpha=args.sac_alpha,
                actor_lr=args.sac_actor_lr,
                critic_lr=args.sac_critic_lr,
                alpha_lr=args.sac_alpha_lr,
                target_entropy=args.sac_target_entropy,
                device=device,
                writer=writer,
            )

            agent.load_model(dir=model_save_dir, remark=args.remark)

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward

                while t <= max_ep_len * slot_length:

                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    _, _, net_feature = env.edge_network.get_state(device)
                    _, _, dag_feature = env.get_dag_task_status(new_task, device)
                    net_feature_flatten = net_feature.view(-1)
                    dag_feature_flatten = dag_feature.view(-1)
                    current_state = torch.cat([net_feature_flatten, dag_feature_flatten], dim=0)

                    start = time.perf_counter()
                    action = agent.select_action(current_state)
                    end = time.perf_counter()
                    writer.add_scalar(f"eval/inference_latency", end - start, global_step=steps)
                    action = action.squeeze(dim=0).unsqueeze(dim=-1)
                    scaled_action = (action + 1) / 2

                    env.step(t, episode, new_task, scaled_action)

                    writer.add_scalar(f"eval/episode_{episode}/reward/delay_reward", reward.delay_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/load_reward", reward.load_penalty, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/energy_reward", reward.energy_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SortinoRatio", reward.current_SortinoRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SharpeRatio", reward.current_SharpeRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/average_delay", reward.current_average_delay, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/variance", reward.variance, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/down_variance", reward.down_variance, global_step=episode_steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    if len(agent.replay_buffer) > max_ep_len:
                        agent.train(256)

                    logging.warning(
                        Style.BRIGHT
                        + Fore.RED
                        + "episode= {}, step= {}, time= {}, step reward= {},episode reward= {},avaerage delay= {}s,success_rate= {}".format(
                            episode, episode_steps, t, reward.get_value(), episode_reward, env.avg_total_delay, env.success_rate
                        )
                        + Style.RESET_ALL
                    )

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("eval/metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("eval/metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("eval/metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("eval/metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("eval/metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("eval/metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

                mode = "overwrite" if episode == 1 else "overwrite_last"

                episode_total_delay_each_step = env.episode_delay_record
                episode_energy_each_step = env.episode_energy_record
                episode_each_step_records = [episode_total_delay_each_step, episode_energy_each_step]

        elif algorithm == "Random":

            steps = 0

            for episode in range(1, args.num_episode + 1):
                episode_reward = 0
                episode_steps = 1
                t = 0
                env.reset()
                reward = env.reward

                while t <= max_ep_len * slot_length:

                    ue = random.choice(env.UEs)
                    new_task = ue.generate_dag_task(t, env)

                    start = time.perf_counter()
                    action = torch.rand(action_len, 1)
                    end = time.perf_counter()
                    writer.add_scalar(f"eval/inference_latency", end - start, global_step=steps)

                    env.step(t, episode, new_task, action, algorithm)

                    writer.add_scalar(f"eval/episode_{episode}/reward/delay_reward", reward.delay_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/load_reward", reward.load_penalty, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/energy_reward", reward.energy_reward, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SortinoRatio", reward.current_SortinoRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/SharpeRatio", reward.current_SharpeRatio, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/average_delay", reward.current_average_delay, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/variance", reward.variance, global_step=episode_steps)
                    writer.add_scalar(f"eval/episode_{episode}/reward/down_variance", reward.down_variance, global_step=episode_steps)

                    steps += 1
                    episode_steps += 1
                    episode_reward += reward.get_value()

                    reward.reset()

                    t = round((t + slot_length) * 100) / 100

                rewards[episode - 1] = episode_reward
                avg_trans_delay[episode - 1] = env.avg_trans_delay
                avg_queue_delay[episode - 1] = env.avg_wait_delay
                avg_comp_delay[episode - 1] = env.avg_comp_delay
                avg_total_delay[episode - 1] = env.avg_total_delay
                avg_energy[episode - 1] = env.avg_energy
                success_rate[episode - 1] = env.success_rate
                writer.add_scalar("eval/metrics/reward", episode_reward, global_step=episode)
                writer.add_scalar("eval/metrics/avg_trans_delay", env.avg_trans_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_energy", env.avg_energy, global_step=episode)
                writer.add_scalar("eval/metrics/avg_queue_delay", env.avg_wait_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_comp_delay", env.avg_comp_delay, global_step=episode)
                writer.add_scalar("eval/metrics/avg_total_delay", env.avg_total_delay, global_step=episode)
                writer.add_scalar("eval/metrics/success_rate", env.success_rate, global_step=episode)
                writer.add_scalar("eval/metrics/delay_VaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[0], global_step=episode)
                writer.add_scalar("eval/metrics/delay_CVaR_0.95", calculate_VaR_CVaR(np.array(env.episode_delay_record))[1], global_step=episode)

                mode = "overwrite" if episode == 1 else "overwrite_last"

                episode_total_delay_each_step = env.episode_delay_record
                episode_energy_each_step = env.episode_energy_record
                episode_each_step_records = [episode_total_delay_each_step, episode_energy_each_step]


if __name__ == "__main__":
    args = readParser()
    config = setup(args, sys.argv)
    main(args, config)
