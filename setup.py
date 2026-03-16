import logging
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from constants import ROOT_DIR
import torch
import sys
import argparse
import shutil
import numpy as np
import random
from typing import Callable, Sequence, Tuple


def readParser():
    parser = argparse.ArgumentParser(description="Heterogeneous DAG")

    # ===========  common config  ==============
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log2file", action="store_true", help="Enable logging to file")
    parser.add_argument("--log2stdout", action="store_false", help="Not enable logging to stdout")
    parser.add_argument("--data_save_dir", default="", type=str, help="save all data to the given file directory")
    parser.add_argument("--clr_data_first", action="store_true", help="clear exited datas before saving new datas")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--mode", default="train", type=str, choices=["train", "eval"])
    parser.add_argument("--device", "-d", default="cuda:2", help="run on CUDA (default: cuda:0)")

    # ==========  common RL config  ============
    parser.add_argument(
        "--algorithm",
        type=str,
        default="HADES",
        choices=["HADES", "SDAC", "QVPO", "ACED", "MESON", "BSAC", "Random"],
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor (default: 0.99)")
    parser.add_argument("--remark", type=str, default="test")

    # common training config
    parser.add_argument("--threshod", "-th", default=128, type=int)
    parser.add_argument("--update_coef", type=float, default=0.5, help="update_interval = update_coef * max_episilon_len")
    parser.add_argument("--train_interval", type=int, default=256, help="the interval between two training process(default: 256)")
    parser.add_argument("--batch_size", type=int, default=32, help="(default: 1024)")
    parser.add_argument("--epochs_num", "-epc", default=1, type=int, help="update epochs_num times every training")
    parser.add_argument("--policy_update_delay", default=4, type=int, help="policy update interval(steps) between two updates")
    parser.add_argument("--delay_alpha_update", type=int, default=2048, help="Soft update coefficient for target networks")

    parser.add_argument("--eval_intervel", "-ei", default=20, type=int)

    parser.add_argument("--reward_type", type=int, default=0)
    parser.add_argument("--delay_coef", type=float, default=0.5, help="delay reward coefficient")
    parser.add_argument("--energy_coef", type=float, default=5e-2, help="energy reward coefficient")
    parser.add_argument("--expected_delay_CVAR", type=float, default=0.3, help="seccond")

    parser.add_argument("--memory_size", default=50000, type=int)

    parser.add_argument("--sdac_lr", type=float, default=3e-4, help="Learning rate for both actor and critic networks")
    parser.add_argument("--alpha_lr", type=float, default=3e-2, help="Learning rate for temperature parameter alpha")
    parser.add_argument("--target_entropy", type=float, default=None, help="Target entropy for automatic entropy tuning (default: -action_dim)")
    parser.add_argument("--update_actor_target_every", type=int, default=1, help="update actor target per iteration (default: 1)")

    # ================================= Env config =======================================
    parser.add_argument("--num_episode", type=int, default=10, help="episode num (default: 300)")
    parser.add_argument("--max_ep_len", type=int, default=128, help="maximum episode length (default: 2048)")

    parser.add_argument("--slot_length", type=float, default=0.2, help="system slot length")
    parser.add_argument("--es_num", type=int, default=15, help="number of es")
    parser.add_argument("--ue_num", "-u", default=8, type=int, help="UEs num")

    parser.add_argument("--dag_num", type=int, default=5)
    parser.add_argument("--prob", type=int, default=1, help="the probability of generating DAG task in each time slot")
    parser.add_argument("--dag_node_num", type=int, default=5, help="The number of DAG's node")
    parser.add_argument("--dag_max_out", type=int, default=2, help="The max out degree of DAG nodes")
    parser.add_argument("--dag_source", type=str, default="huawei", choices=["huawei", "cluster"], help="DAG generation source")
    parser.add_argument("--dag_dataset_shuffle", action="store_true", help="Shuffle dataset DAG order before sampling")

    # ===================================  DIPO  =========================================
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")

    parser.add_argument("--beta_schedule", type=str, default="cosine", help="linear, cosine or vp")
    parser.add_argument("--diffusion_lr", type=float, default=0.0003, help="diffusion learning rate (default: 0.0003)")
    parser.add_argument("--critic_lr", type=float, default=0.00003, help="critic learning rate (default: 0.0003)")
    parser.add_argument("--action_lr", type=float, default=0.03, help="diffusion learning rate (default: 0.03)")
    parser.add_argument("--noise_ratio", type=float, default=1.0, help="noise ratio in sample process (default: 1.0)")
    parser.add_argument("--action_update_epochs", type=int, default=20, help="每次更新过程中action更新多少次(action gradient根据Q更新的epoch数),action gradient steps (default: 20)")
    parser.add_argument("--ratio", type=float, default=0.1, help="the ratio of action grad norm to action_dim (default: 0.1)")
    parser.add_argument("--ac_grad_norm", type=float, default=1.0, help="actor and critic grad norm (default: 1.0)")
    parser.add_argument("--cri_grad_norm", type=float, default=1.0, help="actor and critic grad norm (default: 1.0)")

    # =========================================  SDAC =======================================
    # Network architectures
    parser.add_argument("--sdac_q_hidden_sizes", type=Sequence[int], default=(256, 256), help="Q network hidden layer sizes")
    parser.add_argument("--sdac_policy_hidden_sizes", type=Sequence[int], default=(256, 256), help="Policy network hidden layer sizes")
    # Diffusion process parameters
    parser.add_argument("--diffusion_steps", type=int, default=5, help="Number of diffusion timesteps")
    parser.add_argument("--sdac_beta_schedule", type=str, default="linear")
    parser.add_argument("--num_particles", type=int, default=1, help="Number of action samples (particles) to generate")
    # Training parameters
    parser.add_argument("--sdac_noise_scale", type=float, default=0.1, help="Scale of Gaussian noise added to actions for exploration")

    # ========================================  QVPO  =====================================
    parser.add_argument("--alpha_mean", type=float, default=0.001, help="running mean update weight (default: 0.1)")

    parser.add_argument("--alpha_std", type=float, default=0.001, help="running std update weight (default: 0.001)")

    parser.add_argument("--beta", type=float, default=1.0, help="expQ weight (default: 1.0)")

    parser.add_argument("--weighted", type=bool, default=True, help="weighted training")

    parser.add_argument("--aug", type=bool, default=True, help="augmentation")

    parser.add_argument("--train_sample", type=int, default=64, help="train_sample (default: 64)")

    parser.add_argument("--chosen", type=int, default=1, help="chosen actions (default:1)")

    parser.add_argument("--q_neg", type=float, default=0.0, help="q_neg (default: 0.0)")

    parser.add_argument("--behavior_sample", type=int, default=4, help="behavior_sample (default: 1)")
    parser.add_argument("--target_sample", type=int, default=4, help="target_sample (default: behavior sample)")

    parser.add_argument("--eval_sample", type=int, default=32, help="eval_sample (default: 512)")

    parser.add_argument("--deterministic", action="store_true", help="deterministic mode")

    parser.add_argument("--q_transform", type=str, default="qadv", help="q_transform (default: qrelu)")

    parser.add_argument("--gradient", action="store_true", help="aug gradient")

    parser.add_argument("--cut", type=float, default=1.0, help="cut (default: 1.0)")
    parser.add_argument("--times", type=int, default=1, help="times (default: 1)")

    parser.add_argument("--epsilon", type=float, default=0.0, help="eps greedy (default: 0.0)")
    parser.add_argument("--entropy_alpha", type=float, default=0.02, help="entropy_alpha (default: 0.02)")

    # =========================================  GCN-PPO  =====================================
    parser.add_argument("--ppo_lr_actor", type=float, default=0.0003)
    parser.add_argument("--ppo_lr_critic", type=float, default=0.0003)

    parser.add_argument("--eps_clip", "-cp", default=0.15, type=float, help="epsilon clip")
    parser.add_argument("--coef_entropy", "-ce", default=0.005, type=float, help="PPO_d的动作熵惩罚")

    parser.add_argument("--action_std_init", "-asi", default=0.15, type=float)
    parser.add_argument("--action_std_decay_rate", default=0.02, type=float)
    parser.add_argument("--min_action_std", "-mas", default=0.02, type=float)

    parser.add_argument("--gcn_layer_num", "-gln", default=2, type=int)
    parser.add_argument("--attention", action="store_true", help="Enable attention")
    parser.add_argument("--normalize", action="store_true", help="Enable normalize")

    # =======================================  DDPG  ====================================
    parser.add_argument("--ddpg_actor_lr", type=float, default=0.0003)
    parser.add_argument("--ddpg_critic_lr", type=float, default=0.0003)
    parser.add_argument("--ddpg_init_noise_std", type=float, default=0.2)
    parser.add_argument("--ddpg_action_std_decay_rate", type=float, default=0.95)
    parser.add_argument("--ddpg_min_action_std", type=float, default=0.01)

    # SAC
    parser.add_argument("--sac_actor_lr", default=0.0003, type=float)
    parser.add_argument("--sac_critic_lr", default=0.0003, type=float)
    parser.add_argument("--sac_buffer_capacity", default=100000, type=float)
    parser.add_argument("--sac_alpha", default=0.2, type=float)
    parser.add_argument("--sac_alpha_lr", default=0.0003, type=float)
    parser.add_argument("--sac_target_entropy", default=None)

    return parser.parse_args()


def setup(args, argv):
    start_time = datetime.now()

    # 记录关键信息
    logging.critical("Commond= {} \n Time= {}, Algorithm= {}, Remark= {}, Pid= {}, Device= {}".format(" ".join(argv), start_time, args.algorithm, args.remark, os.getpid(), args.device))

    # TensorBoard 设置
    SummaryWriter_dir = "log"

    log_dir = os.path.join(
        SummaryWriter_dir,
        f'{start_time.strftime("%Y-%m-%d")}',
        f"{args.algorithm}",
        f"{args.remark}",
    )
    writer = SummaryWriter(log_dir)

    # 模型保存目录
    model_save_dir = ROOT_DIR + "/model/" + args.algorithm + "/" + args.remark
    os.makedirs(model_save_dir, exist_ok=True)

    # 设备设置
    device = torch.device(args.device)

    # 设置seed
    seed = args.seed
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDA 确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python 和 NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 返回配置
    config = {
        "writer": writer,
        "model_save_dir": model_save_dir,
        "device": device,
    }
    return config
