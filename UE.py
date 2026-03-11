# -*- coing: utf-8 -*-

import constants as cn
import numpy as np
import random
import copy
import logging
from colorama import init, Fore, Style, Back
import math


class UserEquipment:
    def __init__(self, index):
        super().__init__()

        self.index = index
        self.ES = None
        self.seg_in_trans = None
        self.remaining_trans_size = 0
        self.wait_trans_queue = []
        self.config_log = []
        self.control_factor1 = np.random.uniform(0.25, 0.4, (1,))
        self.next_seg_config = []

    def generate_dag_task(self, time, env):

        new_task = copy.deepcopy(random.choices(env.dag_tasks)[0])
        new_task.UE = self
        env.total_job_num += 1
        new_task.create_time = time
        logging.info(Style.BRIGHT + Fore.BLUE + "t= {}, ue-{} generates a new dag task".format(time, self.index) + Style.RESET_ALL)

        return new_task

        logging.error(Style.BRIGHT + Fore.RED + Back.BLACK + "time= {}, ue-{}'s {}th wait_delay_ue < 0, its create time is {}".format(time, self.index, self.seg_in_trans.index, self.seg_in_trans.create_time) + Style.RESET_ALL)
