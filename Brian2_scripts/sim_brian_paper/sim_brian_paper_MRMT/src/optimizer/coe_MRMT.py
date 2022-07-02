from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.src.optimizer.coe import *

class CoE_MRMT_reservoir(CoE):
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state, maxormin,
                 block_init, block_max, increase_threshold):
        super().__init__(f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state, maxormin)
        self.block_current = block_init
        self.block_max = block_max
        self.increase_threshold = increase_threshold

    # def is_increase_block(self):
    #     if self.F_B < self.increase_threshold:
    #         return True
    #     else:
    #         return False

    def update_ranges(self):
        if self.block_current < self.block_max:
            self.block_current += 1
            config_ranges_reservoir_new = \
                [[0, 2 ** self.block_current - 1]] * self.block_current + \
                [[0, 2 ** 0 - 1]] * (self.block_max - self.block_current)
            index_ib = np.where(np.array(self.codes[SubCom]) != None)[0]
            self.ranges[:, SubCom][:, index_ib] = np.array(config_ranges_reservoir_new).T