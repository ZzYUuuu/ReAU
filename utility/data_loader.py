import random
import math

import scipy.sparse as sp
import numpy as np
class Data(object):
    def __init__(self, args):
        self.args = args
        self.path = self.args.dataset_path + args.dataset
        self.filetype = self.args.dataset_type
        self.num_users = 0
        self.num_items = 0
        self.num_nodes = 0
        self.load_data_and_create_sp()
        if int(args.sparsity_test) == 1:
            self.split_test_dict, self.split_state = self.create_sparsity_split()
        elif int(args.sparsity_test) == 2:
            self.split_test_dict, self.split_state = self.create_item_sparsity_split()
    # inter -> interaction

    def load_data_and_create_sp(self):
        train_path = self.path + "/train" + self.filetype
        test_path = self.path + "/test" + self.filetype

        self.unique_train_users, self.train_users, self.train_items, self.train_pos_len, self.train_num_inter, self.train_dict = self.read_file(train_path)
        self.unique_test_users,  self.test_users,  self.test_items,  self.test_pos_len,  self.test_num_inter, self.test_dict = self.read_file(test_path)
        assert len(self.train_users) == len(self.train_items)

        self.num_users += 1
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items

        # U*I
        self.train_mat = sp.coo_matrix((np.ones(len(self.train_users)), (self.train_users, self.train_items)), shape=[self.num_users, self.num_items])
        self.test_mat  = sp.coo_matrix((np.ones(len(self.test_users)),  (self.test_users, self.test_items)),   shape=[self.num_users, self.num_items])

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))
        self.K = self.compute_local_space_scale()
        self.local_space_scale = self.K
        self.args.local_space_scale = self.local_space_scale

    def compute_local_space_scale(self):
        total_interactions = self.train_num_inter + self.test_num_inter
        rho = max(self.num_users / self.num_items, self.num_items / self.num_users)
        gamma = float(getattr(self.args, 'local_gamma', 1.0))
        target_size = gamma * math.sqrt(total_interactions * rho)
        return self.Q_D(target_size)

    @staticmethod
    def Q_D(value):
        if value <= 1:
            return 1
        lower = 2 ** int(math.floor(math.log2(value)))
        upper = lower * 2
        if value - lower <= upper - value:
            return lower
        return upper

    def read_file(self, file_name):
        inter_users, inter_items, unique_user, user_dict = [], [], [], {}
        pos_length = []
        num_inter = 0
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]

                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))

                unique_user.append(user_id)

                inter_users.extend([user_id] * len(pos_id))
                inter_items.extend(pos_id)

                pos_length.append(len(pos_id)) # [10, 20, 10, 15]
                num_inter += len(pos_id)

                for i in range(0, len(pos_id)):
                    if i == 0:
                        user_dict[user_id] = [pos_id[i]]
                    else:
                        user_dict[user_id].append(pos_id[i])

                line = f.readline()

        return np.array(unique_user), np.array(inter_users), np.array(inter_items), pos_length, num_inter, user_dict

    def random_create_awn(self):
        return np.column_stack((self.train_users, self.train_items))


    def sparse_adjacency_matrix(self):
        try:
            normal_adjacency = sp.load_npz(self.path + '/pre_Adj.npz')
            print('\t Adjacency matrix exist. Now loading!')
        except:
            print('\t Adjacency matrix not exist. Now constructing!')
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.train_mat.todok()
            # adjacency_matrix[row1:row2, column1:column2]
            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T

            # A_hat = D^(-1/2) A D(-1/2)
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            normal_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_Adj', normal_adjacency)
            print('\t Adjacency matrix constructed.')
        return normal_adjacency
        
    def sparse_adjacency_matrix_R(self):
        try:
            norm_adjacency = sp.load_npz(self.path + '/pre_R.npz')
            print('\t Adjacency matrix exists. Now loading!')
        except:
            print('\t Adjacency matrix not exist. Now constructing!')
    
            # R: (num_users, num_items)
            R = self.train_mat.tocoo()
    
            # 行度 = 每个用户的交互次数；列度 = 每个物品的交互次数
            row_sum = np.array(R.sum(axis=1))
            col_sum = np.array(R.sum(axis=0))
    
            row_d_inv = np.power(row_sum, -0.5).flatten(); row_d_inv[np.isinf(row_d_inv)] = 0.
            col_d_inv = np.power(col_sum, -0.5).flatten(); col_d_inv[np.isinf(col_d_inv)] = 0.
    
            D_u = sp.diags(row_d_inv)   # 用户度的 -1/2
            D_i = sp.diags(col_d_inv)   # 物品度的 -1/2
    
            # 归一化：R_hat = D_u^{-1/2} R D_i^{-1/2}
            norm_adjacency = D_u.dot(R).dot(D_i).tocsr()
    
            sp.save_npz(self.path + '/pre_R.npz', norm_adjacency)
            print('\t Adjacency matrix constructed.')

        return norm_adjacency


    def sparse_adjacency_matrix_RT(self):
        """
        返回 item->user 的归一化二部图： D_i^{-1/2} R^T D_u^{-1/2}
        如果已有 pre_R.npz（R_hat = D_u^{-1/2} R D_i^{-1/2）），则直接转置得到。
        若没有，则用 self.user_item_net 现算两者并缓存。
        """
        # 1) 先尝试用现成的 R_hat 转置得到 RT_hat
        try:
            R_hat = sp.load_npz(self.path + '/pre_R.npz')           # (num_users, num_items)
            print('\t Loaded normalized R (user->item). Returning its transpose for item->user.')
            RT_hat = R_hat.transpose().tocsr()                       # (num_items, num_users)
            # （可选）形状自检
            # assert RT_hat.shape == (self.num_items, self.num_users)
            return RT_hat
        except Exception as e:
            print('\t Normalized R not found. Recomputing both directions...', e)
    
        # 2) 现算：用真实的 R（user_item_net），而不是空方阵
        R = self.train_mat.tocoo()                               # (U, I)
    
        # 行度=用户交互数；列度=物品交互数
        row_sum = np.array(R.sum(axis=1))                            # (U, 1)
        col_sum = np.array(R.sum(axis=0))                            # (1, I)
    
        du_inv = np.power(row_sum, -0.5).flatten()
        du_inv[np.isinf(du_inv)] = 0.
        di_inv = np.power(col_sum, -0.5).flatten()
        di_inv[np.isinf(di_inv)] = 0.
    
        D_u = sp.diags(du_inv)                                       # (U, U)
        D_i = sp.diags(di_inv)                                       # (I, I)
    
        # R_hat: user->item；RT_hat: item->user
        R_hat  = D_u.dot(R).dot(D_i).tocsr()                         # (U, I)
        RT_hat = R_hat.transpose().tocsr()                           # (I, U)
    
        sp.save_npz(self.path + '/pre_R.npz',  R_hat)
        sp.save_npz(self.path + '/pre_RT.npz', RT_hat)
        print('\t Built and cached both R_hat (U->I) and RT_hat (I->U).')
    
        return RT_hat


    
    def sparse_adjacency_matrix_self(self):
        try:
            normal_adjacency = sp.load_npz(self.path + '/pre_Adj_self.npz')
            print('\t Adjacency matrix exist. Now loading!')
        except:
            print('\t Adjacency matrix not exist. Now constructing!')
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.train_mat.todok()
            # adjacency_matrix[row1:row2, column1:column2]
            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T

            # add self
            adjacency_matrix = adjacency_matrix.todok()
            adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

            # A_hat = D^(-1/2) A D(-1/2)
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            normal_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_Adj_self', normal_adjacency)
            print('\t Adjacency matrix constructed.')
        return normal_adjacency
    def user_item_num(self):
        return self.num_users, self.num_items

    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = dict()

        for uid in all_users:
            train_iids = self.all_positive[uid]
            test_iids = self.test_dict[uid]

            num_iids = len(train_iids) + len(test_iids)

            if num_iids not in user_n_iid.keys():
                user_n_iid[num_iids] = [uid]
            else:
                user_n_iid[num_iids].append(uid)

        split_uids = list()
        temp = []
        count = 1
        fold = 3
#         fold = 4
        n_count = self.train_num_inter + self.test_num_inter
        n_rates = 0
        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.334 * (self.train_num_inter + self.test_num_inter):
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

    def create_item_sparsity_split(self):
        item_popularity = np.bincount(self.train_items, minlength=self.num_items)
        pop_n_test = dict()

        for _, test_iids in self.test_dict.items():
            for iid in test_iids:
                pop = int(item_popularity[iid])
                if pop not in pop_n_test:
                    pop_n_test[pop] = 1
                else:
                    pop_n_test[pop] += 1

        if len(pop_n_test) == 0:
            return [{}, {}, {}, {}, {}], [
                '\t tail-1 items: empty',
                '\t tail-2 items: empty',
                '\t mid items: empty',
                '\t head-2 items: empty',
                '\t head-1 items: empty'
            ]

        total_test_inter = sum(pop_n_test.values())
        targets = [total_test_inter * ratio for ratio in [0.2, 0.4, 0.6, 0.8]]
        thresholds = [None, None, None, None]
        cumulative = 0
        for pop in sorted(pop_n_test):
            cumulative += pop_n_test[pop]
            for idx, target in enumerate(targets):
                if thresholds[idx] is None and cumulative >= target:
                    thresholds[idx] = pop

        max_pop = max(pop_n_test.keys())
        thresholds = [threshold if threshold is not None else max_pop for threshold in thresholds]

        split_test_dict = [{}, {}, {}, {}, {}]
        split_item_sets = [set(), set(), set(), set(), set()]

        for uid, test_iids in self.test_dict.items():
            grouped_items = [[], [], [], [], []]
            for iid in test_iids:
                pop = int(item_popularity[iid])
                if pop <= thresholds[0]:
                    group_idx = 0
                elif pop <= thresholds[1]:
                    group_idx = 1
                elif pop <= thresholds[2]:
                    group_idx = 2
                elif pop <= thresholds[3]:
                    group_idx = 3
                else:
                    group_idx = 4
                grouped_items[group_idx].append(iid)
                split_item_sets[group_idx].add(iid)

            for group_idx, group_items in enumerate(grouped_items):
                if len(group_items) > 0:
                    split_test_dict[group_idx][uid] = group_items

        split_state = []
        labels = ['tail-1', 'tail-2', 'mid', 'head-2', 'head-1']
        for idx, label in enumerate(labels):
            num_users = len(split_test_dict[idx])
            num_items = len(split_item_sets[idx])
            num_rates = sum(len(items) for items in split_test_dict[idx].values())
            if idx == 0:
                state = '\t %s items: train_inter_per_item<=[%d], #users=[%d], #items=[%d], #test_rates=[%d]' % (
                    label, thresholds[0], num_users, num_items, num_rates)
            elif idx == 4:
                state = '\t %s items: train_inter_per_item>(%d), #users=[%d], #items=[%d], #test_rates=[%d]' % (
                    label, thresholds[3], num_users, num_items, num_rates)
            else:
                state = '\t %s items: train_inter_per_item=(%d,%d], #users=[%d], #items=[%d], #test_rates=[%d]' % (
                    label, thresholds[idx - 1], thresholds[idx], num_users, num_items, num_rates)
            split_state.append(state)
            print(state)

        return split_test_dict, split_state

    def get_user_pos_items(self, users):
        self.train_mat_csr = self.train_mat.tocsr()
        positive_items = []
        for user in users:
            positive_items.append(self.train_mat_csr[user].nonzero()[1])
        return positive_items
