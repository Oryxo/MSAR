from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.msar_tools import EarlyStopping, adjust_learning_rate
from utils.msar_metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy

warnings.filterwarnings('ignore')


class Exp_Msar_Forecast(Exp_Basic):

    def __init__(self, args):
        super().__init__(args)
        self.models = [m.to(self.device) for m in self._build_model()]

        model_input_read_len = np.array(self.args.seq_len_list) * np.array(
            self.args.interval_list)
        max_window = max(model_input_read_len)
        self.args.seq_len = max_window + self.args.pred_len

        self.mask = self._get_mask()

        total_params = 0
        print("=" * 60)
        for idx, model in enumerate(self.models):
            model_parameters = filter(lambda p: p.requires_grad,
                                      model.parameters())
            param_count = sum([np.prod(p.size()) for p in model_parameters])
            total_params += param_count

            # memory usage (float32 = 4 bytes)
            memory_usage_bytes = param_count * 4
            memory_usage_MB = memory_usage_bytes / (1024**2)

            print(f"[Model {idx}] Trainable parameters: {param_count}")
            print(f"[Model {idx}] Memory usage: {memory_usage_MB:.2f} MB")

            print("=" * 60)
            print(
                f"Total trainable parameters across all models: {total_params}"
            )
            print(
                f"Total memory usage for trainable parameters: {total_params*4/(1024**2):.2f} MB"
            )

            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            else:
                allocated = reserved = 0.0
            print(f"Static memory footprint (allocated): {allocated:.2f} MB")
            print(f"Static memory footprint (reserved): {reserved:.2f} MB")

    def _get_mask(self):
        """ 
        Get masks for imputation models based on intervals and prediction length. 
        Returns a list of masks, one for each imputation model.
        Redundant in index 0.
        """
        mask = []
        temp_mask = torch.zeros(len(self.args.interval_list),
                                self.args.pred_len,
                                dtype=torch.int32)
        known_indices = set()

        for idx, interval in enumerate(self.args.interval_list):
            indices = torch.arange(0,
                                   self.args.pred_len,
                                   interval,
                                   dtype=torch.long)
            new_indices = [
                idx.item() for idx in indices
                if idx.item() not in known_indices
            ]
            known_indices.update(indices.tolist())
            temp_mask[idx, indices] = 1
            temp_mask[idx, new_indices] = 0
            mask.append(
                torch.concat((torch.ones(self.args.seq_len_list[idx]),
                              temp_mask[idx, ::interval])))
        return mask

    def _build_model(self):
        """
        Build multiple models based on interval_list and seq_len_list.
        The first model is for forecasting, and the rest are for imputation.
        """
        models = []  # List to store all models
        self.args_list = []  # List to store args for each model

        # Iterate through interval_list and seq_len_list
        for i, (interval, seq_len) in enumerate(
                zip(self.args.interval_list, self.args.seq_len_list)):
            # Clone args for each model
            model_args = copy.copy(self.args)
            # model_args['seq_len'] = seq_len
            model_args.interval = interval
            model_args.checkpoints = os.path.join(self.args.checkpoints,
                                                  f'model_{i}')
            # Determine model type: first is forecast, others are imputation
            if i == 0:
                # forecast model
                model_args.seq_len = seq_len
                model_args.pred_len = (self.args.pred_len + interval -
                                       1) // interval
                model_args.true_pred_len = (self.args.pred_len + interval -
                                            1) // interval
                model_args.true_label_len = (self.args.label_len + interval -
                                             1) // interval
                model_args.task_name = 'long_term_forecast'
            else:
                # imputation model
                model_args.seq_len = seq_len + (self.args.pred_len + interval -
                                                1) // interval
                model_args.pred_len = 0  # imputation models do not predict future steps
                model_args.label_len = 0  # imputation models do not use label_len
                model_args.true_seq_len = seq_len + (self.args.pred_len +
                                                     interval - 1) // interval
                model_args.true_pred_len = (self.args.pred_len + interval -
                                            1) // interval
                model_args.task_name = 'imputation'
                # Compact imputation is the default in the CLI. For DUET,
                # --full_imputation is recommended to retain forecasting width.
                if getattr(self.args, 'slim_imputation', False):
                    model_args.d_model = self.args.slim_d_model
                    model_args.d_ff = self.args.slim_d_ff

            # Create model instance
            model = self.model_dict[self.args.model](model_args).float()

            # Wrap with DataParallel if using multiple GPUs
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)

            # Append to lists
            models.append(model)
            self.args_list.append(model_args)

        for i, model_args in enumerate(self.args_list):
            if i == 0:
                # forecast model
                model_args.seq_len = model_args.seq_len * model_args.interval
                model_args.pred_len = self.args.pred_len
            else:
                # imputation model
                model_args.seq_len = model_args.seq_len * model_args.interval

        return models

    def _get_data(self, args, flag):
        data_set, data_loader = data_provider(args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(),
                                 lr=self.args.learning_rate)
        return model_optim

    def train_tf(self, setting, idx):
        train_data, train_loader = self._get_data(args=self.args_list[idx],
                                                  flag='train')
        vali_data, vali_loader = self._get_data(args=self.args_list[idx],
                                                flag='val')
        test_data, test_loader = self._get_data(args=self.args_list[idx],
                                                flag='test')

        path = os.path.join(self.args_list[idx].checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True)

        model_optim = self._select_optimizer(self.models[idx])
        criterion = self._select_criterion()

        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()

        print("++++++++++++++++++++++++++++++++++++++++++")
        print(f"Teacher Forcing Training for Model {idx}:")
        print("++++++++++++++++++++++++++++++++++++++++++")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.models[idx].train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x[:, ::self.args_list[idx].interval, :].float(
                ).to(self.device)
                batch_x_mark = batch_x_mark[:, ::self.args_list[idx].
                                            interval, :].float().to(
                                                self.device)

                if idx == 0:
                    # model for forecast
                    batch_y = batch_y[:, ::self.args_list[idx].
                                      interval, :].float().to(self.device)
                    batch_y_mark = batch_y_mark[:, ::self.args_list[idx].
                                                interval, :].float().to(
                                                    self.device)

                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    # decoder input
                    if idx == 0 and self.args.use_dec_inp:
                        dec_inp = torch.zeros_like(
                            batch_y[:, -self.args_list[idx].true_pred_len:, :]
                        ).float()
                        dec_inp = torch.cat([
                            batch_y[:, :self.args_list[idx].true_label_len, :],
                            dec_inp
                        ],
                                            dim=1).float().to(self.device)
                    else:
                        dec_inp = None
                    # outputs = self.models[idx](batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, aux = self._forward_any(self.models[idx], batch_x,
                                                     batch_x_mark, dec_inp,
                                                     batch_y_mark)
                else:
                    # model for imputation
                    #  mask
                    B, T, N = batch_x.shape
                    mask = self.mask[idx].to(self.device)
                    mask = mask.unsqueeze(0).unsqueeze(-1).repeat(B, 1, N)

                    inp = batch_x.masked_fill(mask == 0, 0)

                    # outputs = self.models[idx](inp, batch_x_mark, None, None, mask)
                    outputs, aux = self._forward_any(self.models[idx], inp,
                                                     batch_x_mark, None, None,
                                                     mask)

                # add support for MS
                f_dim = -1 if self.args.features == 'MS' else 0
                if idx == 0:
                    outputs = outputs[:, -self.args_list[idx].true_pred_len:,
                                      f_dim:]
                    batch_y = batch_y[:, -self.args_list[idx].true_pred_len:,
                                      f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                else:
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]

                    loss = criterion(outputs[mask == 0], batch_x[mask == 0])

                loss = loss + self._auxiliary_loss(aux)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
                        speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1,
                                                   time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali_tf(vali_data, vali_loader, criterion, idx)
            test_loss = self.vali_tf(test_data, test_loader, criterion, idx)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss,
                        test_loss))
            early_stopping(vali_loss, self.models[idx], path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.models[idx].load_state_dict(torch.load(best_model_path))

        return self.models

    def vali_tf(self, vali_data, vali_loader, criterion, idx):
        total_loss = []
        self.models[idx].eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x[:, ::self.args_list[idx].interval, :].float(
                ).to(self.device)
                batch_x_mark = batch_x_mark[:, ::self.args_list[idx].
                                            interval, :].float().to(
                                                self.device)

                if idx == 0:
                    # forecast model
                    batch_y = batch_y[:, ::self.args_list[idx].
                                      interval, :].float()
                    batch_y_mark = batch_y_mark[:, ::self.args_list[idx].
                                                interval, :].float().to(
                                                    self.device)

                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None

                    # decoder input
                    if idx == 0 and self.args.use_dec_inp:
                        dec_inp = torch.zeros_like(
                            batch_y[:, -self.args_list[idx].true_pred_len:, :]
                        ).float()
                        dec_inp = torch.cat([
                            batch_y[:, :self.args_list[idx].true_label_len, :],
                            dec_inp
                        ],
                                            dim=1).float().to(self.device)
                    else:
                        dec_inp = None
                    # outputs = self.models[idx](batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, aux = self._forward_any(self.models[idx], batch_x,
                                                     batch_x_mark, dec_inp,
                                                     batch_y_mark)

                else:
                    # imputation model
                    B, T, N = batch_x.shape
                    mask = self.mask[idx].to(self.device)
                    mask = mask.unsqueeze(0).unsqueeze(-1).repeat(B, 1, N)

                    inp = batch_x.masked_fill(mask == 0, 0)

                    # outputs = self.models[idx](inp, batch_x_mark, None, None, mask)
                    outputs, aux = self._forward_any(self.models[idx], inp,
                                                     batch_x_mark, None, None,
                                                     mask)

                f_dim = -1 if self.args.features == 'MS' else 0

                if idx == 0:
                    outputs = outputs[:, -self.args_list[idx].true_pred_len:,
                                      f_dim:]
                    batch_y = batch_y[:, -self.args_list[idx].true_pred_len:,
                                      f_dim:].to(self.device)

                    pred = outputs.detach()
                    true = batch_y.detach()

                    # loss = criterion(pred, true)
                else:
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]

                    pred = outputs.detach()
                    true = batch_x.detach()
                    mask = mask.detach()

                    # loss = criterion(pred[mask == 0], true[mask == 0])

                # total_loss.append(loss.item())
                if idx == 0:
                    if self.args.data == 'PEMS':
                        B, T, C = pred.shape
                        pred = pred.cpu().numpy()
                        true = true.cpu().numpy()
                        pred = vali_data.inverse_transform(pred.reshape(
                            -1, C)).reshape(B, T, C)
                        true = vali_data.inverse_transform(true.reshape(
                            -1, C)).reshape(B, T, C)
                        mae, mse, rmse, mape, mspe = metric(pred, true)
                        total_loss.append(mae)

                    else:
                        loss = criterion(pred, true)
                        total_loss.append(loss.item())
                else:
                    if self.args.data == 'PEMS':
                        B, T, C = pred.shape
                        pred = pred.cpu().numpy()
                        true = true.cpu().numpy()
                        pred = vali_data.inverse_transform(pred.reshape(
                            -1, C)).reshape(B, T, C)
                        true = vali_data.inverse_transform(true.reshape(
                            -1, C)).reshape(B, T, C)
                        mae, mse, rmse, mape, mspe = metric(
                            pred[mask.cpu().numpy() == 0],
                            true[mask.cpu().numpy() == 0])
                        total_loss.append(mae)

                    else:
                        loss = criterion(pred[mask == 0], true[mask == 0])
                        total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.models[idx].train()
        return total_loss

    def train_joint(self, setting):
        """
        Coarse And Fine Joint Learning
        """
        if len(self.models) == 1:
            return

        # Use the maximum of (seq_len_layers * interval) in the hierarchical stage
        train_data_joint, train_loader_joint = self._get_data(self.args,
                                                              flag='train')
        vali_data_joint, vali_loader_joint = self._get_data(self.args,
                                                            flag='val')
        test_data_joint, test_loader_joint = self._get_data(self.args,
                                                            flag='test')

        time_now = time.time()

        print('Loading All Models...')
        path_list = []
        for idx, model in enumerate(self.models):
            path = os.path.join(self.args_list[idx].checkpoints, setting)
            if idx != 0:  # train joint wo first forecast model
                path_list.append(path)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.models[idx].load_state_dict(torch.load(best_model_path))

        train_steps = len(train_loader_joint)
        optimizer = {}
        for idx, model in enumerate(self.models):
            if idx != 0:  # train joint wo first forecast model
                optimizer[idx] = self._select_optimizer(model)

        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True)
        criterion = self._select_criterion()
        for idx in range(len(self.models)):
            if idx != 0:
                self.models[idx].to(self.device)
        # torch.autograd.set_detect_anomaly(True)

        self.models[0].eval()
        for p in self.models[0].parameters():
            p.requires_grad_(False)

        print("++++++++++++++++++++++++++++++")
        print("Joint Training:")
        print("++++++++++++++++++++++++++++++")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            # train_loss = []

            # self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(train_loader_joint):
                iter_count += 1
                total_loss = 0
                train_loss = []
                input_temp = {}
                output_temp = {}
                target_temp = {}
                input_mark_temp = {}
                target_mark_temp = {}
                concat_temp = {}

                for idx, interval in enumerate(self.args.interval_list):
                    if idx != 0:
                        self.models[idx].train()
                        optimizer[idx].zero_grad()

                    input_temp[idx], target_temp[idx], input_mark_temp[
                        idx], target_mark_temp[idx] = (
                            batch_x[:, -(self.args.seq_len_list[idx] *
                                         interval)::interval, :].float().to(
                                             self.device),
                            # batch_y[:, -self.args.pred_len::interval, :].float().to(self.device),
                            batch_y[:,
                                    -(self.args.pred_len + self.args.label_len
                                      )::interval, :].float().to(self.device),
                            # batch_x_mark[:, -(self.args.seq_len_list[idx] * interval)::interval, :].float().to(self.device),
                            batch_x_mark[:, -self.args_list[idx].
                                         seq_len::interval, :].float().to(
                                             self.device),
                            # batch_y_mark[:, -self.args.pred_len::interval, :].float().to(self.device),
                            batch_y_mark[:,
                                         -(self.args.pred_len +
                                           self.args.label_len)::interval, :].
                            float().to(self.device),
                        )

                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        input_mark_temp[idx] = None
                        target_mark_temp[idx] = None

                    if idx == 0 and self.args.use_dec_inp:
                        dec_inp = torch.zeros_like(
                            target_temp[idx]
                            [:,
                             -self.args_list[idx].true_pred_len:, :]).float()
                        dec_inp = torch.cat([
                            target_temp[idx][:, :self.args_list[idx].
                                             true_label_len, :], dec_inp
                        ],
                                            dim=1).float().to(self.device)
                    else:
                        dec_inp = None

                    if idx == 0:
                        # output_temp[idx] = self.models[idx](input_temp[idx], input_mark_temp[idx], dec_inp, target_mark_temp[idx])
                        output_temp[idx], aux = self._forward_any(
                            self.models[idx], input_temp[idx],
                            input_mark_temp[idx], dec_inp,
                            target_mark_temp[idx])

                    else:
                        B, T, N = input_temp[idx].shape
                        mask = self.mask[idx].to(self.device)
                        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(B, 1, N)

                        concat_temp[idx] = torch.zeros(
                            (B, self.args.pred_len, N)).to(self.device)

                        for temp_idx, interval_temp in enumerate(range(idx)):
                            concat_temp[idx][:, ::self.args.interval_list[
                                temp_idx], :] = output_temp[temp_idx]
                        input_temp[idx] = torch.cat(
                            (input_temp[idx], concat_temp[idx]
                             [:, ::self.args.interval_list[idx], :]),
                            dim=1)  # * mask[idx]

                        # output_temp[idx] = self.models[idx](input_temp[idx], input_mark_temp[idx], None, None, mask)
                        output_temp[idx], aux = self._forward_any(
                            self.models[idx], input_temp[idx],
                            input_mark_temp[idx], None, None, mask)

                    target_temp[idx] = target_temp[idx][:,
                                                        -self.args_list[idx].
                                                        true_pred_len:, :]
                    output_temp[idx] = output_temp[idx][:,
                                                        -self.args_list[idx].
                                                        true_pred_len:, :]
                    loss = criterion(output_temp[idx], target_temp[idx])

                    loss = loss + self._auxiliary_loss(aux)

                    if idx != 0:
                        train_loss.append((total_loss + loss).item())
                        total_loss = total_loss + loss

                    if (i + 1) % 100 == 0 and idx == len(self.models) - 1:
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                                i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / (iter_count +
                                                            0.00001)
                        left_time = speed * (
                            (self.args.train_epochs - epoch) * train_steps - i)
                        print(
                            '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
                                speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                total_loss.backward()
                for idx in optimizer.keys():
                    optimizer[idx].step()

            print("Epoch: {} cost time: {}".format(epoch + 1,
                                                   time.time() - epoch_time))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data_joint, vali_loader_joint,
                                  criterion)
            test_loss = self.vali(test_data_joint, test_loader_joint,
                                  criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss,
                        test_loss))
            early_stopping(vali_loss, self.models[1:], path_list)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            for idx in optimizer.keys():
                adjust_learning_rate(optimizer[idx], epoch + 1,
                                     self.args_list[idx])

        for idx, model in enumerate(self.models):
            path = os.path.join(self.args_list[idx].checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.models[idx].load_state_dict(torch.load(best_model_path))

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        for idx in range(len(self.models)):
            self.models[idx].eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                input_temp = {}
                output_temp = {}
                concat_temp = {}

                target_temp = {}
                input_mark_temp = {}
                target_mark_temp = {}

                for idx, interval in enumerate(self.args.interval_list):

                    input_temp[idx], target_temp[idx], input_mark_temp[
                        idx], target_mark_temp[idx] = (
                            batch_x[:, -(self.args.seq_len_list[idx] *
                                         interval)::interval, :].float().to(
                                             self.device),
                            # batch_y[:, -self.args.pred_len::interval, :].float().to(self.device),
                            batch_y[:,
                                    -(self.args.pred_len + self.args.label_len
                                      )::interval, :].float().to(self.device),
                            # batch_x_mark[:, -(self.args.seq_len_list[idx] * interval)::interval, :].float().to(self.device),
                            batch_x_mark[:, -self.args_list[idx].
                                         seq_len::interval, :].float().to(
                                             self.device),
                            # batch_y_mark[:, -self.args.pred_len::interval, :].float().to(self.device),
                            batch_y_mark[:,
                                         -(self.args.pred_len +
                                           self.args.label_len)::interval, :].
                            float().to(self.device),
                        )

                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        input_mark_temp[idx] = None
                        target_mark_temp[idx] = None

                    if idx == 0 and self.args.use_dec_inp:
                        dec_inp = torch.zeros_like(
                            target_temp[idx]
                            [:,
                             -self.args_list[idx].true_pred_len:, :]).float()
                        dec_inp = torch.cat([
                            target_temp[idx][:, :self.args_list[idx].
                                             true_label_len, :], dec_inp
                        ],
                                            dim=1).float().to(self.device)
                    else:
                        dec_inp = None

                    if idx == 0:
                        # output_temp[idx] = self.models[idx](input_temp[idx], input_mark_temp[idx], dec_inp, target_mark_temp[idx])
                        output_temp[idx], aux = self._forward_any(
                            self.models[idx], input_temp[idx],
                            input_mark_temp[idx], dec_inp,
                            target_mark_temp[idx])
                    else:
                        B, T, N = input_temp[idx].shape
                        mask = self.mask[idx].to(self.device)
                        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(B, 1, N)

                        concat_temp[idx] = torch.zeros(
                            (B, self.args.pred_len, N)).to(self.device)

                        for temp_idx, interval_temp in enumerate(range(idx)):
                            concat_temp[idx][:, ::self.args.interval_list[
                                temp_idx], :] = output_temp[temp_idx]
                        input_temp[idx] = torch.cat(
                            (input_temp[idx], concat_temp[idx]
                             [:, ::self.args.interval_list[idx], :]),
                            dim=1)  # * mask[idx]
                        # output_temp[idx] = self.models[idx](input_temp[idx], input_mark_temp[idx], None, None, mask)
                        output_temp[idx], aux = self._forward_any(
                            self.models[idx], input_temp[idx],
                            input_mark_temp[idx], None, None, mask)

                    output_temp[idx] = output_temp[idx][:,
                                                        -self.args_list[idx].
                                                        true_pred_len:, :]

                output = output_temp[max(output_temp.keys())]

                # loss = criterion(output[:, -(self.args.pred_len):, :], batch_y[:, -(self.args.pred_len):, :]).item()
                # total_loss.append(loss)

                pred = output[:, -(self.args.pred_len):, :].detach()
                true = batch_y[:, -(self.args.pred_len):, :].detach()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(
                        -1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(
                        -1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae)

                else:
                    loss = criterion(pred, true).item()
                    total_loss.append(loss)

        for idx in range(len(self.models)):
            self.models[idx].train()

        total_loss = np.average(total_loss)

        return total_loss

    def train(self, setting):
        # teacher forcing training for each model
        for i in range(len(self.args.interval_list)):
            self.train_tf(setting, i)
        # joint training
        self.train_joint(setting)
        return self.models

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(self.args, flag='test')

        if test:
            print('loading model')
            for idx, model in enumerate(self.models):
                path = os.path.join(self.args_list[idx].checkpoints, setting)
                best_model_path = path + '/' + 'checkpoint.pth'
                self.models[idx].load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for idx in range(len(self.models)):
            self.models[idx].eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                input_temp = {}
                output_temp = {}
                concat_temp = {}

                target_temp = {}
                input_mark_temp = {}
                target_mark_temp = {}

                for idx, interval in enumerate(self.args.interval_list):
                    input_temp[idx], target_temp[idx], input_mark_temp[
                        idx], target_mark_temp[idx] = (
                            batch_x[:, -(self.args.seq_len_list[idx] *
                                         interval)::interval, :].float().to(
                                             self.device),
                            # batch_y[:, -self.args.pred_len::interval, :].float().to(self.device),
                            batch_y[:,
                                    -(self.args.pred_len + self.args.label_len
                                      )::interval, :].float().to(self.device),
                            # batch_x_mark[:, -(self.args.seq_len_list[idx] * interval)::interval, :].float().to(self.device),
                            batch_x_mark[:, -self.args_list[idx].
                                         seq_len::interval, :].float().to(
                                             self.device),
                            # batch_y_mark[:, -self.args.pred_len::interval, :].float().to(self.device),
                            batch_y_mark[:,
                                         -(self.args.pred_len +
                                           self.args.label_len)::interval, :].
                            float().to(self.device),
                        )

                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        input_mark_temp[idx] = None
                        target_mark_temp[idx] = None

                    if idx == 0 and self.args.use_dec_inp:
                        dec_inp = torch.zeros_like(
                            target_temp[idx]
                            [:,
                             -self.args_list[idx].true_pred_len:, :]).float()
                        dec_inp = torch.cat([
                            target_temp[idx][:, :self.args_list[idx].
                                             true_label_len, :], dec_inp
                        ],
                                            dim=1).float().to(self.device)
                    else:
                        dec_inp = None

                    if idx == 0:
                        # output_temp[idx] = self.models[idx](input_temp[idx], input_mark_temp[idx], dec_inp, target_mark_temp[idx])
                        output_temp[idx], aux = self._forward_any(
                            self.models[idx], input_temp[idx],
                            input_mark_temp[idx], dec_inp,
                            target_mark_temp[idx])
                    else:
                        B, T, N = input_temp[idx].shape
                        mask = self.mask[idx].to(self.device)
                        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(B, 1, N)

                        concat_temp[idx] = torch.zeros(
                            (B, self.args.pred_len, N)).to(self.device)

                        for temp_idx, interval_temp in enumerate(range(idx)):
                            concat_temp[idx][:, ::self.args.interval_list[
                                temp_idx], :] = output_temp[temp_idx]
                        input_temp[idx] = torch.cat(
                            (input_temp[idx], concat_temp[idx]
                             [:, ::self.args.interval_list[idx], :]),
                            dim=1)  # * mask[idx]
                        # output_temp[idx] = self.models[idx](input_temp[idx], input_mark_temp[idx], None, None, mask)
                        output_temp[idx], aux = self._forward_any(
                            self.models[idx], input_temp[idx],
                            input_mark_temp[idx], None, None, mask)

                    output_temp[idx] = output_temp[idx][:,
                                                        -self.args_list[idx].
                                                        true_pred_len:, :]
                    # output_temp[idx] = output_temp[idx][:, -self.new_args_list[idx].pred_len:, :]

                outputs = output_temp[max(output_temp.keys())]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(
                            outputs,
                            [1, 1,
                             int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1],
                                        -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1],
                                        -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(
                B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(
                B, T, C)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if self.args.data == 'PEMS':
            f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
        else:
            f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        # Retain the selected checkpoints for inspection and later evaluation.
        return
