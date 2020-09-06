import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import torch
import time
import glob
import traceback
from ltr.admin import loading, multigpu


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update statistics
            batch_size = data['train_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)


class LTRDistillationTrainer(LTRTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.student_net.module if multigpu.is_multi_gpu(self.actor.student_net) else self.actor.student_net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }


        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)


    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.student_net.module if multigpu.is_multi_gpu(self.actor.student_net) else self.actor.student_net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = loading.torch_load_legacy(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        return True

