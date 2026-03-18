import os
import os.path as osp
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SaveBestValLossHook(Hook):
    def __init__(self,
                 key='loss_val',
                 rule='less',
                 ckpt_name='best_loss_val.pth',
                 out_dir=None,
                 save_optimizer=False,
                 strict=False,
                 verbose=True):
        self.key = key
        self.rule = rule
        self.ckpt_name = ckpt_name
        self.out_dir = out_dir
        self.save_optimizer = save_optimizer
        self.strict = strict
        self.verbose = verbose

        assert self.rule in ['less', 'greater']
        if self.rule == 'less':
            self.best_score = float('inf')
            self.is_better = lambda cur, best: cur < best
        else:
            self.best_score = -float('inf')
            self.is_better = lambda cur, best: cur > best

        self.best_epoch = -1
        self._val_metrics = []

    def before_val_epoch(self, runner):
        self._val_metrics = []
        if self.verbose:
            runner.logger.info(
                f'[{self.__class__.__name__}] start collecting "{self.key}" from runner.outputs["log_vars"]'
            )

    def after_val_iter(self, runner):
        outputs = getattr(runner, 'outputs', None)

        if not isinstance(outputs, dict):
            if self.strict:
                raise TypeError(
                    f'[{self.__class__.__name__}] runner.outputs is not dict: {type(outputs)}'
                )
            return

        log_vars = outputs.get('log_vars', None)
        if not isinstance(log_vars, dict):
            if self.strict:
                raise KeyError(
                    f'[{self.__class__.__name__}] "log_vars" not found in runner.outputs'
                )
            return

        if self.key not in log_vars:
            if self.strict:
                raise KeyError(
                    f'[{self.__class__.__name__}] "{self.key}" not found in runner.outputs["log_vars"], '
                    f'available keys: {list(log_vars.keys())}'
                )
            return

        cur_val = log_vars[self.key]
        try:
            cur_val = float(cur_val)
        except Exception:
            if self.strict:
                raise TypeError(
                    f'[{self.__class__.__name__}] "{self.key}" value cannot be converted to float: {cur_val}'
                )
            return

        self._val_metrics.append(cur_val)

    def after_val_epoch(self, runner):
        if len(self._val_metrics) == 0:
            runner.logger.warning(
                f'[{self.__class__.__name__}] no valid "{self.key}" collected in this val epoch.'
            )
            return

        cur_score = sum(self._val_metrics) / len(self._val_metrics)

        if self.verbose:
            runner.logger.info(
                f'[{self.__class__.__name__}] current averaged {self.key} = {cur_score:.6f}, '
                f'best = {self.best_score:.6f}'
            )

        if self.is_better(cur_score, self.best_score):
            old_best = self.best_score
            self.best_score = cur_score
            self.best_epoch = runner.epoch + 1

            out_dir = self.out_dir if self.out_dir is not None else runner.work_dir
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = osp.join(out_dir, self.ckpt_name)

            runner.save_checkpoint(
                out_dir,
                filename_tmpl=self.ckpt_name,
                save_optimizer=self.save_optimizer,
                create_symlink=False
            )

            runner.logger.info(
                f'[{self.__class__.__name__}] {self.key} improved from '
                f'{old_best:.6f} to {cur_score:.6f} at epoch {self.best_epoch}. '
                f'Saved best checkpoint to: {ckpt_path}'
            )