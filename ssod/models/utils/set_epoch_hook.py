from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class RoiEpochSetHook(Hook):
    """Data-loading sampler for distributed training.
    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def before_train_iter(self, runner):
        if hasattr(runner.model.module.student.roi_head.bbox_head, 'set_iter'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.model.module.student.roi_head.bbox_head.set_iter(runner.iter)
        if hasattr(runner.model.module.teacher.roi_head.bbox_head, 'set_iter'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.model.module.teacher.roi_head.bbox_head.set_iter(runner.iter)

@HOOKS.register_module()
class StudentRoiWarmEpochSetHook(Hook):
    """Data-loading sampler for distributed training.
    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """
    def before_run(self, runner):
        if hasattr(runner.model.module.student.roi_head.bbox_head, 'set_warm_iter'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.model.module.student.roi_head.bbox_head.set_warm_iter('student')

@HOOKS.register_module()
class TeacherRoiWarmEpochSetHook(Hook):
    """Data-loading sampler for distributed training.
    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def before_run(self, runner):
        if hasattr(runner.model.module.teacher.roi_head.bbox_head, 'set_warm_iter'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.model.module.teacher.roi_head.bbox_head.set_warm_iter('teacher')
