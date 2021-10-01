_base_ = "base.py"

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="VOCDataset",
            ann_file="C:/Users/Alex/WorkSpace/dataset/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt",
            img_prefix="C:/Users/Alex/WorkSpace/dataset/VOCdevkit/VOC2007/",
        ),
        unsup=dict(
            type="VOCDataset",
            ann_file="C:/Users/Alex/WorkSpace/dataset/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt",
            img_prefix="C:/Users/Alex/WorkSpace/dataset/VOCdevkit/VOC2012/",
        ),
    ),
    val=dict(
        type="VOCDataset",
        ann_file="C:/Users/Alex/WorkSpace/dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt",
        img_prefix="C:/Users/Alex/WorkSpace/dataset/VOCdevkit/VOC2007/"
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

work_dir = "work_dirs/${cfg_name}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    # fold="${fold}",
                    # percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
