# Sacred 让配置可组合：默认 config + named_config + 命令行覆盖。
from sacred import Experiment

# 实验名会出现在 Sacred 日志和运行入口。
ex = Experiment("VLMo")


def _loss_names(d):
    """建立所有任务损失开关，未显式开启的任务权重默认为 0。"""
    # 0 表示本阶段不训练该目标；正数表示在 forward 中启用。
    ret = {
        "itm": 0, # 图文匹配：融合后判断一对图文是否正确。
        "itc": 0, # 图文对比：全局向量双向检索。
        "mlm": 0, # 图文联合遮词：用图像和文字上下文补词。
        "textmlm": 0, # 纯文字遮词：语言预训练阶段。
        "vqa": 0, # 视觉问答：给图像和问题预测答案。
        "nlvr2": 0, # 两张图加一句话的真假推理。
        "irtr": 0, # 图文检索下游微调。
    }
    # 用 d 覆盖需要启用的任务，例如 {"itm":1,"itc":1,"mlm":1}。
    ret.update(d)
    return ret


@ex.config
def config():
    """全部任务共享的默认值；具体任务用下方 named_config 覆盖。"""
    # 实验名和随机种子用于日志命名及复现。
    exp_name = "vlmo"
    seed = 1
    # 默认图文预训练使用四个数据来源；真正文件位于 data_root。
    datasets = ["coco", "vg", "sbu", "gcc"]
    # 默认同时启用 ITM、ITC、MLM 三个目标。
    loss_names = _loss_names({"itm": 1, "itc": 1, "mlm": 1})
    # 目标总 batch，单卡实际 batch 可更小，再由梯度累积达到此数。
    batch_size = 1024  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    # 训练采用随机增强，验证采用固定缩放。
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    # 图片缩放到 224×224，patch16 会产生 14×14 个图像块。
    image_size = 224
    draw_false_image = 0
    image_only = False
    text_only = False

    # Text Setting
    vqav2_label_size = 3129
    # 句子统一到 40 个 token；旧初始化权重可能采用更长的 196。
    max_text_len = 40
    max_text_len_of_initckpt = 196
    # 英文 BERT 分词器及其词表大小。
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    # 遮盖约 15% 的词作为 MLM 预测目标。
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    # timm 注册名；对应 multiway_transformer.py 中的工厂函数。
    model_arch = "vlmo_base_patch16"
    # 越深的层越可能随机跳过残差分支。
    drop_path_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    # AdamW 学习率、权重衰减与 warmup/衰减设置。
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 200000
    warmup_steps = 0.1
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    # 下游检索评估时才需要 Recall 和 rerank 指标。
    get_recall_metric = False
    get_recall_rerank_metric = False
    k_test = 32

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    use_sharded_training = False
    resume_during_training = False

    # below params varies with the environment
    # 环境相关路径和 GPU 配置要由实际运行者填写。
    data_root = ""
    log_dir = "result"
    # 单卡每步 4 张，与上面总 batch 1024 是不同概念。
    per_gpu_batchsize = 4  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


# ----------------------- language pretraining config -----------------------


@ex.named_config
def task_textmlm_base():
    """第二阶段纯文字预训练：只开启 textmlm，输入 Wikibooks 文本。"""
    exp_name = "textmlm_base"
    datasets = ["wikibk"]
    loss_names = _loss_names({"textmlm": 1})
    batch_size = 1024
    max_text_len = 196
    learning_rate = 2e-4
    whole_word_masking = True
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_textmlm_base_plus():
    """与上面相同的文字任务，但使用 VLMo Base+ 骨干。"""
    exp_name = "textmlm_base_plus"
    datasets = ["wikibk"]
    loss_names = _loss_names({"textmlm": 1})
    batch_size = 1024
    max_text_len = 196
    learning_rate = 2e-4
    whole_word_masking = True
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_plus_patch16"


# ----------------------- vision-language pretraining config -----------------------


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm_itc_base():
    """第三阶段图文预训练：MLM、ITM、ITC 三目标同时启用。"""
    exp_name = "mlm_itm_itc_base"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = 1024
    whole_word_masking = True
    learning_rate = 2e-4
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_mlm_itm_itc_base_plus():
    """三目标图文预训练，骨干换成 Base+。"""
    exp_name = "mlm_itm_itc_base_plus"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = 1024
    whole_word_masking = True
    learning_rate = 1e-4
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_mlm_itm_itc_large():
    """三目标图文预训练，大型骨干和较小学习率。"""
    exp_name = "mlm_itm_itc_large"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = 1024
    whole_word_masking = True
    learning_rate = 5e-5
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vit_large_patch16_224"


# ----------------------- NLVR2 fine-tuning configs -----------------------


@ex.named_config
def task_finetune_nlvr2_base():
    """NLVR2：Base、224 图、两图一文真假分类。"""
    exp_name = "finetune_nlvr2_base"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 5e-5
    val_transform_keys = ["square_transform"]
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_nlvr2_base_plus():
    """NLVR2：换 Base+ 骨干，学习率与随机深度略有调整。"""
    exp_name = "finetune_nlvr2_base_plus"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.2
    val_transform_keys = ["square_transform"]
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_nlvr2_base_image384():
    """NLVR2：Base，图片提高到 384×384，因此位置向量需适配。"""
    exp_name = "finetune_nlvr2_base_image384"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 5e-5
    val_transform_keys = ["square_transform"]
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_nlvr2_base_plus_image384():
    """NLVR2：Base+，384×384 图片。"""
    exp_name = "finetune_nlvr2_base_plus_image384"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.2
    val_transform_keys = ["square_transform"]
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_nlvr2_large():
    """NLVR2：Large、224 图，使用较小学习率。"""
    exp_name = "finetune_nlvr2_large"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


@ex.named_config
def task_finetune_nlvr2_large_image384():
    """NLVR2：Large、384 图。"""
    exp_name = "finetune_nlvr2_large_image384"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- VQAv2 Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_vqa_base_image480():
    """VQAv2：Base、480 图，答案头学习率放大 20 倍。"""
    exp_name = "finetune_vqa_base_image480"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    lr_mult = 20
    image_size = 480
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_vqa_base_plus_image480():
    """VQAv2：Base+、480 图，其余设置与上一组接近。"""
    exp_name = "finetune_vqa_base_plus_image480"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    lr_mult = 20
    image_size = 480
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_vqa_large_image480():
    """VQAv2：Large、480 图，骨干学习率更低。"""
    exp_name = "finetune_vqa_large_image480"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1.5e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    lr_mult = 20
    image_size = 480
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- F30K IR/TR Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_irtr_f30k_base():
    """Flickr30k 检索：Base，启用 irtr 和 Recall 指标。"""
    exp_name = "finetune_irtr_f30k_base"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.15
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_irtr_f30k_base_image384():
    """Flickr30k 检索：Base，图片 384×384。"""
    exp_name = "finetune_irtr_f30k_base_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.15
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_irtr_f30k_base_plus_image384():
    """Flickr30k 检索：Base+，图片 384×384。"""
    exp_name = "finetune_irtr_f30k_base_plus_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_irtr_f30k_large_image384():
    """Flickr30k 检索：Large，图片 384×384。"""
    exp_name = "finetune_irtr_f30k_large_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 2e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- COCO IR/TR Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_irtr_coco_base_image384():
    """COCO 检索：Base，384 图，比 F30K 使用更多训练步骤。"""
    exp_name = "finetune_irtr_coco_base_image384"
    datasets = ["coco"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 3000
    warmup_steps = 300
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_irtr_coco_base_plus_image384():
    """COCO 检索：Base+，384 图。"""
    exp_name = "finetune_irtr_coco_base_plus_image384"
    datasets = ["coco"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 3000
    warmup_steps = 300
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_irtr_coco_large_image384():
    """COCO 检索：Large，384 图。"""
    exp_name = "finetune_irtr_coco_large_image384"
    datasets = ["coco"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 3000
    warmup_steps = 300
    get_recall_metric = True
    learning_rate = 2e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- Other configs -----------------------


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end
@ex.named_config
def step1_5k():
    """覆盖训练计划为 1500 步，前 150 步 warmup。"""
    max_epoch = 100
    warmup_steps = 150
    max_steps = 1500


@ex.named_config
def step3k():
    """覆盖训练计划为 3000 步，前 300 步 warmup。"""
    max_epoch = 100
    warmup_steps = 300
    max_steps = 3000


@ex.named_config
def step200k():
    """图文预训练常用 20 万步计划。"""
    max_epoch = 200
    warmup_steps = 2500
    max_steps = 200000


@ex.named_config
def step500k():
    """更长的 50 万步计划。"""
    max_epoch = 500
    warmup_steps = 2500
    max_steps = 500000
