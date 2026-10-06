# 只用 PyTorch 构造属性 Tensor；这些常量不是自动学习的 embedding 参数。
import torch

# moe_type=路由类型；当前此查表函数只实现 attribute，不表示论文其他路由都在本文件完整实现。
def get_moe_embedding(moe_type):

    if moe_type == 'attribute':
        # 字典混合两类键：任务类型→input/target 子字典，共享候选集合名→直接 Tensor。
        # 特别注意：本地常量与论文表 2 数值存在差异，不要直接按表 2 解释每一位。
        Task_attribute = {
            # task input -- TASK_TYPE & data_type
            'image_classification': {
                "input":
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
            },
            'video_classification': {
                "input":
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
            },
            'text_mlm': {
                "input":
                torch.tensor([[0, 1, 0, 1, 0, 1, 0, 0]], dtype=torch.float),
            },
            'image_caption': {
                "input":
                torch.tensor(
                    [[1, 1, 0, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0, 1]],
                    dtype=torch.float)
            },
            'video_caption': {
                "input":
                torch.tensor(
                    [[1, 1, 0, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0, 1]],
                    dtype=torch.float)
            },
            'image_retrieval': {
                'input':
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
                'target':
                torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            },
            'video_retrieval': {
                'input':
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
                'target':
                torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            },
            'text_classification': {
                "input":
                torch.tensor([[0, 1, 0, 1, 0, 1, 0, 0]], dtype=torch.float),
            },
            

            # SHARED_TARGETS
            "ImageNet1k":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "ImageNet22k":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "MomentsInTime":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "Kinetics700":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "Kinetics400":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "Vocab_Word":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "CoLA-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "MNLI-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "MRPC-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "QNLI-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "QQP-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "RTE-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "SST-2-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
        }
        return Task_attribute
    else:
        raise NotImplementedError(f'please check MOE_TYPE {moe_type}')



# 按路由类型、任务类型、输入/目标侧查表；不是从图片内容算出属性。
def get_embed_with_task_type(moe_type: str, task_type: str, data_type: str):
    # 未启用路由时返回 None；调用方要与 gate 配置一致，不能给启用的属性 gate 缺失属性。
    if moe_type is None:
        return None
    embedding_dict = get_moe_embedding(moe_type)
    return embedding_dict[task_type][data_type]


# tagert 是原命名拼写，意为 target；共享候选例如 ImageNet1k、Vocab_Word。
def get_embed_with_shared_tagert_name(moe_type: str, set_name: str,):
    # 未启用路由时返回 None；调用方要与 gate 配置一致，不能给启用的属性 gate 缺失属性。
    if moe_type is None:
        return None
    embedding_dict = get_moe_embedding(moe_type)
    return embedding_dict[set_name]
