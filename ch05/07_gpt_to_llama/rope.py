import torch


def get_rotary_matrix(seq_len, dim, base=10000):
    """生成RoPE的旋转矩阵"""
    # 生成不同频率的正弦和余弦值
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # shape为[dim//2]
    # 生成位置索引
    position = torch.arange(seq_len).float()  # shape为[seq_len]
    # 计算每个位置和维度对应的角度
    theta = torch.outer(position, theta)  # 计算外积，其中第(i, j)个元素是position[i] * theta[j]；shape为[seq_len, dim//2]
    # 计算正弦和余弦值
    cos = torch.cos(theta)  # shape为[seq_len, dim//2]
    sin = torch.sin(theta)  # shape为[seq_len, dim//2]
    return cos, sin


# NTK缩放版本的RoPE旋转矩阵
def get_ntk_rotary_matrix(seq_len, dim, base=10000, scaling_factor=1.0):
    """NTK缩放版本的RoPE旋转矩阵"""
    # 应用缩放因子
    effective_base = base * (scaling_factor ** (dim / (dim - 2)))
    
    # 生成不同频率的基础角度
    theta = 1.0 / (effective_base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 生成位置索引
    position = torch.arange(seq_len).float()
    
    # 计算每个位置和维度对应的角度
    theta = torch.outer(position, theta)
    
    # 计算正弦和余弦值
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin


def apply_rotary_embedding(x, cos, sin):
    """应用旋转位置编码"""
    # 假设x的形状为[batch_size, seq_len, dim]
    # 将向量视为复数，每两个维度一组
    x_reshape = x.view(*x.shape[:-1], -1, 2)  # shape为[batch_size, seq_len, dim//2, 2]，即沿着特征维度拆分
    
    # 构建正弦和余弦矩阵，使其与x_reshape形状匹配
    cos_expanded = cos.view(1, cos.shape[0], cos.shape[1], 1)  # shape为[1, seq_len, dim//2, 1]
    sin_expanded = sin.view(1, sin.shape[0], sin.shape[1], 1)  # shape为[1, seq_len, dim//2, 1]
    
    # 旋转操作（复数乘法）
    # [x_real, x_imag] * (cos + i*sin) = [x_real*cos - x_imag*sin, x_real*sin + x_imag*cos]
    x_out_1 = x_reshape[:, :, :, 0:1] * cos_expanded - x_reshape[:, :, :, 1:2] * sin_expanded
    x_out_2 = x_reshape[:, :, :, 0:1] * sin_expanded + x_reshape[:, :, :, 1:2] * cos_expanded
    
    # 合并结果
    x_out = torch.cat([x_out_1, x_out_2], dim=-1)  # shape为[batch_size, seq_len, dim//2, 2]
    return x_out.view(*x.shape)


# 示例用法
def apply_rope(x):
    """对输入向量应用RoPE位置编码"""
    batch_size, seq_len, dim = x.shape
    cos, sin = get_rotary_matrix(seq_len, dim)
    return apply_rotary_embedding(x, cos, sin)


# 测试代码
if __name__ == "__main__":
    # 创建一个随机输入张量
    batch_size, seq_len, dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, dim)
    
    # 应用RoPE
    x_with_rope = apply_rope(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {x_with_rope.shape}")

    # 使用示例
    scaling_factor = 2.0  # 适用于将上下文长度扩展为原来的2倍
    extended_cos, extended_sin = get_ntk_rotary_matrix(
        seq_len=4096,  # 扩展后的序列长度
        dim=512,       # 向量维度
        scaling_factor=scaling_factor
    )