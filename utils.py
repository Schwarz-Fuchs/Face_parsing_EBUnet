from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153), 
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)], 
                     dtype=np.uint8) 
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    #label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy

def generate_label(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))
                
    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)	

    return label_batch

def generate_label_plain(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        #pred = pred.reshape((1, 512, 512))
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        label_batch.append(p.numpy())
                
    label_batch = np.array(label_batch)

    return label_batch

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


# 为了和训练代码保持一致，建议使用相同的高斯模糊逻辑
def gaussian_blur_compat(img_tensor, sigma=3.0):
    # 计算 kernel_size，通常设为 3*sigma 的奇数
    k_size = int(sigma * 3) * 2 + 1
    # torchvision 的这个函数在旧版本中也存在
    return TF.gaussian_blur(img_tensor, [k_size, k_size], [sigma, sigma])


def denorm(x):
    """反标准化，用于可视化"""
    return x * 0.5 + 0.5


def check_frequency_swap(orig_path, style_path, save_path, imsize=512, sigma=3.0):
    # 1. 读取并预处理图片 ( convert('RGB') 保证是3通道 )
    orig_img = Image.open(orig_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        # 模拟训练时的标准化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    orig_tensor = transform(orig_img).unsqueeze(0)  # [1, 3, H, W]
    style_tensor = transform(style_img).unsqueeze(0)

    # 2. 提取特征
    low_pass_orig = gaussian_blur_compat(orig_tensor, sigma)
    high_pass_orig = orig_tensor - low_pass_orig

    low_pass_style = gaussian_blur_compat(style_tensor, sigma)
    # high_pass_style = style_tensor - low_pass_style # 这里暂时不用

    # 3. 交换低频并合成新图
    mixed_tensor = high_pass_orig + low_pass_style
    # 限制在合法像素范围内 (-1 到 1，因为是标准化后的)
    mixed_tensor = torch.clamp(mixed_tensor, -1.0, 1.0)

    # 4. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行：输入和中间特征
    axes[0, 0].imshow(denorm(orig_tensor[0]).permute(1, 2, 0).numpy())
    axes[0, 0].set_title("1. Original Image A (Contains Item)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denorm(style_tensor[0]).permute(1, 2, 0).numpy())
    axes[0, 1].set_title("2. Style Image B (Background)")
    axes[0, 1].axis('off')

    # 重点看这个：高通特征，应该只有项链、五官的轮廓
    # 为了清晰，这里使用绝对值并放大亮度
    hp_vis = torch.abs(high_pass_orig[0]).permute(1, 2, 0).numpy()
    hp_vis = np.clip(hp_vis * 5.0, 0, 1)  # 放大5倍细节
    axes[0, 2].imshow(hp_vis, cmap='gray')
    axes[0, 2].set_title("3. High-Pass A (Details - Amplified)")
    axes[0, 2].axis('off')

    # 第二行：低通特征和最终合成
    lp_a_vis = denorm(low_pass_orig[0]).permute(1, 2, 0).numpy()
    axes[1, 0].imshow(lp_a_vis)
    axes[1, 0].set_title("4. Low-Pass A (Orig Style)")
    axes[1, 0].axis('off')

    lp_b_vis = denorm(low_pass_style[0]).permute(1, 2, 0).numpy()
    axes[1, 1].imshow(lp_b_vis)
    axes[1, 1].set_title("5. Low-Pass B (Target Style)")
    axes[1, 1].axis('off')

    # 最终合成图
    final_vis = denorm(mixed_tensor[0]).permute(1, 2, 0).numpy()
    axes[1, 2].imshow(final_vis)
    axes[1, 2].set_title("6. Frequency Swap Result (A_HP + B_LP)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Check image saved at: {save_path}")


def get_rare_list(img_dir, label_dir, target_classes=[3, 14, 15, 16]):
    """
    遍历标签，提取包含目标类别的图片路径
    target_classes: 默认 8(ear_r), 9(ear_l), 16(neck_l)
    """
    rare_list = []

    # 获取所有标签文件名
    label_names = [f for f in os.listdir(label_dir) if f.endswith('.png')]

    print(f"开始扫描标签文件，共 {len(label_names)} 张...")

    for name in tqdm(label_names):
        label_path = os.path.join(label_dir, name)
        # 使用 PIL 加载并转为 numpy
        label = np.array(Image.open(label_path))

        # 检查当前标签中是否包含任何一个目标类别
        # np.isin 会返回一个布尔矩阵，只要有一个 True 就说明该图含有稀有类
        if np.isin(target_classes, label).any():
            # 换回对应的图片文件名 (假设是 .jpg)
            img_name = name.replace('.png', '.jpg')
            img_path = os.path.join(img_dir, img_name).replace('\\', '/')
            rare_list.append(img_path)

    print(f"\n扫描完成！")
    print(f"找到含有稀有类别的图片: {len(rare_list)} 张")

    return rare_list

if __name__=='__main__':
    # --- 使用示例 ---
    img_directory = './data/train/images'  # 修改为你的图片目录
    label_directory = './data/train/masks'  # 修改为你的掩码目录
    # 执行扫描
    my_rare_list = get_rare_list(img_directory, label_directory)
    # 打印前 5 个路径看看格式对不对
    if my_rare_list:
        print("前 5 个路径示例:", my_rare_list[:5])
