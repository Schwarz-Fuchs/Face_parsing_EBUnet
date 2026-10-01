import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
import cv2
import PIL
from EBUnet import EBUNet_Enhanced, EBUNet
from utils import *
from PIL import Image
import math

def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    return transform

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    f = dir.split('/')[-1].split('_')[-1]
    print (dir, len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]))
    filenames = [f for f in os.listdir(dir) if f.endswith('.jpg') or f.endswith('.png')]
    for name in filenames:
        img_path = os.path.join(dir, name).replace('\\', '/')
        images.append(img_path)
    return images

class Tester(object):
    def __init__(self, config):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version).replace('\\','/')
        self.sample_path = os.path.join(config.sample_path, self.version).replace('\\','/')
        self.model_save_path = os.path.join(config.model_save_path, self.version).replace('\\','/')
        self.test_label_path = config.test_label_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        self.build_model()

    def test(self):
        transform = transformer(True, True, True, False, self.imsize) 
        test_paths = make_dataset(self.test_image_path)
        make_folder(self.test_label_path, '')
        make_folder(self.test_color_label_path, '')

        checkpoint=torch.load(os.path.join(self.model_save_path, self.model_name).replace('\\', '/'))
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.G.load_state_dict(state_dict)
        self.G.eval()

        batch_num = math.ceil(len(test_paths) / self.batch_size)

        with torch.no_grad():
            for i in range(batch_num):
                # 1. 获取当前 Batch 的路径切片
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_paths = test_paths[start_idx:end_idx]
                imgs = []
                for path in batch_paths:
                    img = transform(Image.open(path))
                    imgs.append(img)
                imgs = torch.stack(imgs).cuda()
                labels_predict = self.G(imgs)
                labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
                labels_predict_color = generate_label(labels_predict, self.imsize)
                # 2. 遍历当前 Batch 的路径，提取名称并保存
                for k, path in enumerate(batch_paths):
                    # 获取带后缀的文件名，例如 "cat_001.jpg"
                    full_name = os.path.basename(path)
                    # 获取不带后缀的文件名，例如 "cat_001"
                    name_without_ext = os.path.splitext(full_name)[0]
                    # 定义保存名称（保持原名，但统一后缀为 .png）
                    save_name = name_without_ext + '.png'
                    # 3. 执行保存
                    cv2.imwrite(os.path.join(self.test_label_path, save_name), labels_predict_plain[k])
                    save_image(labels_predict_color[k], os.path.join(self.test_color_label_path, save_name))

                print(f"Batch {i + 1}/{batch_num} done.")

    def find_best_filters(self, model_class, target_params_m=1.5, n_classes=19):
        """
        model_class: 你的 unet 类
        target_params_m: 参数上限（单位：M）
        """
        best_filters = []
        multiplier = 1.5
        step = 0.05
        print(f"Searching for best filters under {target_params_m}M...")
        while multiplier > 0.1:
            current_filters = []
            base_channels = 32
            for i in range(5):
                c = (base_channels * (2 ** i)) * multiplier
                c_aligned = max(16, int(round(c / 16.0) * 16))
                current_filters.append(c_aligned)
            try:
                tmp_model = model_class(n_classes=n_classes, filters=current_filters)
                params = sum(p.numel() for p in tmp_model.parameters()) / 1e6
                if params <= target_params_m:
                    print(f"Found! Multiplier: {multiplier:.2f} | Params: {params:.2f}M | Filters: {current_filters}")
                    return current_filters
            except Exception as e:
                pass
            multiplier -= step
        return [16, 32, 64, 128, 256]

    def build_model(self):
        #chosen_filters = self.find_best_filters(unet)
        # chosen_filters=[16, 32, 64, 128, 256]
        #print('chosen filters: {}'.format(chosen_filters))

        self.G = EBUNet_Enhanced(classes=19).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
        # print networks
        #print(self.G)
