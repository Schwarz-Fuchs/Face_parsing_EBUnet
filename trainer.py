import time
import datetime
from EBUnet import EBUNet, EBUNet_Enhanced
from utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from loss_functions import *
from scipy.ndimage import label as label_cc
import matplotlib
import torchvision.transforms as T
import torch.nn.functional as F


matplotlib.use('Agg') # 强制使用非交互式后端
writer = SummaryWriter('runs/training')

class Trainer(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.train_loader, self.val_loader = data_loader
        # exact model and loss
        self.model = config.model
        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel
        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_scheduler=config.lr_scheduler
        self.weight_init_type = config.weight_init_type

        #self.lr_decay = config.lr_decay -- now using CosineAnnealing
        self.warmup_steps=config.warmup_steps
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
        self.evaluate_step=config.evaluate_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.weight_decay=config.weight_decay

        # Path
        self.log_path = os.path.join(config.log_path, self.version).replace('\\','/')
        self.log_path = os.path.join(self.log_path, 'log.txt').replace('\\','/')

        self.sample_path = os.path.join(config.sample_path, self.version).replace('\\','/')
        self.model_save_path = os.path.join(config.model_save_path, self.version).replace('\\','/')

        # init log file
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_checkpoint(self.pretrained_model)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_G.pth'.format(self.pretrained_model)).replace('\\','/')))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def init_weights(self, net):
        print(f"Initializing network with {self.weight_init_type}...")
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if self.weight_init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, self.warmup_steps))
        # end warmup
        return 1.0

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

        #chosen_filters=self.find_best_filters(unet)
        #chosen_filters=[8, 16, 32, 64, 128]
        #print('chosen filters: {}'.format(chosen_filters))
        #self.G = unet(filters=chosen_filters).cuda()

        self.G = EBUNet_Enhanced(classes=19).cuda()
        self.init_weights(self.G)

        if self.parallel:
            self.G = nn.DataParallel(self.G)

        self.g_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.G.parameters()),
            self.g_lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
        )

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=self.lr_lambda)
        # lr scheduler -- CosineAnnealing
        if self.lr_scheduler=="CosineAnnealing":
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, T_max=(self.total_step - self.warmup_steps))

        total_params = sum(p.numel() for p in self.G.parameters())
        print(f"Total parameters: {total_params:,}")
        if total_params > 1821085:
            print(f"Warning! Exceed {total_params - 1821085} parameters")
        else:
            print(f"Good! Within limit.")

    def calculate_class_weights(self, num_classes=19):
        weight_path = os.path.join(self.model_save_path, 'median_weights.npy')
        if os.path.exists(weight_path):
            print(f"Loading cached Median Frequency weights from {weight_path}...")
            weights = np.load(weight_path)
            return torch.FloatTensor(weights).cuda()

        print(f"No cache found. Calculating distribution of labels files for {num_classes} classes...")
        counts = np.zeros(num_classes)
        label_files = [f for f in os.listdir(self.label_path) if f.endswith('.png') or f.endswith('.jpg')]

        for file_name in tqdm(label_files):

            mask = np.array(Image.open(os.path.join(self.label_path, file_name).replace('\\', '/')))
            unique, pixel_counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, pixel_counts):
                if u < num_classes:
                    counts[u] += c
        # --- Median Frequency Balancing ---
        total_pixels = counts.sum()
        frequencies = counts / (total_pixels + 1e-10)
        nonzero_freqs = frequencies[frequencies > 0]
        if len(nonzero_freqs) > 0:
            median_freq = np.median(nonzero_freqs)
        else:
            median_freq = 1.0
        weights = median_freq / (frequencies + 1e-6)
        weights = np.clip(weights, 0.5, 4)
        ## manually settings here
        #weights[18] = 1.5 # cloth
        #weights[16] = 8.0 # necklace
        #weights[14] = 6.0  # hat
        #weights[15] = 6.0  # ear_r
        #weights[3] = 4.0  # eye_g

        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        np.save(weight_path, weights)
        print(f"Weights calculated and saved to {weight_path}")

        return torch.FloatTensor(weights).cuda()

    def get_ohem_stats(self, predict, target, thresh=0.7):
        """
        分析当前 Batch 中有多少比例的像素被判定为 'Hard'
        """
        with torch.no_grad():
            # 1. 展平并建立有效掩码
            target_flat = target.view(-1)
            valid_mask = (target_flat != -1)  # 假设 -1 是 ignore_index
            if not valid_mask.any():
                return 0.0
            # 2. 计算概率
            # 仅对有效像素提取概率，节省计算量
            prob = F.softmax(predict, dim=1).permute(0, 2, 3, 1).reshape(-1, predict.size(1))
            curr_prob = prob[valid_mask]
            # 3. 提取对应真实类别的预测概率 (安全 gather)
            target_valid = target_flat[valid_mask].unsqueeze(1)
            curr_prob = curr_prob.gather(1, target_valid).squeeze()
            # 4. 计算硬样本比例
            # curr_prob 已经是针对有效像素的了，直接计算小于 thresh 的比例
            hard_pixels = (curr_prob < thresh).sum().item()
            total_valid_pixels = valid_mask.sum().item()
            hard_ratio = hard_pixels / total_valid_pixels

        return hard_ratio

    def get_blobs(self, labels_indices):
        if labels_indices.dim() == 4:
            labels_indices = labels_indices.squeeze(1)
        B, H, W = labels_indices.shape
        device = labels_indices.device
        # 1. 在 CPU 上操作，完全不触碰显存
        labels_np = labels_indices.detach().cpu().numpy()
        final_blobs_np = np.zeros((B, H, W), dtype=np.int32)
        global_id_offset = 0
        for i in range(B):
            unique_classes = np.unique(labels_np[i])
            for cls_id in unique_classes:
                if cls_id == 0: continue
                class_mask = (labels_np[i] == cls_id)
                labeled_array, num_features = label_cc(class_mask)
                if num_features > 0:
                    mask = labeled_array > 0
                    final_blobs_np[i][mask] = labeled_array[mask] + global_id_offset
                    global_id_offset += num_features
        # 2. 一次性上传 GPU，大幅减少显存申请次数
        final_blobs = torch.from_numpy(final_blobs_np).to(device)
        return final_blobs.unsqueeze(1)

    @torch.no_grad()
    def calculate_fscore(self, preds, labels, num_classes=19, eps=1e-7):
        """
        preds: [B, 19, H, W]
        labels: [B, H, W]
        """
        predictions = torch.argmax(preds, dim=1)
        fscore_per_class = []
        for i in range(num_classes):
            pred_i = (predictions == i)
            label_i = (labels == i)
            tp = (pred_i & label_i).sum().float()
            fp = (pred_i & ~label_i).sum().float()
            fn = (~pred_i & label_i).sum().float()
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            fscore_per_class.append(f1.item())
        return fscore_per_class

    # cut mix
    def cutmix_data(self,inputs, masks, alpha=1.0):
        """
        inputs: [Batch, 3, H, W]
        masks:  [Batch, H, W]
        """
        indices = torch.randperm(inputs.size(0))
        shuffled_inputs = inputs[indices]
        shuffled_masks = masks[indices]
        # 从 Beta 分布中随机采样一个比例系数 lam (lambda)
        # alpha=1.0 时，Beta(1,1) 等同于均匀分布，决定了裁剪区域的大小
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
        # change img
        inputs[:, :, bbx1:bbx2, bby1:bby2] = shuffled_inputs[:, :, bbx1:bbx2, bby1:bby2]
        # change label
        masks[:, bbx1:bbx2, bby1:bby2] = shuffled_masks[: , bbx1:bbx2, bby1:bby2]
        return inputs, masks

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        # S_cut/S_original = (1 - lam)
        cut_rat = np.sqrt(1. - lam)
        # get cut_W and cut_H
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        # choose cut center randomly
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        # create cut box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def evaluate(self, val_loader):

        self.G.eval()
        total_fscores = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.cuda()
                labels_indices = labels.squeeze(1).long().cuda()
                preds = self.G(imgs)
                # calculate_fscore 返回的是一个长度为 19 的 list
                batch_fscores = self.calculate_fscore(preds, labels_indices)
                total_fscores.append(batch_fscores)
        return total_fscores

    def save_checkpoint(self, step, filename=None):
        if filename is None:
            filename = f'checkpoint_{step + 1}_steps.pth'
        save_path = os.path.join(self.model_save_path, filename).replace('\\', '/')
        # 处理 DataParallel 的情况，确保保存的是原始模型权重
        model_state = self.G.module.state_dict() if self.parallel else self.G.state_dict()
        state = {
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.g_optimizer.state_dict(),
            'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict() if self.warmup_scheduler else None,
            'base_scheduler_state_dict': self.base_scheduler.state_dict() if hasattr(self, 'base_scheduler') else None,
        }
        torch.save(state, save_path)
        print(f"[*] Checkpoint saved at step {step + 1} to {save_path}")

    def load_checkpoint(self, resume_step):
        resume_path = os.path.join(self.model_save_path, f'checkpoint_{resume_step}_steps.pth').replace('\\', '/')
        if not os.path.exists(resume_path):
            print(f"[!] No checkpoint found at {resume_path}")
            return 0
        print(f"[*] Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path)
        # 1. 加载模型权重 (支持 DataParallel)
        if self.parallel:
            self.G.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.G.load_state_dict(checkpoint['model_state_dict'])
        # 2. 加载优化器状态
        self.g_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 4. 加载调度器状态
        if checkpoint['warmup_scheduler_state_dict']:
            self.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
        if checkpoint['base_scheduler_state_dict'] and hasattr(self, 'base_scheduler'):
            self.base_scheduler.load_state_dict(checkpoint['base_scheduler_state_dict'])
        print(f"[*] Successfully resumed from step {checkpoint['step'] + 1}")
        return checkpoint['step'] + 1

    def train(self):
        data_iter = iter(self.train_loader)
        #save model base on training step
        model_save_step = int(self.model_save_step)
        #save model based on epoch
        #model_save_step = int(self.model_save_step * step_per_epoch)
        start = self.pretrained_model + 1 if self.pretrained_model else 0
        start_time = time.time()

        # 建议提前定义 Loss 函数
        # 可以给小类别（如眼睛 4,5, 嘴唇 12,13）分配更高权重
        weights_vector=self.calculate_class_weights()
        print("weights_vector =", weights_vector)
        criterion_WCE = nn.CrossEntropyLoss(weight=weights_vector).cuda()
        criterion_SDICE = MemoryEfficientSoftDiceLoss(n_classes=19, smooth=1e-6).cuda()
        criterion_OHEM = OHEMLoss(weight=weights_vector).cuda()
        #criterion_CONSENSUS = StructureConsensuLossFunction(consensus_loss_alpha=0.1,consensus_loss_beta=1).cuda()

        for step in range(start, self.total_step):
            self.G.train()
            try:
                imgs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                imgs, labels = next(data_iter)
            #print(imgs.shape) #([B, 3, 512, 512])
            #print(labels.shape) #([B, 512, 512])

            # apply random crop
            i, j, h, w = T.RandomResizedCrop.get_params(
                imgs, scale=(0.8, 1.0), ratio=(1.0, 1.0)
            )
            imgs = T.functional.crop(imgs, i, j, h, w)
            imgs = F.interpolate(imgs, size=(self.imsize, self.imsize), mode='bilinear', align_corners=False)

            labels = T.functional.crop(labels, i, j, h, w)
            labels = F.interpolate(labels.unsqueeze(1).float(), size=(self.imsize, self.imsize),
                                   mode='nearest').squeeze(1)

            #apply cut-mix
            if np.random.random() > 0.5:
                imgs, labels = self.cutmix_data(imgs, labels, alpha=1.0)

            labels_indices = (labels).long().cuda() #([B, 512, 512])
            #print(labels_indices.shape)
            #print(torch.unique(labels_indices)) #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
            imgs = imgs.cuda()
            # Forward


            # 2. generate blobs (CPU)
            with torch.no_grad():
                blobs = self.get_blobs(labels_indices) #([B, 1, 512, 512])
                '''
                #check blob map
                sample_blob = blobs[0].squeeze().cpu().numpy()
                plt.figure(figsize=(10, 5))
                plt.imshow(sample_blob, cmap='nipy_spectral')
                plt.title(f"Blobs Visualization (Unique IDs: {len(np.unique(sample_blob))})")
                plt.colorbar()
                plt.savefig('sample.png')
                plt.close()
                '''
            labels_predict = self.G(imgs)  # [B, 19, H, W]
            # disable cons_loss currently
            cons_loss=0

            progress = step / self.total_step
            ohem_start_ratio=0.3
            thresh=0.7

            if progress < ohem_start_ratio:
                # 阶段 1: 基础训练，使用标准 CrossEntropy
                current_ratio = 0.0
                c_loss = criterion_WCE(labels_predict, labels_indices)
                d_loss = criterion_SDICE(labels_predict, labels_indices)
                total_loss = c_loss + 0.1 * d_loss
            else:
                # 阶段 2: OHEM 训练
                # 动态 min_kept: 随着训练进行，保留比例从 0.8 逐渐降至 0.4
                # 公式: 随着 progress 增加，ratio 减小，模型越来越死磕最难的像素
                current_ratio = max(0.3, 0.8 - (progress - ohem_start_ratio))
                min_kept =  self.imsize* self.imsize*self.batch_size*current_ratio
                c_loss = criterion_OHEM(labels_predict, labels_indices, min_kept=min_kept,thresh=thresh)
                d_loss = criterion_SDICE(labels_predict, labels_indices)
                total_loss = c_loss + 0.5 * d_loss  # + 1.0 * cons_loss

            self.reset_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=5.0)
            self.g_optimizer.step()

            if step < self.warmup_steps:
                self.warmup_scheduler.step()
            elif self.lr_scheduler is not None:
                self.base_scheduler.step()

            if (step + 1) % self.log_step == 0:
                train_batch_fscores = self.calculate_fscore(labels_predict.float(), labels_indices)
                train_batch_macro = np.mean(train_batch_fscores)
                elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                curr_lr = self.g_optimizer.param_groups[0]['lr']

                OHEM_rate=self.get_ohem_stats(labels_predict,labels_indices,thresh=thresh)

                print(
                    f"Elapsed [{elapsed}], Step [{step + 1}/{self.total_step}], OHEM_STATUS:{current_ratio}, Hard_ratio: {OHEM_rate:.4f}, Loss: {total_loss.item():.4f}, CE_loss:{c_loss.item():.4f}, "
                    f"DICE_loss:{d_loss.item():.4f}, CONES_loss:{0:.4f}, LR: {curr_lr:.6f} ,Train_F1:{train_batch_macro:.4f}")
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"Elapsed [{elapsed}], Step [{step + 1}/{self.total_step}], OHEM_STATUS:{current_ratio}, Hard_ratio: {OHEM_rate:.4f}, Loss: {total_loss.item():.4f}, CE_loss:{c_loss.item():.4f}, "
                            f"DICE_loss:{d_loss.item():.4f},CONES_loss:{0:.4f}, LR: {curr_lr:.6f} ,Train_F1:{train_batch_macro:.4f}")
                    f.write(f'\n')
                writer.add_scalar('Loss/total', total_loss.item(), step+1)

            if (step + 1) % self.sample_step == 0:
                label_batch_predict = generate_label(labels_predict, self.imsize)
                writer.add_image('Result/Img', (imgs[0].data + 1) / 2.0, step)
                writer.add_image('Result/Predict', label_batch_predict[0], step)

            if (step + 1) % self.evaluate_step == 0:
                print("Starting evaluation...")

                class_names = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                               'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat',
                               'ear_r', 'neck_l', 'neck', 'cloth']

                total_fscores = self.evaluate(self.val_loader)
                mean_fscores = np.nanmean(np.array(total_fscores), axis=0)
                macro_f1 = np.nanmean(mean_fscores)
                print("\n" + "=" * 30)
                print(f"{'Class Name':<12} | {'F1-Score':<8}")
                print("-" * 30)
                for name, score in zip(class_names, mean_fscores):
                    print(f"{name:<12} | {score:.4f}")
                print("=" * 30)

                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"\nStep Evaluation Detailed F1:\n")
                    for name, score in zip(class_names, mean_fscores):
                        f.write(f"{name}: {score:.4f}, ")
                    f.write(f"\nMacro F1: {macro_f1:.4f}\n")

                print(f"Validation Macro F1-score: {macro_f1:.4f}")
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"Validation Macro F1-score: {macro_f1:.4f}")
                    f.write(f'\n')
                writer.add_scalar('macro_f1', macro_f1, step+1)

                for name, score in zip(class_names, mean_fscores):
                    writer.add_scalar(f'Per_Class_F1/{name}', score, step+1)

            ## there are no test data available currently, add test step if you got test label
            # save ckpt
            if (step + 1) % model_save_step == 0:
                self.save_checkpoint(step + 1, f'checkpoint_{step + 1}_steps.pth')
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"pth file saved!")
                    f.write(f'\n')

            del labels_predict, blobs, total_loss, c_loss, d_loss, cons_loss
