import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='parsenet', choices=['parsenet'])
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--version', type=str, default='parsenet')

    # Training setting
    parser.add_argument('--total_step', type=int, default=20000)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--g_lr', type=float, default=0.0005)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealing')
    parser.add_argument('--weight_init_type', type=str, default='kaiming')
    # Testing setting
    parser.add_argument('--test_size', type=int, default=2824) 
    parser.add_argument('--model_name', type=str, default='model.pth')
    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)
    # Misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--parallel', type=str2bool, default=False)

    # Path
    parser.add_argument('--img_path', type=str, default='./data/train/images')
    parser.add_argument('--label_path', type=str, default='./data/train/masks')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--test_image_path', type=str, default='./data/val/images')
    parser.add_argument('--test_label_path', type=str, default='./data/val/predicted_masks')
    parser.add_argument('--test_color_label_path', type=str, default='./data/val/predicted_masks_color')
    parser.add_argument('--over_sampler', type=str2bool, default=False)

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--evaluate_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=float, default=5000)

    return parser.parse_args()
