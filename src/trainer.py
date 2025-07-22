import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 🆕 matplotlib追加
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド設定
import numpy as np

# パス修正: 親ディレクトリからインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import DiceLoss

# 🆕 プロット生成関数を追加
def plot_training_progress(epochs, train_losses, val_losses, 
                          train_dice_losses, val_dice_losses,
                          train_ce_losses, val_ce_losses,
                          save_path, title="Training Progress"):
    """
    学習進捗をプロットしてPNGとして保存（nnU-Net風）
    """
    if len(epochs) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Total Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Loss
    axes[0, 1].plot(epochs, train_dice_losses, 'b-', label='Train Dice Loss', linewidth=2)
    axes[0, 1].plot(epochs, val_dice_losses, 'r-', label='Val Dice Loss', linewidth=2)
    axes[0, 1].set_title('Dice Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CrossEntropy Loss
    axes[1, 0].plot(epochs, train_ce_losses, 'b-', label='Train CE Loss', linewidth=2)
    axes[1, 0].plot(epochs, val_ce_losses, 'r-', label='Val CE Loss', linewidth=2)
    axes[1, 0].set_title('CrossEntropy Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CE Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学習状況サマリー
    if len(train_losses) > 0:
        recent_train_loss = train_losses[-1] if len(train_losses) > 0 else 0
        recent_val_loss = val_losses[-1] if len(val_losses) > 0 else 0
        best_val_loss = min(val_losses) if len(val_losses) > 0 else float('inf')
        
        summary_text = f"Current Status:\n"
        summary_text += f"Train Loss: {recent_train_loss:.4f}\n"
        summary_text += f"Val Loss: {recent_val_loss:.4f}\n"
        summary_text += f"Best Val Loss: {best_val_loss:.4f}\n"
        summary_text += f"Epoch: {epochs[-1] if epochs else 0}"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # メモリリーク防止
    
    print(f"📊 Progress plot saved: {save_path}")


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"


def trainer_cellmix(args, model, snapshot_path):
    
    from datasets.dataset_cellmix import CellMixDataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # 🆕 環境変数からaugmentation_modeを取得
    augmentation_mode = os.environ.get('AUGMENTATION_MODE', 'standard')
    
    # Data Augmentation設定
    db_train = CellMixDataset(
        base_dir=args.root_path, 
        list_dir=args.list_dir, 
        split="train",
        transform=RandomGenerator(
            output_size=[args.img_size, args.img_size],
            augmentation_mode=augmentation_mode  # 🆕 環境変数から取得
        )
    )
    
    # 🔧 検証用データセット（拡張なし/軽微）
    db_val = CellMixDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="val",
        transform=RandomGenerator(
            output_size=[args.img_size, args.img_size],
            augmentation_mode='standard'  # 🆕 検証は常に標準モード
        )
    )
    
    # 🆕 使用中のモードをログ出力
    logging.info(f"Data Augmentation mode: {augmentation_mode}")
    
    # 可視化用データリスト
    train_losses = []
    val_losses = []  
    train_dice_losses = []
    val_dice_losses = []
    train_ce_losses = []
    val_ce_losses = []
    epochs_list = []
    
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))
    
    # 🆕 DataLoader作成
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    # 🆕 モデル設定
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    # 🆕 損失関数とオプティマイザー
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # 🆕 TensorBoard
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    
    # 🆕 iterator定義
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10

    # 🆕 完全な学習ループ
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        
        # 学習フェーズ
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", 
                                          total=len(train_loader), leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        # 学習ログ記録
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        
        # 学習データを記録
        epochs_list.append(epoch_num + 1)
        train_losses.append(batch_loss)
        train_dice_losses.append(batch_dice_loss)
        train_ce_losses.append(batch_ce_loss)
        
        # バリデーション
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss_val = 0
            batch_ce_loss_val = 0
            
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                  total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss_val += loss_dice.item()
                    batch_ce_loss_val += loss_ce.item()
            
            batch_ce_loss_val /= len(val_loader)
            batch_dice_loss_val /= len(val_loader)
            batch_loss_val = 0.4 * batch_ce_loss_val + 0.6 * batch_dice_loss_val
            logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                epoch_num, batch_loss_val, batch_ce_loss_val, batch_dice_loss_val))
            
            # バリデーションデータを記録
            val_losses.append(batch_loss_val)
            val_dice_losses.append(batch_dice_loss_val)
            val_ce_losses.append(batch_ce_loss_val)
            
            # 🎨 進捗プロット生成
            plot_training_progress(
                epochs_list[-len(val_losses):],
                train_losses[-len(val_losses):], 
                val_losses, 
                train_dice_losses[-len(val_losses):], 
                val_dice_losses,
                train_ce_losses[-len(val_losses):], 
                val_ce_losses,
                save_path=os.path.join(snapshot_path, 'progress.png'),
                title=f'CellMix Training Progress - Epoch {epoch_num + 1}'
            )
            
            if batch_loss_val < best_loss:
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                best_loss = batch_loss_val
            else:
                save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
    
    writer.close()
    return "Training Finished!"
    