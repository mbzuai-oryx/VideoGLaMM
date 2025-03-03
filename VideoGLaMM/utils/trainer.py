import time
from tqdm import tqdm
import torch
import os
import shutil
from datetime import datetime
from utils.utils import (AverageMeter, ProgressMeter, Summary, dict_to_cuda,intersectionAndUnionGPU)
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def get_ds_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    return ds_config



class LISATrainer:
    def __init__(self, model_engine, train_loader, scheduler, args, exp_name=None) -> None:
        self.args = args
        # save dirs
        if not exp_name:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Tensorboard log dir
        self.log_dir = os.path.join(args.logs_base_dir, exp_name)
        if args.local_rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpoints save dir
        self.ckpt_save_dir = os.path.join(args.ckpt_base_dir, exp_name)
        if args.local_rank == 0:
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
        
        # model, dataloaders, etc
        self.model_engine = model_engine
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.start_epoch = 0
        
        # init metrics
        self.best_score, self.cur_ciou = 0.0, 0.0

        
    def resume(self, resume_dir):
        ### Resume from deepspeed checkpoint
        load_path, client_state = self.model_engine.load_checkpoint(resume_dir)
        with open(os.path.join(resume_dir, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        self.start_epoch = (int(ckpt_dir.replace("global_step", "")) // self.args.steps_per_epoch)
        print("resume training from {}, start from epoch {}".format(resume_dir, self.start_epoch))


    def __train_epoch(self, train_loader, model, epoch, scheduler, writer, train_iter, args):
        """Main training loop."""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        ce_losses = AverageMeter("CeLoss", ":.4f")
        mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
        mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
        mask_losses = AverageMeter("MaskLoss", ":.4f")

        progress = ProgressMeter(
            args.steps_per_epoch,
            [
                batch_time,
                losses,
                ce_losses,
                mask_losses,
                mask_bce_losses,
                mask_dice_losses,
            ],
            prefix="Epoch: [{}]".format(epoch),
        ) 

        # switch to train mode
        model.train()
        end = time.time()
        
        for global_step in tqdm(range(args.steps_per_epoch)):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)
                    
                print('...', end='')

                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                if args.precision == "fp16":
                    input_dict["images_for_sam"]  = [a.half() for a in input_dict["images_for_sam"]]
                    input_dict["images"] = [a.half() if a is not None else None for a in input_dict["images"]]
                    input_dict["context_images"] = [a.half() if a is not None else None for a in input_dict["context_images"]]
                elif args.precision == "bf16":
                    input_dict["images_for_sam"]  = [a.bfloat16() for a in input_dict["images_for_sam"]]
                    input_dict["images"] = [a.bfloat16() if a is not None else None for a in input_dict["images"]]
                    input_dict["context_images"] = [a.bfloat16() if a is not None else None for a in input_dict["context_images"]]
                else:
                    input_dict["images_for_sam"]  = [a.float() for a in input_dict["images_for_sam"]]
                    input_dict["images"] = [a.float() if a is not None else None for a in input_dict["images"]]
                    input_dict["context_images"] = [a.float() if a is not None else None for a in input_dict["context_images"]]
                    

                output_dict = model(**input_dict)
                

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                mask_bce_loss = output_dict["mask_bce_loss"]
                mask_dice_loss = output_dict["mask_dice_loss"]
                mask_loss = output_dict["mask_loss"]

                # batch_size = input_dict["images_for_sam"].size(0)
                batch_size = len(input_dict["images_for_sam"])
                losses.update(loss.item(), batch_size)
                ce_losses.update(ce_loss.item(), batch_size)
                mask_bce_losses.update(mask_bce_loss.item(), batch_size)
                mask_dice_losses.update(mask_dice_loss.item(), batch_size)
                mask_losses.update(mask_loss.item(), batch_size)
                model.backward(loss)
                
                # Ensure all CUDA operations complete
                # torch.cuda.synchronize() #NOTE for debugging only
        
                model.step()
                
            # torch.distributed.barrier()  # Synchronize after each epoch #NOTE for debugging only

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if global_step % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()

                    losses.all_reduce()
                    ce_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()

                if args.local_rank == 0:
                    progress.display(global_step + 1)
                    global_step_tb = epoch * args.steps_per_epoch + global_step
                    writer.add_scalar("train/loss", losses.avg, global_step_tb)
                    writer.add_scalar("train/ce_loss", ce_losses.avg, global_step_tb)
                    writer.add_scalar(
                        "train/mask_bce_loss", mask_bce_losses.avg, global_step_tb
                    )
                    writer.add_scalar(
                        "train/mask_dice_loss", mask_dice_losses.avg, global_step_tb
                    )
                    writer.add_scalar("train/mask_loss", mask_losses.avg, global_step_tb)
                    writer.add_scalar(
                        "metrics/total_secs_per_batch", batch_time.avg, global_step_tb
                    )
                    writer.add_scalar(
                        "metrics/data_secs_per_batch", data_time.avg, global_step_tb
                    )

                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

            if global_step != 0:
                try:
                    curr_lr = scheduler.get_last_lr()
                    if args.local_rank == 0:
                        global_step_tb = epoch * args.steps_per_epoch + global_step
                        writer.add_scalar("train/lr", curr_lr[0], global_step_tb)
                except:
                    pass

        return train_iter


    def train(self, epochs):            
        # training loop
        train_iter = iter(self.train_loader)
        for epoch in range(self.start_epoch, epochs):
            print(f"Starting epoch-{epoch} ...")
            # train for one epoch
            train_iter = self.__train_epoch(self.train_loader, self.model_engine, epoch, self.scheduler, self.writer, train_iter, self.args)

            # if not args.no_eval:
                # giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
                # is_best = giou > self.best_score
                # self.best_score = max(giou, self.best_score)
                # self.cur_ciou = ciou if is_best else self.cur_ciou

            # if args.no_eval or is_best:
                # if self.args.local_rank == 0:
                #     torch.save(
                #         {"epoch": epoch}, 
                #         os.path.join( self.log_dir, "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(self.best_score, self.cur_ciou) ),)
                #     try:
                #         if os.path.exists(self.ckpt_save_dir):
                #             shutil.rmtree(self.ckpt_save_dir)
                #     except:
                #         pass
                # torch.distributed.barrier()
                # self.model_engine.save_checkpoint(self.ckpt_save_dir)
                
            # if self.args.local_rank == 0:
            #     torch.save(
            #         {"epoch": epoch}, 
            #         os.path.join( self.log_dir, "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(self.best_score, self.cur_ciou) ),)
            #     try:
            #         if os.path.exists(self.ckpt_save_dir):
            #             shutil.rmtree(self.ckpt_save_dir)
            #     except:
            #         pass

            # torch.distributed.barrier()
            # self.model_engine.save_checkpoint(self.ckpt_save_dir)
            
             # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
            # save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
            
            # save_dir_name = "ckpt_model_last_epoch"
            # save_dir = os.path.join(self.log_dir, save_dir_name)
            save_dir = self.ckpt_save_dir
            # Ensure the directory exists
            if self.args.local_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                ckpt_filename = f"epoch_{epoch}.pth"
                torch.save({"epoch": epoch}, os.path.join(save_dir, ckpt_filename))
            torch.distributed.barrier()
            self.model_engine.save_checkpoint(save_dir)
    

class LISAValidator:
    def __init__(self, model_engine=None, val_loader=None, args=None, exp_name=None) -> None:
        self.args = args
        # save dirs
        if not exp_name:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Tensorboard log dir
        self.log_dir = os.path.join(args.logs_base_dir, 'logs', exp_name)
        if args.local_rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # model, dataloaders, etc
        self.model_engine = model_engine
        
        self.val_loader = val_loader
        
        # init metrics
        self.best_score, self.cur_ciou = 0.0, 0.0        


    def validate_on_reasonseg(self, model_engine=None, val_loader=None, args=None, writer=None, epoch=0):
        # giou, ciou = self.__validate(self.val_loader, self.model_engine, 0, self.writer, self.args)
        
        if not model_engine:
            model_engine = self.model_engine
        if not val_loader:
            val_loader = self.val_loader
        if not args:
            args =  self.args
        
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

        model_engine.eval()

        for input_dict in tqdm(val_loader):
            torch.cuda.empty_cache()

            input_dict = dict_to_cuda(input_dict)
            
            if args.precision == "fp16":
                input_dict["images_for_sam"]  = [a.half() for a in input_dict["images_for_sam"]]
                input_dict["images"] = [a.half() if a is not None else None for a in input_dict["images"]]
                input_dict["context_images"] = [a.half() if a is not None else None for a in input_dict["context_images"]]
            elif args.precision == "bf16":
                input_dict["images_for_sam"]  = [a.bfloat16() for a in input_dict["images_for_sam"]]
                input_dict["images"] = [a.bfloat16() if a is not None else None for a in input_dict["images"]]
                input_dict["context_images"] = [a.bfloat16() if a is not None else None for a in input_dict["context_images"]]
            else:
                input_dict["images_for_sam"]  = [a.float() for a in input_dict["images_for_sam"]]
                input_dict["images"] = [a.float() if a is not None else None for a in input_dict["images"]]
                input_dict["context_images"] = [a.float() if a is not None else None for a in input_dict["context_images"]]

            with torch.no_grad():
                output_dict = model_engine(**input_dict)

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection), union_meter.update(
                union
            ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        if args.local_rank == 0:
            if writer:
                writer.add_scalar("val/reason_seg/giou", giou, epoch)
                writer.add_scalar("val/reason_seg/ciou", ciou, epoch)
                print("reason_seg:", "giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        print("reason_seg:", "giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        return giou, ciou
        
    def validate_on_mevis(self, model_engine=None, val_loader=None, args=None, writer=None, epoch=0, save_masks_for_bench=False):

        if not model_engine:
            model_engine = self.model_engine
        if not val_loader:
            val_loader = self.val_loader
        if not args:
            args =  self.args
                    
        if not save_masks_for_bench:
            intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
            union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
            acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

        model_engine.eval()

        for input_dict in tqdm(val_loader):
            torch.cuda.empty_cache()

            input_dict = dict_to_cuda(input_dict)
            
            if args.precision == "fp16":
                input_dict["images_for_sam"]  = [a.half() for a in input_dict["images_for_sam"]]
                input_dict["images"] = [a.half() if a is not None else None for a in input_dict["images"]]
                input_dict["context_images"] = [a.half() if a is not None else None for a in input_dict["context_images"]]
            elif args.precision == "bf16":
                input_dict["images_for_sam"]  = [a.bfloat16() for a in input_dict["images_for_sam"]]
                input_dict["images"] = [a.bfloat16() if a is not None else None for a in input_dict["images"]]
                input_dict["context_images"] = [a.bfloat16() if a is not None else None for a in input_dict["context_images"]]
            else:
                input_dict["images_for_sam"]  = [a.float() for a in input_dict["images_for_sam"]]
                input_dict["images"] = [a.float() if a is not None else None for a in input_dict["images"]]
                input_dict["context_images"] = [a.float() if a is not None else None for a in input_dict["context_images"]]

            with torch.no_grad():
                output_dict = model_engine(**input_dict)

            if not save_masks_for_bench:
                masks_list = output_dict["gt_masks"]   # [batch, 1, 30, 360, 640]
                masks_list = masks_list[0].int()       # [1, 30, 360, 640]
                assert len(masks_list) == 1 
                masks_list = masks_list[0] # [30, 360, 640]
            
            pred_masks = output_dict["pred_masks"] # pred_masks : batch x (30, torch.Size([1, 360, 640]))
            pred_masks = pred_masks [0] # (30, torch.Size([1, 360, 640]))
            pred_masks = torch.stack(pred_masks, dim=0).permute(1, 0, 2, 3) # [1, 30, 360, 640]
            output_list = (pred_masks > 0).int() # [1, 30, 360, 640]
            assert len(output_list) == 1 
            output_list = output_list[0] # # [30, 360, 640]
            
            if save_masks_for_bench:
                video_name, exp_id = input_dict['image_paths'][0]
            
                for i in range(output_list.shape[0]):
                    mevis_output_save_dir = os.path.join(self.log_dir, 'mevis_output', video_name, exp_id)
                    os.makedirs(mevis_output_save_dir, exist_ok=True)
                    output_path = os.path.join(mevis_output_save_dir, f"{i:05d}.png")
                    output_mask_i = output_list[i] # torch.Size([360, 640]
                    
                    mask_array = output_mask_i.cpu().numpy()
                    # Converting the numpy array to an unsigned integer 8-bit array
                    mask_array = (mask_array * 255).astype(np.uint8)
                    # Creating a PIL Image
                    mask_image = Image.fromarray(mask_array)
                    # Saving the image as PNG
                    mask_image.save(output_path)
            else:
                intersection, union, acc_iou = 0.0, 0.0, 0.0
                for mask_i, output_i in zip(masks_list, output_list):
                    # intersection_i, union_i, _ = intersectionAndUnionGPU_for_video(
                    intersection_i, union_i, _ = intersectionAndUnionGPU(
                        output_i.contiguous().clone(), mask_i.contiguous(), K=2, ignore_index=255
                    )
                    intersection += intersection_i
                    union += union_i
                    acc_iou += intersection_i / (union_i + 1e-5)
                    acc_iou[union_i == 0] += 1.0  # no-object target
                intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
                acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
                intersection_meter.update(intersection), union_meter.update(
                    union
                ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        if not save_masks_for_bench:
            intersection_meter.all_reduce()
            union_meter.all_reduce()
            acc_iou_meter.all_reduce()

            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            ciou = iou_class[1]
            giou = acc_iou_meter.avg[1]

            if args.local_rank == 0:
                if writer:
                    writer.add_scalar("val/giou", giou, epoch)
                    writer.add_scalar("val/ciou", ciou, epoch)
                    print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

            return giou, ciou
        
        return
    