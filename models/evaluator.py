import os
import numpy as np
import matplotlib.pyplot as plt

from models.STRobustNet import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import cv2
import time
from torchstat import stat

# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# from ptflops import get_model_complexity_info
# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.net_G.vis_token = True
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.vis_token = True # args.vis_token
        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.version = args.net_G
        self.dataname = args.data_name
        self.project_name = args.project_name

        self.token_A = None
        self.token_B = None
        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(self.G_pred, dim=1, keepdim=True)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred, self.token_A, self.token_B = self.net_G(img_in1, img_in2)
        self.score_map = torch.softmax(self.G_pred, dim=1)[:, 1, :, :]
            # print(self.G_pred.max())

    def visualize_token(self, token_A, token_B, save_pathA, save_pathB):
        for i in range(token_A.shape[0]):
            heat = token_A[i].detach().cpu().numpy()
            # heat = 
            # print(imgL)
            heat_min = heat.min()
            heat_max = heat.max()
            heat = (heat-heat_min)/ (heat_max - heat_min)
            # heat = ((heat-u)/np.sqrt(s+1e-5)) + 1
            heatmap = cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET) # 利用色彩空间转换将heatmap凸显
            # heatmap = np.float32(heatmap) / 255 # 归一化
            # cam = heatmap + np.float32(np.array(L_im.cpu())) # 将heatmap 叠加到原图
            # cam = cam / np.max(cam)
            cv2.imwrite(save_pathA+'_'+str(i)+".png", np.uint8(heatmap)) # 生成图像

            heat = token_B[i].detach().cpu().numpy()
            heat_min = heat.min()
            heat_max = heat.max()
            heat = (heat-heat_min)/ (heat_max - heat_min)
            # print(imgL)
            heatmap = cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET) # 利用色彩空间转换将heatmap凸显
            # heatmap = np.float32(heatmap) / 255 # 归一化
            # cam = heatmap + np.float32(np.array(L_im.cpu())) # 将heatmap 叠加到原图
            # cam = cam / np.max(cam)
            cv2.imwrite(save_pathB+'_'+str(i)+".png", np.uint8(heatmap)) # 生成图像
    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        self.global_id = 0
        path = os.path.join("./vis", self.project_name)
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "pred"), exist_ok=True)
        os.makedirs(os.path.join(path, "bi_pred"), exist_ok=True)
        os.makedirs(os.path.join(path, "gt"), exist_ok=True)
        os.makedirs(os.path.join(path, "score_map"), exist_ok=True)
        os.makedirs(os.path.join(path, "A"), exist_ok=True)
        os.makedirs(os.path.join(path, "B"), exist_ok=True)
        if self.vis_token:
            os.makedirs(os.path.join(path, "tokenA"), exist_ok=True)
            os.makedirs(os.path.join(path, "tokenB"), exist_ok=True)
            os.makedirs(os.path.join(path, "ori_tokenA"), exist_ok=True)
            os.makedirs(os.path.join(path, "ori_tokenB"), exist_ok=True)

        num_params = sum(param.numel() for param in self.net_G.parameters())
        print("params:", num_params)
        # print("warm up")
        # dummy_input = torch.rand(1, 3, 256, 256).to(torch.device('cuda:0'))
        # with torch.no_grad():
        #     # time0 = time.time()
        #     for _ in range(2000):
        #         _ = self.net_G(dummy_input, dummy_input)
        #     # time1 = time.time()
        # print("finish warm up")

        
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # # # times = torch.zeros(len(self.dataloader)) # 存储每轮iteration的时间
        
        # avg_time = []
        # for i in range(5): # 测试轮数
        #     total_time = 0.0
        #     for self.batch_id, batch in enumerate(self.dataloader, 0):
        #         bs = batch['A'].shape[0]
        #         A = batch['A'].cpu().detach()
        #         B = batch['B'].cpu().detach()

        #         with torch.no_grad():
        #             
        #             self._forward_pass(batch)
        #             ender.record()
        #             torch.cuda.synchronize()
        #             curr_time = starter.elapsed_time(ender)
        #             total_time += curr_time

        #     print("total time:", total_time)
        #     print("avg_time:", total_time / 2048)
        #     avg_time.append(total_time / 2048)
        # print("avg all:", sum(avg_time) / len(avg_time))
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
        # tensor = torch.rand(1, 3, 256, 256).cuda()
        # flops = FlopCountAnalysis(self.net_G, (tensor, tensor))
       #  print(flop_count_table(flops))
        # print(parameter_count_table(self.net_G))
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            bs = batch['A'].shape[0]
            A = batch['A'].cpu().detach()
            B = batch['B'].cpu().detach()

            with torch.no_grad():
                self._forward_pass(batch)

        
                self._collect_running_batch_states()
                pred_mask = self._visualize_pred()
                self.score_map = self.score_map.cpu()
                gt = batch['L'] * 255
                for i in range(bs):
                    mask = pred_mask[i].permute(1,2,0).detach().cpu().numpy()
                    label = gt[i].permute(1,2,0).detach().cpu().numpy()
                    score_map = self.score_map[i]
                    output_a = cv2.convertScaleAbs(np.array(A[i]).transpose(1, 2, 0)[:, :, ::-1], alpha=(255.0))
                    output_b = cv2.convertScaleAbs(np.array(B[i]).transpose(1, 2, 0)[:, :, ::-1], alpha=(255.0))
                    # output_a
                    if self.vis_token:
                        save_pathA = os.path.join(path, "tokenA", str(self.global_id))
                        save_pathB = os.path.join(path, "tokenB", str(self.global_id))
                        save_path_oriA = os.path.join(path, "ori_tokenA", str(self.global_id))
                        save_path_oriB = os.path.join(path, "ori_tokenB", str(self.global_id))
                        self.visualize_token(self.token_A[i], self.token_B[i], save_pathA, save_pathB)
                        # self.visualize_token(self.ori_token_A[i], self.ori_token_B[i], save_path_oriA, save_path_oriB)


                    output_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
                    for i in range(mask.shape[0]):
                        for j in range(mask.shape[1]):
                            if mask[i][j][0] == label[i][j][0] and mask[i][j][0] == 255: # 预测正确 蓝色
                                output_mask[i][j][0] = 255
                            elif mask[i][j][0] == 0 and label[i][j][0] == 255: # 预测无变化，实际变化 绿色
                                output_mask[i][j][1] = 255
                            elif mask[i][j][0] == 255 and label[i][j][0] == 0:# 预测变化，实际无变化 红色
                                output_mask[i][j][2] = 255
              
                    score_map = cv2.applyColorMap(np.uint8(255*score_map),  cv2.COLORMAP_JET) # 利用色彩空间转换将heatmap凸显

                    cv2.imwrite(os.path.join(path, "A", str(self.global_id)+".png"), output_a)
                    cv2.imwrite(os.path.join(path, "B", str(self.global_id)+".png"), output_b)
                    cv2.imwrite(os.path.join(path, "score_map", str(self.global_id)+".png"), score_map)
                    cv2.imwrite(os.path.join(path, "pred", str(self.global_id)+".png"), output_mask)
                    cv2.imwrite(os.path.join(path, "bi_pred", str(self.global_id)+".png"), mask)
                    cv2.imwrite(os.path.join(path, "gt", str(self.global_id)+".png"), label)
                    self.global_id += 1
        
        self._collect_epoch_states()
