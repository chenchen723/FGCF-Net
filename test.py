
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2

from lib.network import DSHNet
from dataloader import test_dataset

if __name__ == '__main__':
    method_name = 'DSHNet'
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')#输入图像的大小
    parser.add_argument('--pth_path', type=str, default='./save/Model3.pth')#保存的权重的路径
    opt = parser.parse_args()#将参数存储在变量opt中
    
    model = DSHNet()#引入lib.network中的网络模型
    model.cuda()#加载到GPU上
    model.load_state_dict(torch.load(opt.pth_path), strict=False)#加载保存的模型权重
    model.eval()#模型为评估模式
    
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:#遍历数据集列表的名称
        data_path = f'./dataset/TestDataset/{_data_name}'#将数据集的名称_data_name添加到路径中
        save_path = f'./results/{_data_name}/'#保存分割的结果

        if not os.path.exists(save_path):#是否存在保存分割结果的路径如果不存在就重新创建
            os.makedirs(save_path, exist_ok=True)
        
        print(f'Evaluating {data_path}')#打印正在处理的数据集路径
        
        test_loader = test_dataset(#创建测试集的路径它包括真实的图像和对应的掩码
            image_root=f'{data_path}/images/',#测试图像的路径
            gt_root=f'{data_path}/masks/',#真实的掩码
            testsize=opt.testsize
        )
        
        total_samples = test_loader.size#获取测试数据加载器中样本的数量
        metrics = {'DSC': 0.0, 'JACARD': 0.0, 'MAE': 0.0}
        
        for i in range(total_samples):#遍历测试数据集中所有的样本数量
            # 关键修改点1：确保正确解包数据
            data = test_loader.load_data()#从测试数据加载器中加载一个样本保存在data中
            #对data长度进行解析
            if len(data) == 4:
                image, gt, image_dct, name = data#image输入的图像，gt真值，name文件名image_dct可能的辅助输出
            else:
                image, gt, name = data[:3]
                image_dct = None  # 根据实际情况调整
            
            # 关键修改点2：转换文件名类型
            if isinstance(name, torch.Tensor):
                name = name.item()  # 如果name是数字型Tensor
            name = str(name).split('/')[-1]  # 确保是字符串且无路径
            
            # 数据处理
            gt = np.asarray(gt, np.float32)#将真值转化为numpy数组，然后进行归一化
            gt /= (gt.max() + 1e-8)
            
            # 模型推理
            with torch.no_grad():
                image = image.cuda()
                image_dct = image_dct.cuda() if image_dct is not None else None
                
                if image_dct is not None:
                    pred, masks, mid_preds = model(image, image_dct)
                else:
                    pred, masks, mid_preds = model(image)
                
                res = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear')#将预测结果的大小转化为掩码的大小
                res = res.sigmoid().squeeze().cpu().numpy()#对预测结果进行预测
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res_uint8 = (res * 255).astype(np.uint8)  # 合并转换操作
            
            # 关键修改点3：正确的文件保存
            cv2.imwrite(os.path.join(save_path, f"{name}.png"), res_uint8)#将预测结果保存为name.png
            
            # 指标计算
            input_mask = (res >= 0.5).astype(np.uint8)#转化为二值掩码
            target_mask = (gt >= 0.5).astype(np.uint8)
            
            intersection = np.sum(input_mask & target_mask)#预测掩码和真是掩码的交集
            union = np.sum(input_mask | target_mask)#掩码和真值的并集
            
            metrics['JACARD'] += (intersection + 1e-8) / (union + 1e-8)#交并比
            metrics['DSC'] += (2. * intersection + 1e-8) / (np.sum(input_mask) + np.sum(target_mask) + 1e-8)#2 倍交集与预测掩码和真值掩码总和的比值
            metrics['MAE'] += np.mean(np.abs(gt - res))#预测结果与真值掩码之间绝对误差的平均值
        
        # 结果输出
        print('*****************************************************')
        print(f'{_data_name} Results:')
        for k, v in metrics.items():
            avg = v / total_samples
            print(f'{k}: {avg:.4f}')
        print('*****************************************************')
