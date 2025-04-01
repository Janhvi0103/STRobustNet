import torch
import torch.nn.functional as F


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    # if input.shape[-1] != target.shape[-1]:
    #     input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    # print(type(target))
    # print(type(weight))
    # print(target.shape)
    # print(weight.shape)
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def per_pixel_cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    # print(type(target))
    # print(type(weight))
    # print(target.shape)
    # print(weight.shape)
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduce=False, reduction=reduction, size_average=False)



def edge(img):
    # edge_kernal = torch.Tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    edge_kernal = torch.Tensor([[-1,-1,-1,-1,-1],[-1,-1,-2,-1,-1],[-1,-2,28,-2,-1],[-1,-1,-2,-1,-1],[-1,-1,-1,-1,-1]])
    
    edge_kernal =edge_kernal.reshape((1,1,5,5))
    out = F.conv2d(img.unsqueeze(1),edge_kernal.cuda(),padding=2)
    return torch.abs(out).squeeze(1)
def edge_loss(dis, gt):
    # w0 = 2
    weight = (F.sigmoid(edge(gt.float()))) + 1
    
    loss = per_pixel_cross_entropy(dis,gt)
    loss = (weight * loss).mean()

    return loss#+l_corr


def compute_smooth_loss(inputs, outputs):
        inputs = inputs.float()
        outputs = outputs.float()

        def gradient_x(img):
            gx = img[:,:-1,:] - img[:,1:,:]
            return gx

        def gradient_y(img):
            gy = img[:,:,:-1] - img[:,:,1:]
            return gy
       
        
        depth_grad_x = gradient_x(outputs)
        depth_grad_y = gradient_y(outputs)
        image_grad_x = gradient_x(inputs)
        image_grad_y = gradient_y(inputs)

        weights_x = torch.exp(-(torch.abs(image_grad_x)))
        weights_y = torch.exp(-(torch.abs(image_grad_y)))
        smoothness_x = depth_grad_x*weights_x
        smoothness_y = depth_grad_y*weights_y

        loss_x = torch.mean(torch.abs(smoothness_x))
        loss_y = torch.mean(torch.abs(smoothness_y))

        total_loss = loss_x + loss_y
        
        return total_loss


def iou_loss(pred, target, size_average=True):
    pred = torch.softmax(pred, dim=1)
    pred = pred[:,1,:,:]
    b = pred.shape[0]
    IoU = 0.0
    eps = 1e-6
    for i in range(0,b):
        #compute the IoU of the foreground

        Iand1 = torch.sum(target[i,:,:]*pred[i,:,:])
        Ior1 = torch.sum(target[i,:,:]) + torch.sum(pred[i,:,:])-Iand1
        IoU1 = (Iand1+eps)/(Ior1+eps)

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

