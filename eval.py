import time
import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from utils import evalIoU

def eval(args, model, loader_val, criterion, classfication, epoch):
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        #New confusion matrix 
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0
        
        for step, (images,classi, images2, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                classi = classi.cuda()
                images2 = images2.cuda()
                labels = labels.cuda()
            inputs = Variable(images, volatile=True)
            inputs2 = Variable(images2,volatile=True)
            targets = Variable(labels, volatile=True)
            p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12 = model(inputs, inputs2)
            
            loss = criterion(p1, targets[:, 0])
            loss1 = criterion(p2, targets[:, 0])
            loss2 = criterion(p3, targets[:, 0])
            loss3 = criterion(p4, targets[:, 0])
            loss4 = criterion(p5, targets[:, 0])
            loss5 = criterion(p6, targets[:, 0])
            loss6 = criterion(p7, targets[:, 0])
            loss7 = criterion(p8, targets[:, 0])
            loss8 = criterion(p9, targets[:, 0])
            loss9 = criterion(p10, targets[:, 0])
            loss10 = criterion(p11, targets[:, 0])
            loss11 = criterion(p12, targets[:, 0])

            loss += loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11
            
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)
            
            average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
            
            if args.iouVal: # add to confMatrix
                add_to_confMatrix(p7, labels,confMatrix, perImageStats, nbPixels)
                
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
            print('VAL loss: {} (epoch: {}, step: {})'.format(average,epoch,step), 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                        
        average_epoch_loss_train = sum(epoch_loss_val) / len(epoch_loss_val)
        iouAvgStr, iouVal, classScoreList = cal_iou(evalIoU, confMatrix)
        print ("EPOCH IoU on VAL set: ", iouAvgStr)

        return average_epoch_loss_val, iouVal
            
def add_to_confMatrix(prediction, groundtruth, confMatrix, perImageStats, nbPixels):
    if isinstance(prediction, list):   #merge multi-gpu tensors
        outputs_cpu = prediction[0].cpu()
        for i in range(1,len(prediction)):
            outputs_cpu = torch.cat((outputs_cpu, prediction[i].cpu()), 0)
    else:
        outputs_cpu = prediction.cpu()
    for i in range(0, outputs_cpu.size(0)):   #args.batch_size,evaluate iou of each batch
        prediction = ToPILImage()(outputs_cpu[i].max(0)[1].data.unsqueeze(0).byte())
        groundtruth_image = ToPILImage()(groundtruth[i].cpu().byte())
        nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth_image, confMatrix, perImageStats, evalIoU.args)    

def cal_iou(evalIoU, confMatrix):
        iou = 0
        classScoreList = {}
        for label in evalIoU.args.evalLabels:
            labelName = evalIoU.trainId2label[label].name
            classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)

        iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol
        iou = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
        return iouAvgStr, iou, classScoreList
    
