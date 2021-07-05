#encoding=utf-8
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import cv2
import glob2
import os
import matplotlib.pyplot as plt

eps = 2.2204e-16


def parameter():
    p = {}
    p['gtThreshold'] = 0.5
    p['beta'] = np.sqrt(0.3)
    p['thNum'] = 100
    p['thList'] = np.linspace(0, 1, p['thNum'])

    return p


def im2double(im):
    return cv2.normalize(im.astype('float'),
                         None,
                         0.0, 1.0,
                         cv2.NORM_MINMAX)


def prCount(gtMask, curSMap, p):
    gtH, gtW = gtMask.shape[0:2]
    algH, algW = curSMap.shape[0:2]

    if gtH != algH or gtW != algW:
        curSMap = cv2.resize(curSMap, (gtW, gtH))

    gtMask = (gtMask >= p['gtThreshold']).astype(np.float32)
    gtInd = np.where(gtMask > 0)
    gtCnt = np.sum(gtMask)

    if gtCnt == 0:
        prec = []
        recall = []
    else:
        hitCnt = np.zeros((p['thNum'], 1), np.float32)
        algCnt = np.zeros((p['thNum'], 1), np.float32)

        for k, curTh in enumerate(p['thList']):
            thSMap = (curSMap >= curTh).astype(np.float32)
            hitCnt[k] = np.sum(thSMap[gtInd])
            algCnt[k] = np.sum(thSMap)

        prec = hitCnt / (algCnt+eps)
        recall = hitCnt / gtCnt

    return prec, recall


def PR_Curve(resDir, gtDir):
    p = parameter()
    beta = p['beta']

    gtImgs = glob2.iglob(gtDir + '/*.png')

    prec = []
    recall = []

    for gtName in gtImgs:
        dir, name = os.path.split(gtName)
        mapName = os.path.join(resDir, name)
        if method!='CGL' and method!='MFSR' and method!='TSAA'and method!='PDNet' and method!='FMCF' and method!='Ours':#
           mapName = mapName.replace('.png', '_' + 'RT_' + '.png')
        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))

        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))

        if curMap.shape[0] != curGT.shape[0]:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))

        curPrec, curRecall = prCount(curGT, curMap, p)

        prec.append(curPrec)
        recall.append(curRecall)


    prec = np.hstack(prec[:])
    recall = np.hstack(recall[:])

    prec = np.mean(prec, 1)
    recall = np.mean(recall, 1)

    prec_ = np.mean(prec)
    recall_ = np.mean(recall)

    #pr曲线
    #if method=='MRCMC':
    
     #  plt.plot(recall, prec, label = "MRCMC", linewidth = 1) #, color = "purple", linestyle=':'
  #  elif method=='BMPM':
      # plt.plot(recall, prec, label = "BMPM", color = "black", linewidth = 1) #, color = "green" , linestyle=':'
   # elif method=='Ours':
      # plt.plot(recall, prec, label = "Ours", color = "red", linewidth = 1)
  #  elif method=='FMCF':
      # plt.plot(recall, prec, label = "FMCF", linewidth = 1)
   # elif method=='DSS': 
      #plt.plot(recall, prec, label = "DSS", linewidth = 1) #, color = "brown" , linestyle=':'
   # elif method=='Amulet':
     # plt.plot(recall, prec, label = "Amulet", linewidth = 1) #, color = "orange" , linestyle=':'
    #elif method=='NLDF':
       #plt.plot(recall, prec, label = "NLDF", linewidth = 1) #, color = "black" , linestyle=':'
   # elif method=='UCF':
      # plt.plot(recall, prec, label = "UCF", linewidth = 1) #, color = "yellow" , linestyle=':'
   # elif method=='CGL':
      # plt.plot(recall, prec, label = "CGL", linewidth = 1) # , linestyle=':'
   # elif method=='CPD':
       #plt.plot(recall, prec, label = "CPD", linewidth = 1) #, linestyle=':'
   # elif method=='MFSR': 
      # plt.plot(recall, prec, label = "MFSR", linewidth = 1) # , linestyle=':'
   # elif method=='TSAA':
      # plt.plot(recall, prec, label = "TSAA", linewidth = 1) #, linestyle=':'
   # elif method=='PDNet': 
      # plt.plot(recall, prec, label = "PDNet", linewidth = 1) # , linestyle=':'

    
    #plt.legend()
    #plt.xlim(0, 1)
    #new_ticks = np.linspace(0,1,6)
   # plt.xticks(new_ticks)
   # plt.ylim(0.1, 1)
   # new_ticks = np.linspace(0.1,1,10)
   # plt.yticks(new_ticks)
   # plt.xlabel("Recall", fontsize=14)
   # plt.ylabel("Precision", fontsize=14)
                                #plt.title('RGB-T')
   # plt.grid(linestyle='--')
   # plt.show()
   # save_name = os.path.join('/home/ly/disk2/xiao/RGBT1', 'RGBT13'+'_pr.png')
   # plt.savefig(save_name) 

    # compute the max F-Score
    score = (1+beta**2)*prec*recall / (beta**2*prec + recall)
    curTh = np.argmax(score)
    curScore = np.max(score)
    meanScore = np.mean(score)

    res = {}
    res['prec'] = prec
    res['recall'] = recall
    res['prec_'] = prec_
    res['recall_'] = recall_
    res['curScore'] = curScore
    res['meanScore'] = meanScore
    res['curTh'] = curTh

    return res


def MAE_Value(resDir, gtDir):
    p = parameter()
    gtThreshold = p['gtThreshold']

    gtImgs = glob2.iglob(gtDir + '/*.png')

    MAE = []

    for gtName in gtImgs:
        dir, name = os.path.split(gtName)
        mapName= os.path.join(resDir, name)
        if method!='CGL' and method!='MFSR' and method!='TSAA'and method!='PDNet' and method!='FMCF' and method!='Ours':# 
           mapName = mapName.replace('.png', '_' + 'RT_' + '.png')
        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))

        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))
        curGT = (curGT >= gtThreshold).astype(np.float32)

        if curMap.shape[0] != curGT.shape[0]:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))

        diff = np.abs(curMap - curGT)

        MAE.append(np.mean(diff))

    return np.mean(MAE)


if __name__ == "__main__":

    #methods =['1']
    methods =['UCF','Amulet','DSS','BMPM', 'CPD','TSAA', 'PDNet', 'MRCMC', 'CGL', 'MFSR', 'FMCF', 'Ours']
   # methods =['MRCMC','UCF','Amulet','NLDF','DSS','BMPM', 'CGL', 'CPD', 'MFSR', 'TSAA', 'PDNet','Ours']
    label = ('UCF','Amulet','DSS','BMPM', 'CPD','TSAA', 'PDNet', 'MRCMC', 'CGL', 'MFSR', 'FMCF', 'Ours')
    #methods = ['sMFCF']
    mean_prec = []
    mean_recall= []
    mean_fmea = []

    for method in methods:
        
      resDir = 'Result-'+ method#'RGBD/' + method  #single/ datas
      gtDir = 'DATA/GT-test154'#'data/GT-test200'##'/home/ly/disk2/xiao/RGBT2/data/GT-test346'#'/home/ly/disk2/xiao/RGBT2/data/GT-test346'#

      mae = MAE_Value(resDir, gtDir)
      pr = PR_Curve(resDir, gtDir)

      print (method)
      mean_prec.append(pr['prec_'])
      mean_recall.append(pr['recall_'])
      mean_fmea.append(pr['meanScore'])
      print ('max F:', pr['curScore'])
      print ('mean F:', pr['meanScore'])
      print ('MAE:', mae)

    x =list(np.arange(len(methods))) 

    total_width, n = 0.6, 3
    width = total_width / n
    
    rect=plt.bar(x, mean_prec, tick_label='', width=width, label='precision', fc='b')
    for i in range(len(x)):
         x[i]+=width
    rect1=plt.bar(x, mean_recall, width=width, label='recall', fc='g') #,tick_label= methods
    plt.xticks(x, label, fontsize=8)
    for i in range(len(x)):
         x[i]+=width
    rect2=plt.bar(x, mean_fmea, width=width, label='F-measure', fc='r')
    plt.legend(loc='lower left')

    
    #plt.ylim(0.2, 1)
    
    
    plt.show()
    save_name = os.path.join('/home/ly/disk2/xiao/RGBT1', 'RGBT13'+'_bar.png')
    plt.savefig(save_name)



    


