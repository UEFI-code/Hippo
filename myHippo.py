import torch
import torch.nn as nn
import torch.nn.functional as F

class myHippo(nn.Module):
    def __init__(self, poolSize, poolDim):
        super(myHippo, self).__init__()
        self.poolDim = poolDim
        self.memPool = []
        for i in range(poolSize):
            self.memPool.append(torch.randn(poolDim))
    
    def resonancer(self, x):
        outputBuffer = torch.zeros(self.poolDim)
        for i in self.memPool:
            simliar = F.cosine_similarity(x.view(1, self.poolDim), i.view(1, self.poolDim)) #Note for Bad Memory
            outputBuffer += simliar * i
            i += simliar * x
            if(i.abs().max() != 0):
                i /= i.abs().max()
        return outputBuffer / outputBuffer.abs().max()
    
    def curiouser(self, x):
        newMemID = 0
        levelP = None
        levelN = None
        for i in range(len(self.memPool)):
            simliar = F.cosine_similarity(x.view(1, self.poolDim), self.memPool[i].view(1, self.poolDim))
            if simliar > 0: #Positive Memory
                if levelP is None:
                    levelP = simliar
                else:
                    levelP += simliar
            else: #Negative Memory
                if levelN is None:
                    levelN = simliar
                else:
                    levelN += simliar
            if self.memPool[i].abs().sum() < self.memPool[newMemID].abs().sum(): #Found New Memory Index
                newMemID = i
        levelP = 1 / levelP
        levelN = -1 * levelN
        print('Curious level Positive: ', levelP)
        print('Curious level Negative: ', levelN)
        print('Choose new memory ID: ', newMemID)
        self.memPool[newMemID] += x * (levelP + levelN)
        if(self.memPool[newMemID].abs().max() != 0):
            self.memPool[newMemID] /= self.memPool[newMemID].abs().max()

myTestObj = myHippo(8, 16)
print('myTestObj.memPool = ', myTestObj.memPool)
testValue = torch.rand(16)
print('testValue = ', testValue)
print('myTestObj.resonancer(testValue) = ', myTestObj.resonancer(testValue))
print('myTestObj.memPool = ', myTestObj.memPool)
print('myTestObj.curiouser(testValue) = ', myTestObj.curiouser(testValue))
print('myTestObj.memPool = ', myTestObj.memPool)



           

