import torch,pandas
from torch import optim,nn
from Models import basic_FC
from torch.utils.data import DataLoader
from data_precprocessing import Dataset
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda"

root = "D:\projects\Projects\House_price_prediction\data"

class Trainer:
    def __init__(self,root,net):
        #load training data
        self.train_dataset = Dataset(root,True)
        self.train_loader = DataLoader(self.train_dataset,batch_size=16)#,shuffle=True)
        #load test data
        self.test_dataset = Dataset(root,False)
        self.test_loader = DataLoader(self.test_dataset,batch_size=16)#,shuffle=True)
        #creat model
        self.net = net
        #load model to GPU
        self.net.to(DEVICE)
        #Creat optimizer
        self.opt = optim.Adam(self.net.parameters())
        #Loss function
        self.loss_func = nn.MSELoss()

        # for i, (data, train_tags) in enumerate(self.train_loader):
        #     print(data,train_tags)



    def __call__(self):
        summaryWriter = SummaryWriter("logs")
        step = 0
        for epoch in range(100000):
            print("epo:",epoch)
            sum_loss = 0
            sum_acc = 0
            sum_train_loss = 0
            for i,(data,labels) in enumerate(self.train_loader):
                #load data to GPU
                data,labels = data.to(DEVICE),labels.to(DEVICE)
                # print("test:",data.shape)
                # print("test:", labels.shape)
                # print(i)
                # print(data.shape)
                out_train = self.net.forward(data)
                # print(out_train.shape)
                # print(data)
                # print("out",out_train)
                # print(labels)
                loss_train = torch.sqrt(self.loss_func(out_train,labels))       ## RMSE

                # print("loss: ",loss_train)
                self.opt.zero_grad()
                loss_train.backward()
                self.opt.step()

                sum_train_loss = sum_train_loss + loss_train
                # print("sum",sum_train_loss)
                #k = i

                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()


                if i%10 == 0:
                    print("test")
                    for j, (data, labels) in enumerate(self.test_loader):
                        # load data to GPU
                        data, labels = data.to(DEVICE), labels.to(DEVICE)
                        #print(j)
                        out = self.net.forward(data)
                        loss = torch.sqrt(self.loss_func(out,labels))

                        # acc = torch.mean(torch.eq(torch.argmax(out, dim=1), torch.argmax(labels, dim=1)).float())
                        # sum_acc = sum_acc + acc
                        sum_loss = sum_loss + loss
                        # print(sum_loss)

                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()

                    _loss = sum_loss #/2
                    # _acc = sum_acc #/2
                    _train_loss = sum_train_loss #/ 2
                    # summaryWriter.add_scalar("acc", _acc, step)
                    summaryWriter.add_scalar("test_loss", _loss, step)
                    summaryWriter.add_scalar("train_loss", _train_loss, step)
                    print("Step:",step)
                    print(_train_loss,_loss)
                    print("train_loss", _train_loss.item())
                    print("test_loss:", _loss.item())
                    # print("acc:", _acc.item())
                    sum_loss = 0
                    # sum_acc = 0
                    sum_train_loss = 0
                    step += 1

                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()

                if step == 70:
                   break
                else:
                    continue
            else:
                if step == 70:
                    break
                if step == 70:
                    break
            if step == 70:
                break






net = basic_FC(305,1)
# net.initialize()
trainer = Trainer(root,net)
trainer()









