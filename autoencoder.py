class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        #should change channel 3
        self.conv_1_1 = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding=1)
        self.relu_1_2 = nn.ReLU(True)
            
        self.conv_2_1 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1)
        self.bn_2_2 = nn.BatchNorm2d(16)
        self.relu_2_3 = nn.ReLU(True)
        
        self.conv_3_1 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.bn_3_2 = nn.BatchNorm2d(32)
        self.relu_3_3 = nn.ReLU(True)
        
        self.conv_4_1=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.bn_4_2=nn.BatchNorm2d(64)
        self.relu_4_3=nn.ReLU(True)
        
        self.conv_5_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.bn_5_2 =nn.BatchNorm2d(128)
        self.relu_5_3 = nn.LeakyReLU()
        
        ## decoder
        self.upsample_6_1 = nn.Upsample(scale_factor=2,mode='nearest')
        self.convtranspose_6_2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.bn_6_3 = nn.BatchNorm2d(64)
        self.relu_6_4 = nn.ReLU(True)
        
        self.upsample_7_1 = nn.Upsample(scale_factor=2,mode='nearest')
        self.convtranspose_7_2 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1)
        self.bn_7_3 = nn.BatchNorm2d(32)
        self.relu_7_4 = nn.ReLU(True)
        
        self.upsample_8_1 = nn.Upsample(scale_factor=2,mode='nearest')
        self.convtranspose_8_2 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1)
        self.bn_8_3 = nn.BatchNorm2d(16)
        self.relu_8_4 = nn.ReLU(True)
        
        self.upsample_9_1 = nn.Upsample(scale_factor=2,mode='nearest')
        self.convtranspose_9_2 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,padding=1)
        self.bn_9_3 = nn.BatchNorm2d(8)
        self.relu_9_4 = nn.ReLU(True)
        #should change channel 6
        
        self.conv_10_1 = nn.ConvTranspose2d(in_channels=8,out_channels=3,kernel_size=3,padding=1)
        self.tanh_10_2=nn.Tanh()
        
        # no masking no *0 or assign value zero not done .        
        
    def forward(self, x,label):
        x1 = self.conv_1_1(x)
        spare = x1.clone()
        x2 = self.relu_1_2(x1)

        
        x3 = self.conv_2_1(x2)
        x4 = self.bn_2_2(x3)
        x5 = self.relu_2_3(x4)

        
        x6 = self.conv_3_1(x5)
        x7 = self.bn_3_2(x6)
        x8 = self.relu_3_3(x7)

        
        x9 = self.conv_4_1(x8)
        x10 = self.bn_4_2(x9)
        x11 = self.relu_4_3(x10)
    
        x12 = self.conv_5_1(x11)
        x13 = self.bn_5_2(x12)
        x14 = self.relu_5_3(x13)
        act = x14.clone()
        dep = x14.clone()

        # Selection block setting zero values based on label
        # [:64] -> fake data latent space 
        # [64:] -> real data latent space
        # 0->fake 1 ->real
        A = torch.nn.Parameter(torch.zeros(64,15,15))
        for i in range(len(label)):
            #real 
            if label[i].item():
                #setting fake latent space into zero
                dep[i,:64] = A
            else:
                dep[i,64:]=A
                
        x15 = self.upsample_6_1(dep) 
        x16 = self.convtranspose_6_2(x15)
        x17 = self.bn_6_3(x16)
        x18 = self.relu_6_4(x17) 
        x19 = self.upsample_7_1(x18)
        x20 = self.convtranspose_7_2(x19)
        x21 = self.bn_7_3(x20)
        x22 = self.relu_7_4(x21)
        x23 = self.upsample_8_1(x22)
        x24 = self.convtranspose_8_2(x23)
        x25 = self.bn_8_3(x24)
        x26 = self.relu_8_4(x25)
        x27 = self.upsample_9_1(x26)
        x28 = self.convtranspose_9_2(x27) 
        x29 = self.bn_9_3(x28)
        x30 = self.relu_9_4(x29)
        x31 = self.conv_10_1(x30)
        x32 = self.tanh_10_2(x31)

        return  x32 , act