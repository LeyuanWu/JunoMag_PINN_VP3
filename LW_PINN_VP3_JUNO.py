###############################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
###############################################
# %% # ! Setup
import tensorflow as tf;
import numpy as np;
import time;
tf.compat.v1.disable_eager_execution();
# %% # ! Functions

def GetConstant(Name):
    if Name=='Rj':
        Rj=71492; # Jupiter radius in km
        return Rj;

def DispMSERMSE(Est,Ref,Name,unit):
    MSE   = np.mean((Est-Ref)**2);
    RMSE  = np.sqrt(MSE);
    print('\n*********************\n',flush=True);
    print('<MSE-%s>: %.4f [%s]^2'\
        %(Name,MSE,unit),flush=True);
    print('<RMSE-%s>: %.4f %s'\
        %(Name,RMSE,unit),flush=True);
    L2NormRef = np.sqrt(np.mean((Ref)**2));
    if L2NormRef != 0:
        E2 = RMSE / L2NormRef;
        print('<E2-%s>: %.2e '%(Name,E2),flush=True);
    print('\n*********************\n',flush=True);

def LoadObsData(pj1,pj2,cutRType,showinfo=True):   
    # ***************** Format ******************
    # PJNUM Year Decimal-Day X Y Z Bx By Bz
    # *******************************************
    match cutRType:
        case 1:
            cutR=2.5;
        case 2:
            cutR=4.0;
        case 3:
            cutR=7.0;
    fileObs='input/Juno_PJ%02d_%02d_%.1fRj.txt'%(pj1,pj2,cutR);
    dataObs=np.loadtxt(fileObs,skiprows=1);
    PJ=dataObs[:,0:1]; Year=dataObs[:,1:2]; DD=dataObs[:,2:3]; 
    xObs=dataObs[:,3:4]; yObs=dataObs[:,4:5]; zObs=dataObs[:,5:6]; 
    bxObs=dataObs[:,6:7]; byObs=dataObs[:,7:8]; bzObs=dataObs[:,8:9];
    RObs=np.sqrt(xObs**2+yObs**2+zObs**2); 
    RObsRj=RObs/GetConstant('Rj');
    bNorm=np.sqrt(bxObs**2+byObs**2+bzObs**2); 
    nObs=xObs.size; 
    if showinfo:
        print('\n*****************************************\n',flush=True);
        print('Number of <Obs>: %d'%(nObs),flush=True);
        print('PJ in [%d,%d];'%(PJ.min(),PJ.max()),flush=True);
        print('Year in [%d,%d];'%(Year.min(),Year.max()),flush=True);
        print('x in [%.4f,%.4f] km;'%(xObs.min(),xObs.max()),flush=True);
        print('y in [%.4f,%.4f] km;'%(yObs.min(),yObs.max()),flush=True);
        print('z in [%.4f,%.4f] km;'%(zObs.min(),zObs.max()),flush=True);
        print('RObs in [%.4f,%.4f] km;'%(RObs.min(),RObs.max()),flush=True);
        print('RObs in [%.3f,%.3f] Rj;'%(RObsRj.min(),RObsRj.max()),flush=True);
        print('bx in [%.9f,%.6f] Gauss;'%(bxObs.min(),bxObs.max()),flush=True);
        print('by in [%.6f,%.6f] Gauss;'%(byObs.min(),byObs.max()),flush=True);
        print('bz in [%.6f,%.6f] Gauss;'%(bzObs.min(),bzObs.max()),flush=True);
        print('bNorm in [%.6f,%.6f] Gauss;'%(bNorm.min(),bNorm.max()),flush=True);
        print('\n*****************************************\n',flush=True);
        print('Distribution of <Obs>: %d'%(nObs),flush=True);
        Rbins=np.concatenate((np.linspace(0.99,1.10,12)[:,None],\
                            np.linspace(1.2,2.0,9)[:,None],\
                            np.linspace(2.5,4.0,4)[:,None],\
                            np.linspace(5.0,8.0,4)[:,None]));
        nRbin=Rbins.size;
        for i in range(nRbin-1):
            r1=Rbins[i]; r2=Rbins[i+1];
            n12=np.count_nonzero(np.logical_and(RObsRj>=r1,RObsRj<r2));
            print('Number of Obs in [%.2f,%.2f) Rj: %d'%(r1,r2,n12),flush=True);
        print('\n*****************************************\n',flush=True);
    return nObs,PJ,Year,DD,xObs,yObs,zObs,bxObs,byObs,bzObs;

def LoadObsData_DC(fileObs,showinfo=True):   
    # ******** Format *********
    # X Y Z Bx By Bz
    # *************************
    dataObs=np.loadtxt(fileObs,skiprows=1);
    xObs=dataObs[:,0:1]; yObs=dataObs[:,1:2]; zObs=dataObs[:,2:3]; 
    bxObs=dataObs[:,3:4]; byObs=dataObs[:,4:5]; bzObs=dataObs[:,5:6];
    RObs=np.sqrt(xObs**2+yObs**2+zObs**2); 
    RObsRj=RObs/GetConstant('Rj');
    bNorm=np.sqrt(bxObs**2+byObs**2+bzObs**2); 
    nObs=xObs.size; 
    if showinfo:
        print('\n*****************************************\n',flush=True);
        print('Number of <Obs>: %d'%(nObs),flush=True);
        print('x in [%.4f,%.4f] km;'%(xObs.min(),xObs.max()),flush=True);
        print('y in [%.4f,%.4f] km;'%(yObs.min(),yObs.max()),flush=True);
        print('z in [%.4f,%.4f] km;'%(zObs.min(),zObs.max()),flush=True);
        print('RObs in [%.4f,%.4f] km;'%(RObs.min(),RObs.max()),flush=True);
        print('RObs in [%.3f,%.3f] Rj;'%(RObsRj.min(),RObsRj.max()),flush=True);
        print('bx in [%.9f,%.6f] Gauss;'%(bxObs.min(),bxObs.max()),flush=True);
        print('by in [%.6f,%.6f] Gauss;'%(byObs.min(),byObs.max()),flush=True);
        print('bz in [%.6f,%.6f] Gauss;'%(bzObs.min(),bzObs.max()),flush=True);
        print('bNorm in [%.6f,%.6f] Gauss;'%(bNorm.min(),bNorm.max()),flush=True);
        print('\n*****************************************\n',flush=True);
        print('Distribution of <Obs>: %d'%(nObs),flush=True);
        Rbins=np.concatenate((np.linspace(0.99,1.10,12)[:,None],\
                            np.linspace(1.2,2.0,9)[:,None],\
                            np.linspace(2.5,4.0,4)[:,None],\
                            np.linspace(5.0,8.0,4)[:,None]));
        nRbin=Rbins.size;
        for i in range(nRbin-1):
            r1=Rbins[i]; r2=Rbins[i+1];
            n12=np.count_nonzero(np.logical_and(RObsRj>=r1,RObsRj<r2));
            print('Number of Obs in [%.2f,%.2f) Rj: %d'%(r1,r2,n12),flush=True);
        print('\n*****************************************\n',flush=True);
    return nObs,xObs,yObs,zObs,bxObs,byObs,bzObs;

def LoadColData(fileCol,showinfo=True):
    # ******** Format *********
    # X Y Z 
    # *************************
    dataCol = np.loadtxt(fileCol,skiprows=1);
    xCol=dataCol[:,0:1]; yCol=dataCol[:,1:2]; zCol=dataCol[:,2:3];
    RCol=np.sqrt(xCol**2+yCol**2+zCol**2);
    RColRj=RCol/GetConstant('Rj');
    nCol=xCol.size; 
    if showinfo:
        print('\n*****************************************\n',flush=True);
        print('Number of <Laplacian-Collocation>: %d'%(nCol),flush=True);
        print('RCol in: [%.4f,%.4f] km;'%(RCol.min(),RCol.max()),flush=True);
        print('RCol in: [%.3f,%.3f] Rj;'%(RColRj.min(),RColRj.max()),flush=True);
        print('Distribution of <Collocation>: %d'%(nCol),flush=True);
        Rbins=np.concatenate((np.linspace(0.60,1.00,41)[:,None],\
                            np.linspace(1.1,2.0,10)[:,None],\
                            np.linspace(2.5,4.0,4)[:,None],\
                            np.linspace(5.0,8.0,4)[:,None]));
        nRbin=Rbins.size;
        for i in range(nRbin-1):
            r1=Rbins[i]; r2=Rbins[i+1];
            n12=np.count_nonzero(np.logical_and(RColRj>=r1,RColRj<r2));
            print('Number of Collocation in [%.2f,%.2f) Rj: %d'%(r1,r2,n12),flush=True);
        print('\n*****************************************\n',flush=True);
    return nCol,xCol,yCol,zCol;

def BuildPINN(pj1,pj2,cutRType,fileCol,nLayer,nNeuron,\
     actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed):
    ######## Build <File Name Head>
    fNameHead=BuildFileNameHead(pj1,pj2,cutRType,nLayer,nNeuron,\
        actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed);
    print('\n*****************************************\n',flush=True);
    print('<File Name Head>: %s'%(fNameHead),flush=True);
    fContHead=BuildFileContentHead(pj1,pj2,cutRType,nLayer,nNeuron,\
        actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed);
    print('\n*****************************************\n',flush=True);
    print('<File Content Head>:\n%s'%(fContHead),flush=True);
    ######## Load <OBS> points
    _,_,_,_,xObs,yObs,zObs,bxObs,byObs,bzObs=\
        LoadObsData(pj1,pj2,cutRType,showinfo=True);
    ######## Load <Collocation> points
    _,xCol,yCol,zCol=LoadColData(fileCol,showinfo=True);
    np.random.seed(rdSeed);
    tf.random.set_seed(rdSeed);
    layers = [3] + [nNeuron]*nLayer + [1];
    obs_Data=[np.float32(xObs), np.float32(yObs),np.float32(zObs),\
        np.float32(bxObs), np.float32(byObs),np.float32(bzObs)]; 
    col_Data=[np.float32(xCol), np.float32(yCol),np.float32(zCol)]; 
    model=CurlCurl3D(layers,actiFun,opti,DW,obs_Data,col_Data);
    return fNameHead, model;

def BuildPINN_DC(pj1,pj2,cutRType,fileObs,fileCol,nLayer,nNeuron,\
     actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed):
    ######## Build <File Name Head>
    fNameHead=BuildFileNameHead(pj1,pj2,cutRType,nLayer,nNeuron,\
        actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed);
    print('\n*****************************************\n',flush=True);
    print('<File Name Head>: %s'%(fNameHead),flush=True);
    fContHead=BuildFileContentHead(pj1,pj2,cutRType,nLayer,nNeuron,\
        actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed);
    print('\n*****************************************\n',flush=True);
    print('<File Content Head>:\n%s'%(fContHead),flush=True);
    ######## Load <OBS> points
    _,xObs,yObs,zObs,bxObs,byObs,bzObs=\
        LoadObsData_DC(fileObs,showinfo=True);
    ######## Load <Collocation> points
    _,xCol,yCol,zCol=LoadColData(fileCol,showinfo=True);
    np.random.seed(rdSeed);
    tf.random.set_seed(rdSeed);
    layers = [3] + [nNeuron]*nLayer + [1];
    obs_Data=[np.float32(xObs), np.float32(yObs),np.float32(zObs),\
        np.float32(bxObs), np.float32(byObs),np.float32(bzObs)]; 
    col_Data=[np.float32(xCol), np.float32(yCol),np.float32(zCol)]; 
    model=CurlCurl3D(layers,actiFun,opti,DW,obs_Data,col_Data);
    return fNameHead, model;

def BuildFileNameHead(pj1,pj2,cutRType,nLayer,nNeuron,\
     actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed):
    match cutRType:
        case 1:
            cutR=2.5;
        case 2:
            cutR=4.0;
        case 3:
            cutR=7.0;
    fNameHead='JUNO_PINN_VP3_PJ%02d_%02d_%.1fRj'%(pj1,pj2,cutR)\
        +'_NN%02d_%03d_%s_%s'%(nLayer,nNeuron,actiFun,opti)\
        +'_DW%d'%(DW)\
        +'_RADk%dc%dn%dd%d'%(k,c,n0,dn)\
        +'_nEpo%07d_BS%07d'%(nEpo,BS)\
        +'_Seed%d'%(rdSeed);
    return fNameHead;

def BuildFileContentHead(pj1,pj2,cutRType,nLayer,nNeuron,\
     actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed):
    match cutRType:
        case 1:
            cutR=2.5;
        case 2:
            cutR=4.0;
        case 3:
            cutR=7.0;
    fContHead0='\n****** Theory ******\n'\
        +'Vector Potential-based\n'\
        +'Using 3 NNs';
    fContHead1='\n****** Obs Data Info ******\n'\
        +'PJ data: orbit %d-%d\n'%(pj1,pj2)\
        +'cutRj: %.1f'%(cutR);
    fContHead2='\n****** Neural Network Info ******\n'\
        +'NN Size and Shape: %d*[%d];\n'%(nLayer,nNeuron)\
        +'NN Activation function: %s;\n'%(actiFun)\
        +'NN Optimizer: %s;\n'%(opti)\
        +'Dynamic Weights: %d;\n'%(DW)\
        +'RAD parameter: k=%d, c=%d, n0=%d, dn=%d;\n'%(k,c,n0,dn)\
        +'Epochs: %d;\n'%(nEpo)\
        +'Batch size: %d;\n'%(BS)\
        +'Seed: %d;\n'%(rdSeed);
    fContHead=fContHead0+fContHead1+fContHead2;
    return fContHead;

def NormalizeCord(xAll,yAll,zAll,showinfo=True):
    xAll1=xAll.min(); yAll1=yAll.min(); zAll1=zAll.min();
    xAll2=xAll.max(); yAll2=yAll.max(); zAll2=zAll.max();
    xyzAll1=np.min([xAll1,yAll1,zAll1]);
    xyzAll2=np.max([xAll2,yAll2,zAll2]);
    cordScale = 2/(xyzAll2-xyzAll1); # 1/km
    cordShift =-2*xyzAll1/(xyzAll2-xyzAll1)-1; # km
    xLB=cordScale*xAll1+cordShift; xUB=cordScale*xAll2+cordShift;
    yLB=cordScale*yAll1+cordShift; yUB=cordScale*yAll2+cordShift;
    zLB=cordScale*zAll1+cordShift; zUB=cordScale*zAll2+cordShift;
    if showinfo:
        print('\n*****************************************\n',flush=True);
        print('<Before scaling>:',flush=True);
        print('X in [%.4f,%.4f]; Y in [%.4f,%.4f]; Z in [%.4f,%.4f];'\
            %(xAll1,xAll2,yAll1,yAll2,zAll1,zAll2),flush=True);
        print('<After scaling>:',flush=True);
        print('X in [%.4f,%.4f]; Y in [%.4f,%.4f]; Z in [%.4f,%.4f];'\
            %(xLB,xUB,yLB,yUB,zLB,zUB),flush=True);
        print('cordScale: %.4e; cordShift: %4e'\
            %(cordScale,cordShift),flush=True);
        print('\n*****************************************\n',flush=True);
    return cordScale,cordShift;

def NormalizeMag(bAll,bLimit,showinfo=True):
    # bLimit: B normalized to [-bLimit,bLimit]
    # if bLimit == 0, NO normalization 
    bAll1=bAll.min();
    bAll2=bAll.max();
    if bLimit>0:
        N=2*bLimit;
        magScale = N/(bAll2-bAll1); # 1/Gauss
        magShift =-N*bAll1/(bAll2-bAll1)-N/2; # Gauss
        bLB=magScale*bAll1+magShift; bUB=magScale*bAll2+magShift;
    elif bLimit == 0:
        magScale = 1.0;
        magShift = 0.0;
        bLB=bAll1; bUB=bAll2;
    if showinfo:
        print('\n*****************************************\n',flush=True);
        print('<Before scaling>:',flush=True);
        print('B in [%.4f,%.4f] Gauss;'%(bAll1,bAll2),flush=True);
        print('<After scaling>:',flush=True);
        print('B in [%.4f,%.4f] Gauss;'%(bLB,bUB),flush=True);
        print('magScale: %.4e; magShift: %4e'\
            %(magScale,magShift),flush=True);
        print('\n*****************************************\n',flush=True);
    return magScale,magShift;

class CurlCurl3D:
    def __init__(self, layers, actiFun, opti, DW, obs_Data, col_Data):  
        ######## Normalization      
        x_obs, y_obs, z_obs, bx_obs, by_obs, bz_obs = obs_Data;
        x_col, y_col, z_col = col_Data;
        xAll=np.concatenate((x_obs,x_col));
        yAll=np.concatenate((y_obs,y_col));
        zAll=np.concatenate((z_obs,z_col));
        self.cordScale,self.cordShift = NormalizeCord(xAll,yAll,zAll);
        x_obs=self.cordScale*x_obs+self.cordShift; 
        y_obs=self.cordScale*y_obs+self.cordShift; 
        z_obs=self.cordScale*z_obs+self.cordShift; 
        x_col=self.cordScale*x_col+self.cordShift; 
        y_col=self.cordScale*y_col+self.cordShift; 
        z_col=self.cordScale*z_col+self.cordShift; 
        bAll=np.concatenate((bx_obs,by_obs,bz_obs));
        self.magScale,self.magShift = NormalizeMag(bAll,0);
        bx_obs=self.magScale*bx_obs+self.magShift; 
        by_obs=self.magScale*by_obs+self.magShift; 
        bz_obs=self.magScale*bz_obs+self.magShift; 
        self.xyzb_obs=np.concatenate((x_obs,y_obs,z_obs,bx_obs,by_obs,bz_obs),axis=1);
        self.xyz_col=np.concatenate((x_col, y_col, z_col),axis=1);
        ######## Dynamic Weights
        self.DW = DW;
        if self.DW:
            self.beta = 0.9;
        else:
            self.beta = 1.0;
        self.w_obs = np.array(1.0,dtype=np.float32);
        ######## Initialize NN 
        self.layers = layers; self.actiFun = actiFun; 
        self.weights1, self.biases1 = self.initialize_NN(self.layers);
        self.weights2, self.biases2 = self.initialize_NN(self.layers);
        self.weights3, self.biases3 = self.initialize_NN(self.layers);
        ######## Define Tensorflow session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(\
            allow_soft_placement=True,log_device_placement=False));
        ######## Define placeholders and computational graph
        self.x_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.y_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.z_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.bx_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.by_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.bz_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.x_col_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.y_col_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        self.z_col_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);
        ######## Define placeholder for dynamic weight
        self.w_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=self.w_obs.shape);
        ######## Evaluate predictions
        self.bx_obs_pred, self.by_obs_pred, self.bz_obs_pred \
            = self.net_curl_A(self.x_obs_tf, self.y_obs_tf, self.z_obs_tf);
        self.curlBx_col_pred, self.curlBy_col_pred, self.curlBz_col_pred = \
            self.net_curl_curl_A(self.x_col_tf, self.y_col_tf, self.z_col_tf);
        ######## The total loss to be minimised
        #### Obs Constraint 
        self.bx_obs_res = self.bx_obs_pred - self.bx_obs_tf;
        self.by_obs_res = self.by_obs_pred - self.by_obs_tf;
        self.bz_obs_res = self.bz_obs_pred - self.bz_obs_tf;
        self.bx_obs_loss = tf.reduce_mean(tf.square(self.bx_obs_res));
        self.by_obs_loss = tf.reduce_mean(tf.square(self.by_obs_res));
        self.bz_obs_loss = tf.reduce_mean(tf.square(self.bz_obs_res));
        #### Lap(Ax,Ay,Az) Constraint 
        self.curlBx_col_res = self.curlBx_col_pred;
        self.curlBy_col_res = self.curlBy_col_pred;
        self.curlBz_col_res = self.curlBz_col_pred;
        self.curlBx_col_loss = tf.reduce_mean(tf.square(self.curlBx_col_res));
        self.curlBy_col_loss = tf.reduce_mean(tf.square(self.curlBy_col_res));
        self.curlBz_col_loss = tf.reduce_mean(tf.square(self.curlBz_col_res));
        #### Loss
        self.obs_loss = self.w_obs_tf * (self.bx_obs_loss + self.by_obs_loss + self.bz_obs_loss);
        self.curlB_col_loss = self.curlBx_col_loss + self.curlBy_col_loss + self.curlBz_col_loss;
        self.loss =  self.obs_loss + self.curlB_col_loss;
        self.true_loss = self.obs_loss/self.w_obs_tf + self.curlB_col_loss;
        ######## Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False);
        starter_learning_rate = 2e-3; 
        self.learning_rate = tf.compat.v1.train.exponential_decay(\
            starter_learning_rate, self.global_step, 30*1000, 0.8, staircase=False); # <2e-3 & 0.8>; <2e-4 & 0.95>
        ######## Passing global_step to minimize() will increment it at each step.
        if opti == 'Adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate);
            self.train_op=self.optimizer.minimize(self.loss, global_step=self.global_step);
        ######## Logger
        self.loss_log=[]; self.true_loss_log = [];
        self.bx_obs_loss_log = []; self.by_obs_loss_log = []; self.bz_obs_loss_log = [];
        self.curlBx_col_loss_log = []; self.curlBy_col_loss_log = []; self.curlBz_col_loss_log = []; 
        self.w_obs_log=[];  self.lr_log=[];
        ######## Saver
        self.saver=tf.compat.v1.train.Saver(\
            tf.compat.v1.trainable_variables(),max_to_keep=10**6);
        ######## Gradients Storage
        self.obs_grad = []; self.curlB_col_grad = [];
        for i in range(len(self.layers) - 1):
            self.obs_grad.append([]);
            self.obs_grad[i].append(tf.gradients(self.obs_loss, self.weights1[i])[0]);
            self.obs_grad[i].append(tf.gradients(self.obs_loss, self.weights2[i])[0]);
            self.obs_grad[i].append(tf.gradients(self.obs_loss, self.weights3[i])[0]);
            self.curlB_col_grad.append([]);
            self.curlB_col_grad[i].append(tf.gradients(self.curlB_col_loss, self.weights1[i])[0]);
            self.curlB_col_grad[i].append(tf.gradients(self.curlB_col_loss, self.weights2[i])[0]);
            self.curlB_col_grad[i].append(tf.gradients(self.curlB_col_loss, self.weights3[i])[0]);
        ######## Compute and store the dynamic weights
        self.mean_obs_grad_log = []; self.max_curlB_col_grad_log = [];
        for i in range(len(self.layers) - 1):
            self.mean_obs_grad_log.append(tf.reduce_mean(tf.abs(self.obs_grad[i][0]))); 
            self.mean_obs_grad_log.append(tf.reduce_mean(tf.abs(self.obs_grad[i][1]))); 
            self.mean_obs_grad_log.append(tf.reduce_mean(tf.abs(self.obs_grad[i][2]))); 
            self.max_curlB_col_grad_log.append(tf.reduce_max(tf.abs(self.curlB_col_grad[i][0]))); 
            self.max_curlB_col_grad_log.append(tf.reduce_max(tf.abs(self.curlB_col_grad[i][1]))); 
            self.max_curlB_col_grad_log.append(tf.reduce_max(tf.abs(self.curlB_col_grad[i][2]))); 
        self.mean_obs_grad = tf.reduce_mean(tf.stack(self.mean_obs_grad_log));
        self.max_curlB_col_grad = tf.reduce_max(tf.stack(self.max_curlB_col_grad_log));
        self.w_obs_hat = self.max_curlB_col_grad / self.mean_obs_grad;
        ######## Initialize Tensorflow variables
        init = tf.compat.v1.global_variables_initializer();
        self.sess.run(init);
    
    def xavier_init(self, size):
        '''This function provides Xavier initialisation 
        (https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)'''
        in_dim = size[0];
        out_dim = size[1];        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim));
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],\
            stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32);

    def initialize_NN(self, layers):
        '''Here, Xavier initialisation is used to initialise 
        the network weights and biases'''        
        weights = [];
        biases = [];
        num_layers = len(layers); 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]]);
            b = tf.Variable(tf.zeros([1,layers[l+1]],dtype=tf.float32),\
                dtype=tf.float32);
            weights.append(W);
            biases.append(b);        
        return weights, biases;

    def forward_pass(self, H, layers, weights, biases, actiFun):
        num_layers = len(layers);
        for l in range(0, num_layers - 2):
            W = weights[l];
            b = biases[l];
            if actiFun=='tanh':
                H = tf.tanh(tf.add(tf.matmul(H, W), b));
            if actiFun=='relu':
                H = tf.keras.activations.relu(tf.add(tf.matmul(H, W), b), alpha=0.1);
            if actiFun=='sigmoid':
                H = tf.keras.activations.sigmoid(tf.add(tf.matmul(H, W), b));
            if actiFun=='gelu':
                H = tf.keras.activations.gelu(tf.add(tf.matmul(H, W), b));
            if actiFun=='swish':
                H = tf.keras.activations.swish(tf.add(tf.matmul(H, W), b));
            if actiFun=='siren':
                H = tf.math.sin(tf.add(tf.matmul(H, W), b));
        W = weights[-1];
        b = biases[-1];
        H = tf.add(tf.matmul(H, W), b);
        return H;

    def net_curl_A(self, x, y, z):
        Ax = self.forward_pass(tf.concat([x,y,z],1),\
            self.layers,self.weights1,self.biases1,self.actiFun);
        Ay = self.forward_pass(tf.concat([x,y,z],1),\
            self.layers,self.weights2,self.biases2,self.actiFun);
        Az = self.forward_pass(tf.concat([x,y,z],1),\
            self.layers,self.weights3,self.biases3,self.actiFun);
        Bx = tf.gradients(Az, y)[0] - tf.gradients(Ay, z)[0];
        By = tf.gradients(Ax, z)[0] - tf.gradients(Az, x)[0];
        Bz = tf.gradients(Ay, x)[0] - tf.gradients(Ax, y)[0];
        return Bx, By, Bz;
    
    def net_curl_curl_A(self, x, y, z):
        Ax = self.forward_pass(tf.concat([x,y,z],1),\
            self.layers,self.weights1,self.biases1,self.actiFun);
        Ay = self.forward_pass(tf.concat([x,y,z],1),\
            self.layers,self.weights2,self.biases2,self.actiFun);
        Az = self.forward_pass(tf.concat([x,y,z],1),\
            self.layers,self.weights3,self.biases3,self.actiFun);
        curlBx = tf.gradients(tf.gradients(Ay, y)[0], x)[0] \
            + tf.gradients(tf.gradients(Az, z)[0], x)[0] \
            - tf.gradients(tf.gradients(Ax, y)[0], y)[0] \
            - tf.gradients(tf.gradients(Ax, z)[0], z)[0];
        curlBy = tf.gradients(tf.gradients(Ax, x)[0], y)[0] \
            + tf.gradients(tf.gradients(Az, z)[0], y)[0] \
            - tf.gradients(tf.gradients(Ay, x)[0], x)[0] \
            - tf.gradients(tf.gradients(Ay, z)[0], z)[0];
        curlBz = tf.gradients(tf.gradients(Ax, x)[0], z)[0] \
            + tf.gradients(tf.gradients(Ay, y)[0], z)[0] \
            - tf.gradients(tf.gradients(Az, x)[0], x)[0] \
            - tf.gradients(tf.gradients(Az, y)[0], y)[0];
        return curlBx, curlBy, curlBz;

    def train(self, nEpo, nBchPerEpo, BS, svPath, k, c, n0, dn):
        start_time = time.time();
        n_obs=self.xyzb_obs.shape[0];
        n_col=self.xyz_col.shape[0];
        RAD_obs=np.ones((n_obs,))/n_obs;
        RAD_col=np.ones((n_col,))/n_col;
        for it in range(nEpo*nBchPerEpo):
            epo=int((it+1)/nBchPerEpo);
            ######## Fetch obs mini-batches
            idx_obs = np.random.choice(n_obs, min(n_obs,BS), replace=False, p=RAD_obs);
            x_obs_batch=self.xyzb_obs[idx_obs,0:1];
            y_obs_batch=self.xyzb_obs[idx_obs,1:2];
            z_obs_batch=self.xyzb_obs[idx_obs,2:3];
            bx_obs_batch=self.xyzb_obs[idx_obs,3:4];
            by_obs_batch=self.xyzb_obs[idx_obs,4:5];
            bz_obs_batch=self.xyzb_obs[idx_obs,5:6];
            ######## Fetch col mini-batches
            idx_col = np.random.choice(n_col, min(n_col,BS), replace=False, p=RAD_col);
            x_col_batch=self.xyz_col[idx_col,0:1];
            y_col_batch=self.xyz_col[idx_col,1:2];
            z_col_batch=self.xyz_col[idx_col,2:3];
            ######## Define a dictionary for associating placeholders with data
            tf_dict = {self.x_obs_tf: x_obs_batch, self.y_obs_tf: y_obs_batch, 
                       self.z_obs_tf: z_obs_batch, self.bx_obs_tf: bx_obs_batch, 
                       self.by_obs_tf: by_obs_batch, self.bz_obs_tf: bz_obs_batch,
                       self.x_col_tf: x_col_batch, self.y_col_tf: y_col_batch, 
                       self.z_col_tf: z_col_batch,
                       self.w_obs_tf: self.w_obs};
            ######## Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict);
            ######## Print
            if (it+1) % nBchPerEpo == 0:
                elapsed = time.time() - start_time;
                ######## Loss terms
                loss_val = self.sess.run(self.loss, tf_dict);
                true_loss_val = self.sess.run(self.true_loss, tf_dict);
                bx_obs_loss_val, by_obs_loss_val, bz_obs_loss_val \
                    = self.sess.run([self.bx_obs_loss, self.by_obs_loss, self.bz_obs_loss],tf_dict);
                curlBx_col_loss_val, curlBy_col_loss_val, curlBz_col_loss_val \
                    = self.sess.run([self.curlBx_col_loss, self.curlBy_col_loss, self.curlBz_col_loss], tf_dict);
                self.loss_log.append(loss_val);
                self.true_loss_log.append(true_loss_val);
                self.bx_obs_loss_log.append(bx_obs_loss_val);
                self.by_obs_loss_log.append(by_obs_loss_val);
                self.bz_obs_loss_log.append(bz_obs_loss_val);
                self.curlBx_col_loss_log.append(curlBx_col_loss_val);
                self.curlBy_col_loss_log.append(curlBy_col_loss_val);
                self.curlBz_col_loss_log.append(curlBz_col_loss_val);
                ######## Dynamic weights
                self.w_obs_log.append(self.w_obs);
                ######## Learning Rate
                lr_val = self.sess.run(self.learning_rate, tf_dict);    
                self.lr_log.append(lr_val);
                print('Epo: %d, '%(epo)\
                    +'Loss: %.2e, '%(loss_val)\
                    +'TLoss: %.2e, '%(true_loss_val)\
                    +'LObs_bx: %.2e, '%(bx_obs_loss_val)\
                    +'LObs_by: %.2e, '%(by_obs_loss_val)\
                    +'LObs_bz: %.2e, '%(bz_obs_loss_val)\
                    +'LCol_curlBx: %.2e, '%(curlBx_col_loss_val)\
                    +'LCol_curlBy: %.2e, '%(curlBy_col_loss_val)\
                    +'LCol_curlBz: %.2e, '%(curlBz_col_loss_val)\
                    +'wObs: %.2e, '%(self.w_obs)\
                    +'Time: %.1f, '%(elapsed)\
                    +'LR: %.2e \n'%(lr_val), flush=True);
                ######## Update Dynamic weights
                w_obs_val = self.sess.run(self.w_obs_hat, tf_dict);
                self.w_obs = (1.0 - self.beta) * w_obs_val + self.beta * self.w_obs;
                start_time = time.time();
                self.saver.save(self.sess,save_path='./'+svPath+'/ckpt',\
                    global_step=epo,write_meta_graph=False);
            if ((it+1) % nBchPerEpo == 0) and (epo >= n0) and (epo % dn == 0):
                ######## Residula of obs
                tf_dict = {self.x_obs_tf: self.xyzb_obs[:,0:1], 
                           self.y_obs_tf: self.xyzb_obs[:,1:2], 
                           self.z_obs_tf: self.xyzb_obs[:,2:3],
                           self.bx_obs_tf: self.xyzb_obs[:,3:4], 
                           self.by_obs_tf: self.xyzb_obs[:,4:5], 
                           self.bz_obs_tf: self.xyzb_obs[:,5:6]};
                bx_obs_res_val, by_obs_res_val, bz_obs_res_val \
                    = self.sess.run([self.bx_obs_res, self.by_obs_res, self.bz_obs_res],tf_dict);
                obs_res_val=np.abs(bx_obs_res_val)+np.abs(by_obs_res_val)+np.abs(bz_obs_res_val);
                RAD_obs=obs_res_val**k/np.mean(obs_res_val**k)+c;
                RAD_obs=RAD_obs.reshape(-1,)/np.sum(RAD_obs);
                ######## Residula of curl_curl_A
                tf_dict = {self.x_col_tf: self.xyz_col[:,0:1], 
                           self.y_col_tf: self.xyz_col[:,1:2], 
                           self.z_col_tf: self.xyz_col[:,2:3]};
                curlBx_col_res_val, curlBy_col_res_val, curlBz_col_res_val \
                    = self.sess.run([self.curlBx_col_res, self.curlBy_col_res, self.curlBz_col_res], tf_dict);
                col_res_val=np.abs(curlBx_col_res_val)+np.abs(curlBy_col_res_val)+np.abs(curlBz_col_res_val);
                RAD_col=col_res_val**k/np.mean(col_res_val**k)+c;
                RAD_col=RAD_col.reshape(-1,)/np.sum(RAD_col);
                print('\n*****************************************\n',flush=True);
                print('Residual for [obs]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(obs_res_val),np.mean(obs_res_val),np.max(obs_res_val)), flush=True);
                print('Prob for [obs]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(RAD_obs),np.mean(RAD_obs),np.max(RAD_obs)), flush=True);
                print('Residual for [col]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(col_res_val),np.mean(col_res_val),np.max(col_res_val)), flush=True);
                print('Prob for [col]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(RAD_col),np.mean(RAD_col),np.max(RAD_col)), flush=True);
                print('\n*****************************************\n',flush=True);
        
    def predict_curl_A(self, x_star, y_star, z_star):
        x_star=self.cordScale*x_star+self.cordShift; 
        y_star=self.cordScale*y_star+self.cordShift; 
        z_star=self.cordScale*z_star+self.cordShift; 
        tf_dict = {self.x_obs_tf: x_star, self.y_obs_tf: y_star, self.z_obs_tf: z_star};
        bx_star, by_star, bz_star = self.sess.run(\
            [self.bx_obs_pred, self.by_obs_pred, self.bz_obs_pred], tf_dict);
        bx_star=(bx_star-self.magShift)/self.magScale; 
        by_star=(by_star-self.magShift)/self.magScale; 
        bz_star=(bz_star-self.magShift)/self.magScale; 
        return bx_star, by_star, bz_star;
    
    def predict_curl_curl_A(self, x_star, y_star, z_star):
        x_star=self.cordScale*x_star+self.cordShift; 
        y_star=self.cordScale*y_star+self.cordShift; 
        z_star=self.cordScale*z_star+self.cordShift; 
        tf_dict = {self.x_col_tf: x_star, self.y_col_tf: y_star, self.z_col_tf: z_star};
        curlBx_star, curlBy_star, curlBz_star = self.sess.run(\
            [self.curlBx_col_pred, self.curlBy_col_pred, self.curlBz_col_pred], tf_dict);
        return curlBx_star, curlBy_star, curlBz_star;

