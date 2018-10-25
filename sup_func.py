import numpy as np
import h5py
from numba import jit,vectorize

catalogue_cache_path = "/gpfs/data/ddsm49/GALFORM/Cache2/abs_mag/"
mag_min = -20.0
mag_max = -19.5

MXXL_path = catalogue_cache_path+"MXXL_"+str(mag_min)+"_"+str(mag_max)+".h5"
GALFORM_path = catalogue_cache_path+"GALF_"+str(mag_min)+"_"+str(mag_max)+".h5"
Alt_GALFORM_path = catalogue_cache_path+"Altg_"+str(mag_min)+"_"+str(mag_max)+".h5"
picsave_path = "./pics/"

grid_level_ra = [0.001,0.003,0.009,0.027,0.081,0.243,0.729,2.187,6.561,19.683]
grid_level_dec = [0.001,0.003,0.009,0.027,0.081,0.243,0.729,2.187,6.561,19.683]

#level setting, in degrees

@jit(nopython = True)
def R2D(rad):
    return 180.0*rad/np.pi

@jit(nopython = True)
def D2R(deg):
    return np.pi*deg/180.0

@jit(nopython = True)
def AngDis1(ra1,ra2,dec1,dec2):
    in1 = np.sin(dec1)*np.sin(dec2)
    in2 = np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.arccos(in1+in2)

@jit(nopython = True)
def AngDis2(ra1,ra2,dec1,dec2):
    in1 = np.sin((dec1-dec2)/2)
    in2 = np.cos(dec1)*np.cos(dec2)*np.sin(ra1-ra2)*np.sin(ra1-ra2)
    return 2*np.arcsin(np.sqrt(in1**2 + in2))

@jit(nopython = True)
def AngDis(ra1,ra2,dec1,dec2):
    q1 = ra1 - ra2
    q2 = dec1 - dec2
    r1 = q1/q2
    p1 = q1*q2
    
    if p1 < (np.pi/180.0)**2:
        return AngDis2(ra1,ra2,dec1,dec2)
    elif r1 < 1/360000:
        return np.abs(dec1 - dec2)
    elif r1 > 360000:
        return np.abs(ra1 - ra2)
    else:
        return AngDis1(ra1,ra2,dec1,dec2)

# gnr = grid numbers of ra, gnd = grid numbers of dec
# axr = discretised axis of ra, axd = discretised axis of dec
# gdm = grid dec min, grm = grid ra min
# gds = grid dec sep, grm = grid ra sep

@jit(nopython = True)
def EucProd(x,y):
    return np.stack(np.meshgrid(x,y),-1).T.reshape(-1,2)

@jit(nopython = True)
def ResCal(axmi,pnt,axs):
    return pnt/axs - axmi/axs - 0.5

@jit(nopython = True)
def Dscrtz(axm,cor,sep):
    return np.int(ResCal(axm,cor,sep))

# dra/ddec = discretised ra/dec

@jit(nopython = True)
def Ssdcor(RA,DEC,grmi,gdmi,grs,gds,ldec):
    dra = Dscrtz(grmi,RA,grs)
    ddec = Dscrtz(gdmi,DEC,gds)
    return (ldec+1)*dra + ddec

@jit(nopython = True)
def NovaCataA(ra,dec,gdmi,grmi,gds,grs,ldec):
    dcor = Ssdcor(ra[0],dec[0],grmi,gdmi,grs,gds,ldec)
    newcat = np.array([dcor])
    for i in range(1,len(ra)):
        dcor = Ssdcor(ra[i],dec[i],grmi,gdmi,grs,gds,ldec)
        newkatze = np.array([dcor])
        newcat = np.concatenate((newcat,newkatze))
    np.save('Gridpoints_'+str(gds)+'_'+str(grs)+'.npy',newcat)
    newcat = 0
    return newcat

@jit
def NovaCataB(ra,dec,gdmi,grmi,gds,grs,ldec):
    dcor = Ssdcor(ra[0],dec[0],grmi,gdmi,grs,gds,ldec)
    newcat = np.array([dcor])
    for i in range(1,len(ra)):
        dcor = Ssdcor(ra[i],dec[i],grmi,gdmi,grs,gds,ldec)
        newkatze = np.array([dcor])
        newcat = np.concatenate((newcat,newkatze))
    return newcat

# gix = grid indexes, here it is not unique
# for original catalogue mass = 1
# ugix,mass = np.unique(gix,return_counts=True)

@jit(nopython = True)
def AngDisStat(ugix,gix,ra,dec,mas,loopn,TyPe = 'DD'):
    for i in range(len(ugix)):
        bool = gix==ugix[i]
        rah = ra[bool]
        dech = dec[bool]
        mass = mas[bool]
        dis = AngDis(rah[0],rah[1],dech[0],dech[1])
        Dat = np.array([dis])
        for j1 in range(1,len(rah)-1):
            for j2 in range(j1+1,len(rah)):
                dis = AngDis(rah[j1],rah[j2],dech[j1],dech[j2])
                Dat = np.array([dis,mass[j1]*mass[j2]])
    np.save('Data_'+str(type)+'_'+str(loopn)+'.npy',Dat)
    return 0

@jit(nopython = True)
def AngDisStatDR(ugix,gix,ra,rar,dec,decr,mas,masr,loopn):
    for i in range(len(ugix)):
        bool = gix==ugix[i]
        rah = ra[bool]
        dech = dec[bool]
        rarh = rar[bool]
        decrh = decr[bool]
        mass = mas[bool]
        massr = masr[bool]
        dis = AngDis(rah[0],rarh[0],dech[0],decrh[0])
        dis01 = AngDis(rah[0],rarh[1],dech[0],decrh[1])
        dis10 = AngDis(rah[1],rarh[0],dech[1],decrh[0])
        Dat = np.array([dis,dis01,dis10])
        for j1 in range(1,len(rah)-1):
            for j2 in range(1,len(rarh)):
                dis = AngDis(rah[j1],rah[j2],dech[j1],dech[j2])
                Dat = np.array([dis,mass[j1]*massr[j2]])
    np.save('Data_DR_'+str(loopn)+'.npy',Dat)
    return 0

@jit(nopython = True)
def InvRC(axmi,pnt,axs):
    return axs*(pnt+0.5) + axmi + 0.5*axs

@jit(nopython = True)
def Index_Decomp_ra(mi,ind,ld,sep)
    return InvRC(mi,ind//ld,sep)

@jit(nopython = True)
def Index_Decomp_dec(mi,ind,ld,sep)
    return InvRC(mi,ind%ld,sep)

@jit
def forward(ugix,ldec,gdmi,grmi,gds,grs):
    l = len(ugix)
    ra = np.zeros(l)
    dec = np.zeros(l)
    for i in range(l):
        ra[i] = Index_Decomp_ra(grmi,ugix[i],ldec,grs)
        dec[i] = Index_Decomp_dec(gdmi,ugix[i],ldec,gds)
    return ra,dec

@jit
def Randomra(ramin,ramax,n):
    raspan = ramax-ramin
    return raspan*np.random.rand(n) + ramin

@jit
def Randomdec(decmin,decmax,n):
    sdmin = np.sin(decmin)
    sdmax = np.sin(decmax)
    sdspan = sdmax-sdmin
    return np.arcsin(sdspan*np.random.rand(n)+sdmin)

def Premain(x):
    return 0

@jit
def Corrs(DecSep,RaSep,ra,dec,lev):
    
    ramin = np.min(ra) - RaSep
    ramax = np.max(ra) + RaSep
    decmin = np.min(np.sin(dec)) - DecSep
    decmax = np.max(np.sin(dec)) + DecSep
    char_len = len(ra)

    # DD

    ldec = np.int((decmax-decmin)/DecSep + 1)
    gix = NovaCataB(ra,dec,decmin,ramin,DecSep,RaSep,ldec)
    ugix,mass = np.unique(gix,return_counts=True)
    mas = np.ones(char_len)
    AngDisStat(ugix,gix,ra,dec,mas,lev)
    
    # DR
    
    rar = Randomra(ramin,ramax,char_len)
    decr = Randomdec(decmin,decmax,char_len)
    gixr = NovaCataB(rar,decr,decmin,ramin,DecSep,RaSep,ldec)
    ugixr = np.unique(gixr)
    ugixpr = np.intersect1d(ugix,ugixr,assume_unique=True)
    AngDisStatDR(ugixpr,gix,ra,rar,dec,decr,mas,mas,lev)
    del gix,ugixpr,ra,dec;gc.collect()
    
    # RR
    
    AngDisStat(ugixr,gixr,rar,decr,mas,lev,TyPe = 'RR')
    del ugixr,gixr,rar,decr,mas;gc.collect()
    
    return forward(ugix,ldec,decmin,ramin,DecSep,RaSep)

def Lowestlevel(DecSep,RaSep,Catp):
    cat0 = h5py.File(Catp,'r')
    ra = D2R(np.array(cat0['ra']))
    dec = D2R(np.array(cat0['dec']))
    cat0.close()
    return Corrs(DecSep,RaSep,ra,dec,0)

def Main(DecSep,RaSep,Catp):
    RaSep = D2R(RaSep)
    DecSep = np.sin(D2R(DecSep))
    ra,dec = Lowestlevel(DecSep[0],RaSep[0],Catp)
    for i in range(1,len(DecSep)):
        ra,dec = Corrs(DecSep[i],RaSep[i],ra,dec,i)

Main(grid_level_dec,grid_level_ra,MXXL_path)
Main(grid_level_dec,grid_level_ra,GALFORM_path)
Main(grid_level_dec,grid_level_ra,alt_GALFORM_path)

'''
def Lowestlevel(DecSep,RaSep,Catp):
    cat0 = h5py.File(Catp,'r')
    ra = D2R(np.array(cat0['ra']))
    dec = D2R(np.array(cat0['dec']))
    cat0.close()
    
    ramin = np.min(ra) - RaSep
    ramax = np.max(ra) + RaSep
    decmin = np.min(np.sin(dec)) - DecSep
    decmax = np.max(np.sin(dec)) + DecSep
    char_len = len(ra)
    
    # DD

    ldec = np.int((decmax-decmin)/DecSep + 1)
    gix = NovaCataB(ra,dec,decmin,ramin,DecSep,RaSep,ldec)
    ugix,mass = np.unique(gix,return_counts=True)
    mas = np.ones(char_len)
    AngDisStat(ugix,gix,ra,dec,mas,0)
    
    # DR
    
    rar = Randomra(ramin,ramax,char_len)
    decr = Randomdec(decmin,decmax,char_len)
    gixr = NovaCataB(rar,decr,decmin,ramin,DecSep,RaSep,ldec)
    ugixr = np.unique(gixr)
    ugixpr = np.intersect1d(ugix,ugixr,assume_unique=True)
    AngDisStatDR(ugixpr,gix,ra,rar,dec,decr,mas,mas,0)
    del gix,ugixpr,ra,dec;gc.collect()

    # RR

    AngDisStat(ugixr,gixr,rar,decr,mas,0,TyPe = 'RR')
    del ugixr,gixr,rar,decr,mas;gc.collect()
    
    return forward(ugix,ldec,decmin,ramin,DecSep,RaSep)

@jit
def Higherlevel(DecSep,RaSep,ra,dec,lev):
    
    ramin = np.min(ra) - RaSep
    ramax = np.max(ra) + RaSep
    decmin = np.min(np.sin(dec)) - DecSep
    decmax = np.max(np.sin(dec)) + DecSep
    char_len = len(ra)

'''











