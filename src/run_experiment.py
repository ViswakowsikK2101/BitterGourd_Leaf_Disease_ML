"""Leakage-safe benchmark: handcrafted leaf features vs a compact CNN."""
from __future__ import annotations
import argparse, json, random, re, time
from pathlib import Path
import cv2, joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED=42; EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
CLASSES=[('bitter','anthracnose','Bitter_Gourd__Anthracnose'),('bitter','downy','Bitter_Gourd__Downy_Mildew'),('bitter','healthy','Bitter_Gourd__Healthy'),('okra','cercospora','Okra__Cercospora_Leaf_Spot'),('okra','healthy','Okra__Healthy'),('pumpkin','downy','Pumpkin__Downy_Mildew'),('pumpkin','healthy','Pumpkin__Healthy'),('ridge','downy','Ridge_Gourd__Downy_Mildew'),('ridge','healthy','Ridge_Gourd__Healthy')]
def seed_everything():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True, warn_only=True)
def label_for(p:Path):
 s=str(p).lower().replace('-',' ').replace('_',' ')
 for crop, disease, label in CLASSES:
  if crop in s and disease in s: return label
 return None
def inventory(root, limit):
 rows=[]
 for p in root.rglob('*'):
  if p.suffix.lower() in EXT and 'augmented' not in str(p).lower():
   label=label_for(p)
   if label: rows.append({'path':str(p.resolve()),'label':label})
 if not rows: raise RuntimeError('No labelled original images found. Point --data-root to extracted raw/original image folders.')
 df=pd.DataFrame(rows).sort_values('path')
 if limit: df=df.groupby('label',group_keys=False).head(limit)
 if df.label.nunique()!=9: print('WARNING: found',df.label.nunique(),'classes; expected 9. Check folder names.')
 return df.reset_index(drop=True)
def safe_imread(path, size=256):
 im=cv2.imread(path)
 if im is None: raise ValueError('Unreadable image: '+path)
 return cv2.resize(im,(size,size),interpolation=cv2.INTER_AREA)
def moments(x):
 x=x.astype(float).ravel(); mu=x.mean(); sd=x.std()+1e-8
 return [mu,sd,np.median(x),np.mean(((x-mu)/sd)**3)]
def glcm_feats(gray):
 q=(gray//32).astype(np.int32); out=[]
 for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
  a=q[max(0,dy):q.shape[0]+min(0,dy),max(0,dx):q.shape[1]+min(0,dx)]
  b=q[max(0,-dy):q.shape[0]-max(0,dy),max(0,-dx):q.shape[1]-max(0,dx)]
  m=np.zeros((8,8)); np.add.at(m,(a.ravel(),b.ravel()),1); m=(m+m.T); m/=m.sum()+1e-12
  i,j=np.indices(m.shape); contrast=(m*(i-j)**2).sum(); dis=(m*np.abs(i-j)).sum(); hom=(m/(1+(i-j)**2)).sum(); energy=np.sqrt((m*m).sum())
  mi=(m*i).sum(); mj=(m*j).sum(); si=np.sqrt((m*(i-mi)**2).sum()); sj=np.sqrt((m*(j-mj)**2).sum()); corr=(m*(i-mi)*(j-mj)).sum()/(si*sj+1e-12)
  out += [contrast,dis,hom,energy,corr]
 return out
def leaf_mask(bgr):
 hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV); sat=hsv[:,:,1]
 _,m=cv2.threshold(sat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8)); return m
def shape_feats(bgr):
 m=leaf_mask(bgr); cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 if not cs:return [0.]*16
 c=max(cs,key=cv2.contourArea); area=cv2.contourArea(c); peri=cv2.arcLength(c,True); x,y,w,h=cv2.boundingRect(c); hull=cv2.convexHull(c); hu=cv2.HuMoments(cv2.moments(c)).flatten(); hu=np.sign(hu)*np.log10(np.abs(hu)+1e-30)
 return [area/(256*256),peri/(4*256),w/(h+1e-8),area/(w*h+1e-8),area/(cv2.contourArea(hull)+1e-8),4*np.pi*area/(peri*peri+1e-8),*hu[:7],float(m.mean()/255),float(w/256),float(h/256)]
def features(path):
 b=safe_imread(path); rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB); hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV); gray=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY); v=[]
 for img in (rgb,hsv):
  for ch in cv2.split(img): v+=moments(ch); v+=list(cv2.calcHist([ch],[0],None,[16],[0,256]).ravel()/ch.size)
 lbp=np.zeros_like(gray)
 for bit,(dy,dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
  sh=np.roll(np.roll(gray,dy,0),dx,1); lbp|=((sh>=gray).astype(np.uint8)<<bit)
 v+=list(np.histogram(lbp,bins=16,range=(0,256),density=True)[0]); v+=glcm_feats(gray); v+=shape_feats(b)
 return np.asarray(v,dtype=np.float32)
def plot_cm(y, pred, labels, name, out):
 cm=confusion_matrix(y,pred,labels=labels,normalize='true'); fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1); fig.colorbar(im,ax=ax,label='Recall')
 ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=[x.replace('_',' ') for x in labels],yticklabels=[x.replace('_',' ') for x in labels],xlabel='Predicted',ylabel='True',title=name); plt.setp(ax.get_xticklabels(),rotation=45,ha='right'); fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)
def scores(y,p,prob=None,enc=None):
 s={'accuracy':accuracy_score(y,p),'macro_precision':precision_score(y,p,average='macro',zero_division=0),'macro_recall':recall_score(y,p,average='macro',zero_division=0),'macro_f1':f1_score(y,p,average='macro'),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_roc_auc_ovr':np.nan}
 if prob is not None and enc is not None:
  try:s['macro_roc_auc_ovr']=roc_auc_score(enc.transform(y),prob,multi_class='ovr',average='macro')
  except ValueError:s['macro_roc_auc_ovr']=np.nan
 return s
def plot_summary(res,out):
 metrics=[('accuracy','Accuracy'),('macro_precision','Precision'),('macro_recall','Recall'),('macro_f1','Macro-F1'),('macro_roc_auc_ovr','ROC-AUC')]; x=np.arange(len(metrics)); width=.82/max(len(res),1);fig,ax=plt.subplots(figsize=(13,6))
 for i,(_,row) in enumerate(res.iterrows()):ax.bar(x+(i-(len(res)-1)/2)*width,[row[k] for k,_ in metrics],width,label=row['model']+' | '+row['representation'])
 ax.set(xticks=x,xticklabels=[v for _,v in metrics],ylim=(0,1.05),ylabel='Held-out score',title='Model comparison on the held-out original-image test set');ax.legend(fontsize=7,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.45));fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,5));ax.bar(range(len(res)),res.fit_seconds,color='#3766a6');ax.set(xticks=range(len(res)),xticklabels=[r['model']+'\n'+r['representation'].replace(' ','\n') for _,r in res.iterrows()],ylabel='Seconds',title='Training time comparison');plt.setp(ax.get_xticklabels(),fontsize=7);fig.tight_layout();fig.savefig(out/'training_time.png',dpi=180);plt.close(fig)
class LeafDS(Dataset):
 def __init__(self,df,enc,aug=False):self.df=df.reset_index(drop=True);self.enc=enc;self.aug=aug
 def __len__(self):return len(self.df)
 def __getitem__(self,i):
  b=safe_imread(self.df.path[i],224); b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
  if self.aug:
   if random.random()<.5:b=np.fliplr(b).copy()
   if random.random()<.5:
    M=cv2.getRotationMatrix2D((112,112),random.uniform(-15,15),1); b=cv2.warpAffine(b,M,(224,224),borderMode=cv2.BORDER_REFLECT)
  x=torch.from_numpy(b.copy().transpose(2,0,1)).float()/255.; return x,int(self.enc.transform([self.df.label[i]])[0])
class TinyCNN(nn.Module):
 def __init__(self,n):
  super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.35),nn.Linear(128,n))
 def forward(self,x):return self.net(x)
def run_cnn(tr,va,te,enc,args,out):
 dev='cuda' if torch.cuda.is_available() else 'cpu'; model=TinyCNN(len(enc.classes_)).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); loss=nn.CrossEntropyLoss(); loaders=[DataLoader(LeafDS(x,enc,a),batch_size=args.batch_size,shuffle=a,num_workers=0) for x,a in [(tr,True),(va,False),(te,False)]]; best=(-1,None); hist=[]
 for ep in range(args.epochs):
  model.train(); total=0
  for x,y in loaders[0]: opt.zero_grad(); z=model(x.to(dev)); l=loss(z,y.to(dev));l.backward();opt.step();total+=l.item()*len(y)
  model.eval(); yy=[];pp=[]
  with torch.no_grad():
   for x,y in loaders[1]:yy+=y.tolist();pp+=model(x.to(dev)).argmax(1).cpu().tolist()
  f=f1_score(yy,pp,average='macro');hist.append({'epoch':ep+1,'train_loss':total/len(tr),'val_macro_f1':f})
  if f>best[0]:best=(f,{k:v.cpu().clone() for k,v in model.state_dict().items()})
 model.load_state_dict(best[1]); model.eval();yy=[];pp=[];prob=[]
 with torch.no_grad():
  for x,y in loaders[2]:
   logits=model(x.to(dev));yy+=y.tolist();pp+=logits.argmax(1).cpu().tolist();prob+=torch.softmax(logits,1).cpu().tolist()
 pd.DataFrame(hist).to_csv(out/'cnn_history.csv',index=False); h=pd.DataFrame(hist); plt.plot(h.epoch,h.train_loss,label='train loss');plt.plot(h.epoch,h.val_macro_f1,label='validation macro-F1');plt.legend();plt.xlabel('Epoch');plt.tight_layout();plt.savefig(out/'cnn_learning_curve.png',dpi=180);plt.close(); torch.save({'state_dict':model.state_dict(),'classes':enc.classes_.tolist()},out.parent/'models'/'tiny_cnn.pt'); return np.array(yy),np.array(pp),np.array(prob),dev
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,default=Path('results'));ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--max-per-class',type=int,default=0);args=ap.parse_args();seed_everything();args.output.mkdir(parents=True,exist_ok=True);(args.output.parent/'models').mkdir(exist_ok=True)
 df=inventory(args.data_root,args.max_per_class);df.to_csv(args.output/'dataset_manifest.csv',index=False);tr,tmp=train_test_split(df,test_size=.30,stratify=df.label,random_state=SEED);va,te=train_test_split(tmp,test_size=.50,stratify=tmp.label,random_state=SEED);enc=LabelEncoder().fit(df.label); X=np.stack([features(p) for p in df.path]); np.savez(args.output/'handcrafted_features.npz',X=X,y=enc.transform(df.label)); idx={p:i for i,p in enumerate(df.path)}; ix=lambda d:np.array([idx[p] for p in d.path]); Xtr,Xte=X[ix(tr)],X[ix(te)]; ytr,yte=tr.label.values,te.label.values
 models={'SVM Linear':SVC(kernel='linear',C=1,class_weight='balanced',probability=True,random_state=SEED),'SVM Polynomial':SVC(kernel='poly',C=1,degree=3,class_weight='balanced',probability=True,random_state=SEED),'SVM RBF':SVC(kernel='rbf',C=10,gamma='scale',class_weight='balanced',probability=True,random_state=SEED),'Random Forest':RandomForestClassifier(n_estimators=400,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED),'XGBoost':XGBClassifier(n_estimators=350,max_depth=7,learning_rate=.05,subsample=.85,colsample_bytree=.8,n_jobs=-1,random_state=SEED,eval_metric='mlogloss')}; rows=[]
 for name,est in models.items():
  for selected in (False,True):
   steps=[('scale',StandardScaler())];
   if selected:steps.append(('pca',PCA(n_components=.95,random_state=SEED)))
   steps.append(('model',est));pipe=Pipeline(steps);t=time.time();pipe.fit(Xtr,enc.transform(ytr) if name=='XGBoost' else ytr);pred=pipe.predict(Xte);pred=enc.inverse_transform(pred.astype(int)) if name=='XGBoost' else pred;prob=pipe.predict_proba(Xte);s=scores(yte,pred,prob,enc);s.update({'model':name,'representation':'PCA 95% variance' if selected else 'All handcrafted features','fit_seconds':time.time()-t,'test_samples':len(te)});rows.append(s);tag=re.sub(' ','_',name)+('_pca' if selected else '_all');joblib.dump(pipe,args.output.parent/'models'/f'{tag}.joblib');plot_cm(yte,pred,enc.classes_,f'{name} — {s["representation"]}',args.output/f'cm_{tag}.png')
 t=time.time();yy,pp,prob,dev=run_cnn(tr,va,te,enc,args,args.output);s=scores(enc.inverse_transform(yy),enc.inverse_transform(pp),prob,enc);s.update({'model':'Tiny CNN','representation':'Pixels + train-only augmentation','fit_seconds':time.time()-t,'test_samples':len(te),'device':dev});rows.append(s);plot_cm(enc.inverse_transform(yy),enc.inverse_transform(pp),enc.classes_,'Tiny CNN',args.output/'cm_tiny_cnn.png')
 res=pd.DataFrame(rows).sort_values('macro_f1',ascending=False);plot_summary(res,args.output);res.to_csv(args.output/'metrics.csv',index=False,float_format='%.4f');best_h=res[res.model!='Tiny CNN'].iloc[0];cnn=res[res.model=='Tiny CNN'].iloc[0]
 verdict='competitive' if best_h.macro_f1>=cnn.macro_f1-.03 else 'not competitive under the 3-point macro-F1 criterion'
 tex='\\begin{table}[htbp]\\centering\\caption{Held-out test performance (generated by run\\_experiment.py).}\\label{tab:results}\\begin{tabular}{llrrr}\\toprule Model & Representation & Accuracy & Macro-F1 & Balanced Acc.\\\\\\midrule\n'+''.join(f"{r.model} & {r.representation} & {r.accuracy:.3f} & {r.macro_f1:.3f} & {r.balanced_accuracy:.3f}\\\\\n" for _,r in res.iterrows())+'\\bottomrule\\end{tabular}\\end{table}\n'
 (args.output/'latex_metrics_table.tex').write_text(tex);(args.output/'conclusion.txt').write_text(f'Best handcrafted: {best_h.model} ({best_h.macro_f1:.4f}); CNN: {cnn.macro_f1:.4f}. Handcrafted features are {verdict}.\n');print(res.to_string(index=False));print((args.output/'conclusion.txt').read_text())
if __name__=='__main__':main()
