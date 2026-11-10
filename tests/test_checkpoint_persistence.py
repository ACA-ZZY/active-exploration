from pathlib import Path
import os,subprocess,sys
import pytest
ROOT=Path(os.environ.get('RELEASE_ROOT',Path(__file__).resolve().parents[1]))
def run(code,*args):
 env=dict(os.environ,PYTHONPATH=str(ROOT/'benchmarks'),PYTHONDONTWRITEBYTECODE='1')
 p=subprocess.run([sys.executable,'-c',code,*map(str,args)],env=env,text=True,capture_output=True)
 assert p.returncode==0,p.stdout+p.stderr

def test_fresh_process_loads_recurrent_checkpoint(tmp_path):
 p=tmp_path/'model.pkl'
 run('''import pickle,sys,torch
from protocol.surrogate import _torch_modules
m=_torch_modules()[1](2,4,3,1,True)
with open(sys.argv[1],'wb') as f:pickle.dump(m,f)
''',p)
 run('''import pickle,sys,torch
with open(sys.argv[1],'rb') as f:m=pickle.load(f)
assert m(torch.zeros(2,2)).shape==(2,1)
''',p)

def test_mf_commit_is_readable_from_a_new_process(tmp_path):
 run('''import sys,types,numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from protocol.surrogate import Surrogate,NetworkRecipe
from external_adapters import mf_al
exec('class FrozenMF:\\n def __init__(self,lf,gp):self.lf=lf;self.gp=gp',globals())
mf_al.FrozenMF=FrozenMF
s=Surrogate(seed=1,recipe=NetworkRecipe(hidden_layer_sizes=(3,)),device='cpu')
s._model=s._shallow_module(2);s._scaler=StandardScaler().fit(np.zeros((2,2)));s._y_scaler=StandardScaler().fit(np.zeros((2,1)))
gp=GaussianProcessRegressor().fit(np.array([[0.,0.],[1.,1.]]),np.zeros(2))
mf_al.commit('two-sided-kink',0,s,gp,np.zeros((2,2)),np.zeros(2),0,'native_terminal',Path(sys.argv[1]),'complete')
''',tmp_path)
 run('''import sys,numpy as np
from external_adapters.mf_al import restore_model
m=restore_model(sys.argv[1],'two-sided-kink')
assert np.isfinite(m.predict(np.zeros((2,2)))).all()
''',tmp_path/'models/native_terminal/model.pkl')

def test_beam_reference_is_tracked_or_not_ignored():
 p=subprocess.run(['git','check-ignore','benchmarks/cases/cantilever_beam_2d/sweep/1000/manifest.json'],cwd=ROOT,capture_output=True)
 assert p.returncode==1,p.stdout.decode()
 assert (ROOT/'benchmarks/cases/cantilever_beam_2d/sweep/1000/manifest.json').is_file()

def test_main_recurrent_refinement_uses_the_recorded_learning_rate():
 run('''import numpy as np,torch
from dataclasses import replace
from protocol.surrogate import Surrogate,RECIPES
rates=[]
original=torch.optim.Adam
def record(parameters,**kw):
 rates.append(kw['lr']);return original(parameters,**kw)
torch.optim.Adam=record
torch.set_num_threads(1)
r=replace(RECIPES['recurrent'],steps=4,hidden=3,layers=1,max_iter=1,refine_iter=1)
s=Surrogate(seed=1,recipe=r,device='cpu')
x=np.arange(12,dtype=float).reshape(6,2)
s.fit(x,np.arange(6,dtype=float));s.continue_fit(x,np.arange(6,dtype=float),new_labels=2)
assert rates==[0.003,0.0005],rates
''')
