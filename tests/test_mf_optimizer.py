import os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get('RELEASE_ROOT',Path(__file__).resolve().parents[1]))
def execute(code,*args):
 p=subprocess.run([sys.executable,'-c',code,*map(str,args)],env=dict(os.environ,PYTHONPATH=str(ROOT/'benchmarks'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
 assert p.returncode==0,p.stdout+p.stderr
@pytest.mark.parametrize('legacy',[False,True])
def test_mf_custom_optimizer_survives_fresh_process(tmp_path,legacy):
 execute('''import sys,pickle,types,numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from protocol.surrogate import Surrogate,NetworkRecipe
from uncertainty_benchmark.common_student import pack_student
from external_adapters import mf_al
optimizer=types.FunctionType(mf_al.optimizer.__code__,mf_al.optimizer.__globals__,'optimizer')
optimizer.__module__='__main__'
s=Surrogate(seed=1,recipe=NetworkRecipe(hidden_layer_sizes=(3,)),device='cpu')
s._model=s._shallow_module(2);s._scaler=StandardScaler().fit(np.zeros((2,2)));s._y_scaler=StandardScaler().fit(np.zeros((2,1)))
x=np.array([[0.,0.],[1.,1.]]);gp=GaussianProcessRegressor(optimizer=optimizer).fit(x,np.zeros(2))
root=Path(sys.argv[1])
if sys.argv[2]=='True':
 class FrozenMF:
  def __init__(self,lf,gp):self.lf=lf;self.gp=gp
 path=root/'models/native_terminal';path.mkdir(parents=True)
 (path/'model.pkl').write_bytes(pickle.dumps(FrozenMF(pack_student(s),gp),protocol=5))
else:
 mf_al.commit('two-sided-kink',0,s,gp,x,np.zeros(2),0,'native_terminal',root,'complete')
 assert b'__main__' not in (root/'models/native_terminal/model.pkl').read_bytes()
 assert gp.optimizer is optimizer
np.save(root/'expected.npy',s.predict(x)+gp.predict(x))
''',tmp_path,legacy)
 execute('''import sys,numpy as np
from pathlib import Path
from external_adapters.mf_al import restore_model
root=Path(sys.argv[1]);m=restore_model(root/'models/native_terminal/model.pkl','two-sided-kink')
x=np.array([[0.,0.],[1.,1.]])
np.testing.assert_allclose(m.predict(x),np.load(root/'expected.npy'),rtol=0,atol=1e-12)
assert m.gp.optimizer.__module__=='external_adapters.mf_al'
''',tmp_path)
