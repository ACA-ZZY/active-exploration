from __future__ import annotations
import argparse
import copy
import json
import pickle
from importlib import import_module
import time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF
from uncertainty_benchmark.contracts import load_case_contract,available_cases
from uncertainty_benchmark.seed_blocks import SeedBlock
from uncertainty_benchmark.common_student import restore_student,pack_student,fit_base_student

RUN=Path("results/native")
SHARED_BLOCK=None
DIMS={c:load_case_contract(c).contract.dimension for c in available_cases()}

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)

def dump(path,value):
    write(path,(json.dumps(value,indent=2)+"\n").encode())

def npzout(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)

def input_ids(x):
    a=np.ascontiguousarray(x,dtype=np.float64)
    return np.array([row.tobytes().hex() for row in a])

def configure(c,device="cpu"):
    resolved=load_case_contract(c)
    return resolved.problem,resolved.contract,resolved.problem,SeedBlock,device

def label(c,b,role):
    if role!="shared.initial_train":raise ValueError(role)
    with np.load(SHARED_BLOCK/"shared/base_training.npz") as z:
        return z["x"],z["y"].reshape(-1)

def base_student(c,b,problem,device):
    with (SHARED_BLOCK/"shared/base_student.pkl").open("rb") as f:
        return restore_student(pickle.load(f),problem,device=device)

class Budget:
    def __init__(self):self.maximum=0;self.count=0
    def cap(self,task,maximum):self.maximum=int(maximum)
    def call(self,task,key,x,response):
        if self.count+len(x)>self.maximum:raise TimeoutError("additional-query budget reached")
        y=np.asarray(response(x)).reshape(-1)
        if not np.isfinite(y).all():raise FloatingPointError("nonfinite true response")
        self.count+=len(x)
        return y

def commit(c,b,lf,gp,x,y,n,checkpoint,root,status):
    dest=root/"models"/checkpoint
    saved_gp=copy.deepcopy(gp)
    if (getattr(saved_gp.optimizer,"__module__",None) == "__main__"
            and getattr(saved_gp.optimizer,"__name__",None) == "optimizer"):
        saved_gp.optimizer=import_module("external_adapters.mf_al").optimizer
    model=dict(lf=pack_student(lf),gp=saved_gp)
    write(dest/"model.pkl",pickle.dumps(model,protocol=5))
    npzout(dest/"training_rows.npz",x=x,y=y)
    row=dict(case_id=c,method_id="MF-AL-SS-Dhulipala",replicate=b,
             checkpoint=checkpoint,budget_actual=n,initial_train_count=len(x)-n,
             weights_path=str(dest/"model.pkl"),native_status=status,
             model_state_id=str(dest.resolve()))
    dump(dest/"model_registry.json",row)
    return row

class _ModelUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module == "__main__" and name in {"FrozenMF", "optimizer"}:
            return globals()[name]
        return super().find_class(module,name)

def restore_model(path,case,device="cpu"):
    with Path(path).open("rb") as f:model=_ModelUnpickler(f).load()
    if isinstance(model,dict):model=FrozenMF(model["lf"],model["gp"])
    model.lf=restore_student(model.lf,load_case_contract(case).problem,device=device)
    return model

def optimizer(obj,theta,bounds):
    r=minimize(obj,theta,method='L-BFGS-B',jac=True,bounds=bounds,options={'maxiter':1000})
    if not np.isfinite(r.fun):raise FloatingPointError('GP marginal likelihood nonfinite')
    return r.x,r.fun

def membership_probability(mu,sd,cut):
    mu=np.asarray(mu);sd=np.asarray(sd);out=np.asarray(mu>=cut,dtype=float);uncertain=sd>0
    out[uncertain]=ndtr((mu[uncertain]-cut)/sd[uncertain]);return out

def componentwise_step(x,rng):
    proposal=x+rng.normal(size=np.shape(x));accept=np.log(rng.random(np.shape(x)))<np.minimum(0.,-.5*(proposal**2-x**2));return np.where(accept,proposal,x)

class FrozenMF:
    def __init__(self,lf,gp):self.lf=lf;self.gp=gp
    def predict(self,x):return np.asarray(self.lf.predict(x)).reshape(-1)+self.gp.predict(x)

def run(c,b,work_root=None,max_seconds=86400,information_cap=None,reuse_root=None):
    started=time.monotonic();m='MF-AL-SS-Dhulipala';root=Path(work_root) if work_root else RUN/'new_runs'/c/m/f'block_{b:03d}';root.mkdir(parents=True,exist_ok=True)
    if (root/'native_result.json').exists():return
    src,contract,problem,S,device=configure(c,device='cpu');lf=base_student(c,b,problem,'cpu')
    x0,y0=label(c,b,'shared.initial_train');trainx=x0.copy();trainy=y0.copy();lfy=np.asarray(lf.predict(x0));residual=trainy-lfy
    direction=1 if c=='hysteretic-lab' else -1;event=.38531750785792235 if direction==1 else 0.
    task=f'native:{c}:{m}:b{b}'+(':tail_resume_20260923' if work_root else '');budget=Budget();budget.cap(task,information_cap or 50000);rng=np.random.default_rng(S(2026090701,b).seed('native_pipeline',method=m))
    historical={}
    if reuse_root:
        for path in (Path(reuse_root)/'queries').glob('*.npz'):
            with np.load(path) as z:
                for key,value in zip(input_ids(z['x']),np.asarray(z['y']).reshape(-1)):historical[key]=float(value)
    gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(1e-4,1e4))*RBF(1.,(1e-2,1e2)),alpha=1e-8,optimizer=optimizer,normalize_y=True,n_restarts_optimizer=0)
    fit_start=time.perf_counter();gp.fit(trainx,residual);fit_seconds=time.perf_counter()-fit_start
    cache={k:(float(y),float(v)) for k,y,v in zip(input_ids(x0),y0,lfy)};n_new=0;proposals=0;levels=[];pop=None;gp_calls=0;lf_calls=len(x0)
    if work_root:commit(c,b,lf,gp,trainx,trainy,0,'tail_N00000',root,'native_not_finished')
    def evaluate(x,cut):
        nonlocal gp,n_new,trainx,trainy,residual,proposals,fit_seconds,gp_calls,lf_calls
        if time.monotonic()-started>max_seconds:raise TimeoutError('registered native time limit' if work_root else 'registered 14400 second native limit')
        proposals+=1
        if proposals>50000:raise TimeoutError('registered 50000 proposal limit')
        key=input_ids(x[None,:])[0]
        if key in cache:return direction*cache[key][0],direction*cache[key][0],0.,True
        fl=float(lf.predict(x[None,:])[0]);lf_calls+=1;corr,sd=gp.predict(x[None,:],return_std=True);gp_calls+=1
        mu=direction*(fl+float(corr[0]));sd=float(sd[0])
        if not np.isfinite([mu,sd]).all():raise FloatingPointError('nonfinite GP fidelity decision')
        u=abs(mu-cut)/sd if sd>0 else (0. if mu==cut else float('inf'))
        if u<2:
            q=root/'queries'/f'{n_new+1:05d}.npz'
            if q.exists():
                with np.load(q) as z:
                    if not np.array_equal(z['x'],x[None,:]):raise RuntimeError('resume proposal drift')
                    y=float(z['y'][0])
            elif key in historical:
                y=historical[key];npzout(q,x=x[None,:],y=np.array([y]),lf=np.array([fl]),threshold=np.array([cut]),u=np.array([u]),new_physical_hf=np.array([0]))
            else:
                y=float(budget.call(task,f'{task}:q{n_new+1}',x[None,:],src.response)[0]);npzout(q,x=x[None,:],y=np.array([y]),lf=np.array([fl]),threshold=np.array([cut]),u=np.array([u]))
            cache[key]=(y,fl);trainx=np.vstack([trainx,x]);trainy=np.r_[trainy,y];residual=np.r_[residual,y-fl];n_new+=1
            gp.kernel=gp.kernel_;t=time.perf_counter();gp.fit(trainx,residual);fit_seconds+=time.perf_counter()-t
            if work_root and (n_new in (1,25) or n_new%100==0 and n_new%500!=0):commit(c,b,lf,gp,trainx,trainy,n_new,f'tail_N{n_new:05d}',root,'native_not_finished')
            if n_new in (500,1000,1500,2000):commit(c,b,lf,gp,trainx,trainy,n_new,f'R{n_new//500}',root,'native_not_finished')
            if information_cap and n_new>=information_cap:
                commit(c,b,lf,gp,trainx,trainy,n_new,'native_budget',root,'resource_limited')
                raise TimeoutError('additional-query budget reached')
            if n_new%25==0:print(json.dumps(dict(case_id=c,replicate=b,new_hf=n_new,proposals=proposals,fit_seconds=fit_seconds)),flush=True)
            return direction*y,direction*y,0.,True
        return mu,mu,sd,False
    raw_product=1.;corrected_product=1.;cov2=0.
    for level in range(50):
        if pop is None:
            xx=[];val=[];mean=[];std=[];known=[];chains=[]
            for i in range(1000):
                cut=min(event,float(np.quantile(val if val else direction*y0,.9)))
                point=rng.standard_normal(DIMS[c]);v,mu,s,k=evaluate(point,cut)
                xx.append(point);val.append(v);mean.append(mu);std.append(s);known.append(k);chains.append(i)
        else:
            order=np.argsort(-pop['value'],kind='stable')[:100];source_cut=float(pop['value'][order[-1]])
            xx=list(pop['x'][order].copy());val=list(pop['value'][order]);mean=list(pop['mu'][order]);std=list(pop['sd'][order]);known=list(pop['known'][order]);chains=list(range(100))
            last=list(range(100))
            for i in range(900):
                chain=i%100;parent=last[chain];point=componentwise_step(xx[parent],rng);cut=min(event,float(np.quantile(val,.9)));v,mu,s,k=evaluate(point,cut)
                if v<source_cut:point=np.array(xx[parent],copy=True);v=val[parent];mu=mean[parent];s=std[parent];k=known[parent]
                xx.append(point);val.append(v);mean.append(mu);std.append(s);known.append(k);chains.append(chain);last[chain]=len(val)-1
        pop=dict(x=np.asarray(xx),value=np.asarray(val),mu=np.asarray(mean),sd=np.asarray(std),known=np.asarray(known),chain_id=np.asarray(chains))
        cut=min(event,float(np.sort(pop['value'])[-100]));probs=membership_probability(pop['mu'],pop['sd'],cut)
        cond=float(probs.mean());rawcond=float(np.mean(pop['value']>=cut));corrected_product*=cond;raw_product*=rawcond
        if level==0:variance=cond*(1-cond)/1000
        else:
            chain_means=np.array([probs[pop['chain_id']==k].mean() for k in range(100)]);variance=float(np.var(chain_means,ddof=1)/100)
        cov2+=variance/cond**2 if cond>0 else float('inf')
        npzout(root/'levels'/f'level_{level:02d}.npz',**pop,membership_probability=probs,threshold=np.array([cut]))
        levels.append(dict(level=level,threshold=cut,conditional_probability=cond,hard_indicator_conditional_probability=rawcond,conditional_variance=variance,states=1000,unique_inputs=len(set(input_ids(pop['x']))),new_hf=n_new))
        if cut>=event:
            terminal=commit(c,b,lf,gp,trainx,trainy,n_new,'native_terminal',root,'native_finished')
            for r in range(1,5):
                if n_new<=r*500 and not (root/'models'/f'R{r}'/'model_registry.json').exists():
                    row=dict(terminal,checkpoint=f'R{r}',budget_target=r*500,budget_match_status='native_early_stop_at_most_B');dump(root/'models'/f'R{r}'/'model_registry.json',row)
            result=dict(case_id=c,method_id=m,replicate=b,status='native_finished',readout_kind='native_system',dataset_id=None,physical_event=event,direction=direction,pf_estimate=corrected_product,pf_hard_indicator_auxiliary=raw_product,relative_standard_error=np.sqrt(cov2),variance_method='independent-chain cluster variance; cross-level covariance not included',new_information_labels=n_new,initial_labels=len(x0),lf_model_unchanged=True,lf_forward_rows=lf_calls,gp_prediction_calls=gp_calls,gp_fit_seconds=fit_seconds,proposals=proposals,levels=levels,model_state_id=terminal['model_state_id'],run_manifest=str(root/'native_result.json'),wall_seconds=time.monotonic()-started)
            dump(root/'native_result.json',result);print(json.dumps(result),flush=True);return
    raise TimeoutError('50-level resource limit; event not reached')

def main(argv=None):
    global SHARED_BLOCK
    parser=argparse.ArgumentParser()
    parser.add_argument("case",choices=available_cases())
    parser.add_argument("--block",type=int,required=True)
    parser.add_argument("--shared-block",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--timeout-seconds",type=float,default=86400)
    parser.add_argument("--additional-budget",type=int,default=50000)
    args=parser.parse_args(argv)
    if not 0 < args.additional_budget <= 50000 or args.timeout_seconds <= 0:
        raise ValueError("invalid query budget or time limit")
    SHARED_BLOCK=args.shared_block.resolve()
    request=json.loads((SHARED_BLOCK/"request.json").read_text())["request"]
    if request["case"]!=args.case or int(request["block"])!=args.block:
        raise ValueError("shared block belongs to another case or run")
    try:
        run(args.case,args.block,args.out,args.timeout_seconds,args.additional_budget)
    except TimeoutError as exc:
        dump(args.out/"resource_limit.json",dict(status="resource_limited",reason=str(exc)))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
