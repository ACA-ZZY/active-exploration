from pathlib import Path
import json
import numpy as np

def dir_fit(student,contract,x,y,initial_y,seed_value,round_index,save_dir):
    import torch
    from scipy.ndimage import convolve1d
    from uncertainty_benchmark.common_student import apply_refinement_policy
    nxt=student.clone_frozen(seed=seed_value);apply_refinement_policy(nxt,contract)
    device=nxt._torch_device()
    from external_adapters import dir_fds as utils
    FDS=utils.FDS
    lo,hi=float(np.min(initial_y)),float(np.max(initial_y));assert hi>lo
    bins=np.clip(np.floor((y-lo)/(hi-lo)*100),0,99).astype(np.int64)
    counts=np.bincount(bins,minlength=100).astype(float)
    smooth=convolve1d(np.sqrt(counts),utils.get_lds_kernel_window('gaussian',5,2),mode='constant')
    weights=1/smooth[bins];weights*=len(weights)/sum(weights)
    assert np.isfinite(weights).all() and np.all(weights>0)
    if nxt.recipe.name=='recurrent':features=nxt._scale_drive(nxt._driving(x)[0])
    else:features=nxt._scaler.transform(nxt._features(x)).astype(np.float32)
    xt=torch.from_numpy(features).to(device);yt=torch.from_numpy(nxt._y_scaler.transform(nxt._encode_target(y).reshape(-1,1)).astype(np.float32)).to(device)
    bt=torch.from_numpy(bins).to(device);wt=torch.from_numpy(weights.astype(np.float32)).to(device)
    members=nxt._model if isinstance(nxt._model,list) else [nxt._model];epochs=contract.refine_epochs
    stats=[]
    for k,member in enumerate(members):
        linear=[n for n in member.modules() if isinstance(n,torch.nn.Linear)][-1]
        fds=FDS(linear.in_features,bucket_num=100,bucket_start=0,start_update=0,start_smooth=1,kernel='gaussian',ks=5,sigma=1,momentum=.9).to(device)
        fds.kernel_window=fds.kernel_window.to(device)
        recurrent=nxt.recipe.name=='recurrent'
        rate=contract.refine_learning_rate
        opt=torch.optim.Adam(member.parameters(),lr=rate,weight_decay=0 if recurrent else nxt.recipe.weight_decay)
        gen=torch.Generator().manual_seed(seed_value+1000*k);n=len(y);bs=nxt.recipe.batch_size
        context={'mode':'raw','labels':None,'epoch':0,'captured':None}
        def hook(_layer,args):
            raw=args[0];context['captured']=raw.detach()
            if context['mode']=='smooth':return (fds.smooth(raw.clone(),context['labels'][:,None],context['epoch']),)
            return args
        handle=linear.register_forward_pre_hook(hook)
        try:
            for epoch in range(epochs):
                member.train();context['epoch']=epoch;context['mode']='smooth';order=torch.randperm(n,generator=gen).to(device)
                for start in range(0,n,bs):
                    idx=order[start:start+bs];context['labels']=bt[idx];opt.zero_grad();pred=member(xt[idx]);loss=((pred-yt[idx]).square().reshape(-1)*wt[idx]).mean()
                    if not torch.isfinite(loss):raise FloatingPointError('DIR nonfinite loss')
                    loss.backward()
                    if recurrent:torch.nn.utils.clip_grad_norm_(member.parameters(),nxt.recipe.grad_clip)
                    opt.step()
                context['mode']='raw';encodings=[]
                with torch.no_grad():
                    for start in range(0,n,bs):member(xt[start:start+bs]);encodings.append(context['captured'])
                    fds.update_last_epoch_stats(epoch);fds.update_running_stats(torch.cat(encodings),bt,epoch)
                if epoch%25==0:print(json.dumps(dict(round=round_index,epoch=epoch,loss=float(loss.detach().cpu()))),flush=True)
            stats.append({key:value.cpu().numpy() for key,value in fds.state_dict().items()})
        finally:handle.remove()
    save_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(save_dir/'lds_weights.npz',bin_id=bins,weight=weights,initial_bin_bounds=np.array([lo,hi]))
    for k,state in enumerate(stats):np.savez_compressed(save_dir/f'fds_state_member_{k}.npz',**state)
    nxt.training_calls.append(len(y))
    nxt.lineage += contract.train_per_round
    return nxt
