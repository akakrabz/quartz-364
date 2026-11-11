#!/usr/bin/env python3
"""Read TensorBoard event files (events.out.tfevents.*) without TensorFlow or TensorBoard.

usage: python3 -B tools/read_tfevents.py <logdir> <out.json>
<logdir> holds one sub-folder per run (as written by torch.utils.tensorboard.SummaryWriter);
the JSON maps run -> tag -> [[step, value], ...]. Used once to extract the Lecture 22
training runs (py-complete/Lecture22/logs) into tools/l22_logs.json for tools/figs_training_b.py.

File format: TFRecord framing (uint64 length, uint32 crc, payload, uint32 crc); each payload is an
Event protobuf (1: wall_time double, 2: step varint, 5: Summary); Summary.value (1) holds tag (1)
and simple_value (2, float32). Only scalars are read.
"""
import struct,sys,os,json,glob
def varint(b,i):
    r=0;s=0
    while True:
        c=b[i];i+=1
        r|=(c&0x7f)<<s;s+=7
        if not c&0x80:break
    return r,i
def fields(b):
    i=0;out=[]
    while i<len(b):
        k,i=varint(b,i);f,w=k>>3,k&7
        if w==0: v,i=varint(b,i)
        elif w==1: v=b[i:i+8];i+=8
        elif w==2:
            l,i=varint(b,i);v=b[i:i+l];i+=l
        elif w==5: v=b[i:i+4];i+=4
        else: raise ValueError(w)
        out.append((f,w,v))
    return out
def read(fn):
    b=open(fn,'rb').read();i=0;res={}
    wall0=None;walls={}
    while i<len(b):
        n=struct.unpack('<Q',b[i:i+8])[0];i+=12
        d=b[i:i+n];i+=n+4
        step=0;wall=None;summ=None
        for f,w,v in fields(d):
            if f==1 and w==1: wall=struct.unpack('<d',v)[0]
            if f==2 and w==0: step=v
            if f==5 and w==2: summ=v
        if wall0 is None: wall0=wall
        if summ:
            for f,w,v in fields(summ):
                if f==1:
                    tag=None;val=None
                    for f2,w2,v2 in fields(v):
                        if f2==1: tag=v2.decode()
                        if f2==2 and w2==5: val=struct.unpack('<f',v2)[0]
                        if f2==8 and w2==2:
                            for f3,w3,v3 in fields(v2):
                                if f3==5 and w3==2: val=struct.unpack('<f',v3[:4])[0]
                                if f3==4 and w3==2 and len(v3)>=4: val=struct.unpack('<f',v3[:4])[0]
                    res.setdefault(tag,[]).append((step,val,wall-wall0))
    return res
if __name__=='__main__':
    root=sys.argv[1];out={}
    for d in sorted(os.listdir(root)):
        fs=sorted(glob.glob(os.path.join(root,d,'events*')))
        if not fs:
            continue
        print(d,len(fs))
        for fn in fs:
            r=read(fn)
            out[d]={k:[(s,v) for s,v,_ in vv] for k,vv in r.items()}
            for k,vv in r.items():
                vals=[v for _,v,_ in vv]
                print('  ',k,len(vv),'steps',vv[0][0],'..',vv[-1][0],'first %.4f last %.4f min %.4f max %.4f'%(vals[0],vals[-1],min(vals),max(vals)),'time %.0fs'%vv[-1][2])
    json.dump(out,open(sys.argv[2],'w'))
