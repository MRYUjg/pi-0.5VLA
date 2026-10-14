import argparse,sys
from collect import T
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
sys.argv=['real_object_teacher_r20.py','--case','can','--seed',str(a.seed),'--out',a.out]
T.main()
