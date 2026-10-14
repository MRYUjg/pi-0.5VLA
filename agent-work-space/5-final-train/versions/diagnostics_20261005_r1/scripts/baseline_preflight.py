import sys,pathlib,json,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'collection_20261005_r2/scripts'))
import collect
collect.AuditedRecorder=collect.core.Recorder
raise SystemExit(collect.main())
