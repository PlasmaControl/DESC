from GamCFunc import GammaC
import os

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
name = "QA1"

bigGamma_c = GammaC(name, s = 0.5, nfulltransits = idx * 5, stepswithin1FP = 150, bpstep = 200)

file = name + "convtest.txt"
f = open(file, "a")
f.write(f"{nfulltransits:1.2f}, {bigGamma_c:1.3e}\n")
f.close()