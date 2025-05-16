# SMD
python -u main.py --model DeepAR --data SMD --input_len 720 --label_len 48  --dropout 0.1

# MSL
python -u main.py --model DeepAR --data MSL --input_len 48 --label_len 24 --dropout 0.1

# SMAP
python -u main.py --model DeepAR --data SMAP --input_len 24 --label_len 12 --dropout 0.1

# SWaT
python -u main.py --model DeepAR --data SWaT --input_len 720 --label_len 48 --dropout 0.1

# PSM
python -u main.py --model DeepAR --data PSM --input_len 720 --label_len 48 --dropout 0.1

# WADI
python -u main.py --model DeepAR --data WADI --input_len 100 --label_len 30 --dropout 0.1

# MBA
python -u main.py --model DeepAR --data MBA --input_len 100 --label_len 30 --dropout 0.1

# UCR
python -u main.py --model DeepAR --data UCR --input_len 48 --label_len 24 --dropout 0.1

# NAB
python -u main.py --model DeepAR --data NAB --input_len 360 --label_len 30 --dropout 0.1

# MSDS
python -u main.py --model DeepAR --data MSDS --input_len 720 --label_len 30 --dropout 0.1