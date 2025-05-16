# SMD
python -u main.py --model Autoformer --data SMD --input_len 720 --label_len 48

# MSL
python -u main.py --model Autoformer --data MSL --input_len 48 --label_len 24

# SMAP
python -u main.py --model Autoformer --data SMAP --input_len 24 --label_len 12

# SWaT
python -u main.py --model Autoformer --data SWaT --input_len 720 --label_len 48

# PSM
python -u main.py --model Autoformer --data PSM --input_len 720 --label_len 48

# WADI
python -u main.py --model Autoformer --data WADI --input_len 100 --label_len 30

# MBA
python -u main.py --model Autoformer --data MBA --input_len 100 --label_len 30

# NAB
python -u main.py --model Autoformer --data NAB --input_len 360 --label_len 30

# MSDS
python -u main.py --model Autoformer --data MSDS --input_len 720 --label_len 30