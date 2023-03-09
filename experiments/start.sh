 #!/usr/bin/env bash

python3 train.py 1 80 > a1.txt 2>&1 &
python3 train.py 2 80 > a2.txt 2>&1 &
python3 train.py 3 80 > a3.txt 2>&1 &
python3 train.py 4 80 > a4.txt 2>&1 &
python3 train.py all 80 > aall.txt 2>&1 &
python3 train.py all 130 > b1.txt 2>&1 &
python3 train.py all 200 > b2.txt 2>&1 &

wait