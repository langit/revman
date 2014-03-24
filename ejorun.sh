#!/bin/bash
# simply run with
# ./ejorun.sh normal

#python2.6 randcov3.py $@ &

python2.6 cov3.py $@ &
python2.6 loadfactor3.py $@ &
python2.6 fareratio3.py $@

./multips2pdf.sh ORcov3[0-9].eps
./multips2pdf.sh ORloadfactor3[0-9].eps
./multips2pdf.sh ORfareratio3[0-9].eps
