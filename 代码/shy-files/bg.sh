if [ $# -eq 1 ]; then
    nohup python $1 > out.log 2>&1 & 
    ps aux | grep "python $1"
else echo "Please provide a script to run"
fi