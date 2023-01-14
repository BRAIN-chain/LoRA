rs="1 2 4 8 16 32 64 128 256 512 1024"
target="gpt2 gpt2-medium gpt2-large gpt2-xl"

for r in $rs
do
    for size in $target
    do
        echo $r $size
        python test/gpt2.py --r $r --size $size > "test/results/${r}_${size}.txt"
    done
    echo $r gptj
    python test/gptj.py --r $r > "test/results/${r}_gptj.txt"
done
