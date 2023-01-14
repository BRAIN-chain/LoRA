rs="1 2 4 8 16 32 64 128 256 512 1024"
target="gpt2 gpt2-medium gpt2-large gpt2-xl"

for r in $rs
do
    for size in $target
    do
        res=`python test/gpt2.py --r $r --size $size`
        echo ${r}_${size}, ${res}
        echo ${res} > "test/results/${r}_${size}.txt"
    done
    res=`python test/gptj.py --r $r`
    echo ${r}_gptj, ${res}
    echo ${res} > "test/results/${r}_gptj.txt"
done
