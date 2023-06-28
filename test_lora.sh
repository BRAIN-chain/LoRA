rs="1 2 4 8 16 32 64 128 256 512 1024"
target="gpt2 gpt2-medium gpt2-large gpt2-xl"

for r in $rs
do
    for size in $target
    do
        res=`python test/gpt2_lora.py --r $r --size $size`
        echo ${r}_${size}, ${res}
        echo ${res} > "test/results/lora/${r}_${size}.txt"
    done
    res=`python test/gptj_lora.py --r $r`
    echo ${r}_gptj, ${res}
    echo ${res} > "test/results/lora/${r}_gptj.txt"
done
