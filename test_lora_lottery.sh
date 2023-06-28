rs="1 2 4 8 16 32 64 128 256 512 1024"
target="gpt2 gpt2-medium gpt2-large gpt2-xl"

for r in $rs
do
    for size in $target
    do
        res=`python test/gpt2_lora_lottery.py --r $r --size $size`
        echo ${r}_${size}_lottery, ${res}
        echo ${res} > "test/results/lora_lottery/${r}_${size}_lottery.txt"
    done
    res=`python test/gptj_lora_lottery.py --r $r`
    echo ${r}_gptj_lottery, ${res}
    echo ${res} > "test/results/lora_lottery/${r}_gptj_lottery.txt"
done
