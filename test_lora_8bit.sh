rs="1 2 4 8 16 32 64 128 256 512 1024"
target="gpt2 gpt2-medium gpt2-large gpt2-xl"

for r in $rs
do
    for size in $target
    do
        res=`python test/gpt2_lora_8bit.py --r $r --size $size`
        echo ${r}_${size}_8bit, ${res}
        echo ${res} > "test/results/lora_8bit/${r}_${size}_8bit.txt"
    done
    res=`python test/gptj_lora_8bit.py --r $r`
    echo ${r}_gptj_8bit, ${res}
    echo ${res} > "test/results/lora_8bit/${r}_gptj_8bit.txt"
done
