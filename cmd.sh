if [ "$#" -lt 2 ]; then
    echo "エラー: 引数が足りません。最初の2つの引数は必須です。" >&2
    return 1
fi

gpu_i=$1
exec_num=$2

conda activate sta

CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=$exec_num  python run_steps.py

# conserve $gpu_i 