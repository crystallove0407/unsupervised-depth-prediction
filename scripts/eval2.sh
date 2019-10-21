echo "10060"
python3 main.py -c config/test_dp_kitti.ini -t kitti \
    --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-10060
python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy

echo "20119"
python3 main.py -c config/test_dp_kitti.ini -t kitti \
    --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-20119
python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy

echo "30178"
python3 main.py -c config/test_dp_kitti.ini -t kitti \
    --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-30178
python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy

echo "40237"
python3 main.py -c config/test_dp_kitti.ini -t kitti \
    --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-40237
python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy

# echo "150886"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-150886
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy

# echo "160945"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-160945
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
#
# echo "171004"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-171004
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
#
# echo "181063"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-181063
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
#
# echo "191122"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-191122
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
#
# echo "201181"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-201181
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
#
# echo "211240"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-211240
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
#
# echo "221299"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-221299
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
#
# echo "231358"
# python3 main.py -c config/test_dp_kitti.ini -t kitti \
#     --restore_model=./KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-231358
# python kitti_eval/eval_depth.py        --split=eigen        --kitti_dir=/home/waterman/dataset/KITTI_eval/        --pred_file=./images/kitti/test_kitti.npy
