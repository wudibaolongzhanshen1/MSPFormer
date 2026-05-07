#export CUDA_VISIBLE_DEVICES=0

model_name=MSPFormer
root_path=/root/workspace/MyDataset/electricity
data=electricity
device=cuda:1

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336'_'6 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336'_'12 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336'_'24 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336'_'48 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id 96'_'6 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_96'_'12 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_96'_'24 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name cross_sequence_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_96'_'48 \
  --model $model_name \
  --data $data \
  --features MS \
  --patch_size 12 \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 320 \
  --dec_in 1 \
  --c_out 1 \
  --device $device \
  --des 'Exp' \
  --itr 1 \
