SEED=$(date +%s)
echo $SEED >> random_seeds.txt
echo "Using seed: $SEED"

python experiments/train_synthetic.py --config experiments/configs/dd_euclidean.yaml --log_timestamp table1 --embed_dim 3 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 3 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_euclidean.yaml --log_timestamp table1 --embed_dim 5 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 5 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_euclidean.yaml --log_timestamp table1 --embed_dim 10 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 10 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_euclidean.yaml --log_timestamp table1 --embed_dim 20 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 20 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_euclidean.yaml --log_timestamp table1 --embed_dim 256 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 256 --seed $SEED

python experiments/train_synthetic.py --config experiments/configs/dd_poincare.yaml --log_timestamp table1 --embed_dim 3 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 3 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_poincare.yaml --log_timestamp table1 --embed_dim 5 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 5 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_poincare.yaml --log_timestamp table1 --embed_dim 10 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 10 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_poincare.yaml --log_timestamp table1 --embed_dim 20 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 20 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_poincare.yaml --log_timestamp table1 --embed_dim 256 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 256 --seed $SEED

python experiments/train_synthetic.py --config experiments/configs/dd_lorentz.yaml --log_timestamp table1 --embed_dim 3 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 3 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_lorentz.yaml --log_timestamp table1 --embed_dim 5 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 5 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_lorentz.yaml --log_timestamp table1 --embed_dim 10 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 10 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_lorentz.yaml --log_timestamp table1 --embed_dim 20 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 20 --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/dd_lorentz.yaml --log_timestamp table1 --embed_dim 256 --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/dd_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 256 --seed $SEED