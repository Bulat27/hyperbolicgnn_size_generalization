SEED=$(date +%s)
echo $SEED >> random_seeds.txt
echo "Using seed: $SEED"

python experiments/train_synthetic.py --config experiments/configs/proteins_euclidean.yaml --log_timestamp table1 --embed_dim 3 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 3 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_euclidean.yaml --log_timestamp table1 --embed_dim 5 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 5 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_euclidean.yaml --log_timestamp table1 --embed_dim 10 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 10 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_euclidean.yaml --log_timestamp table1 --embed_dim 20 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 20 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_euclidean.yaml --log_timestamp table1 --embed_dim 256 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_euclidean.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 256 --dataset "proteins" --seed $SEED

python experiments/train_synthetic.py --config experiments/configs/proteins_poincare.yaml --log_timestamp table1 --embed_dim 3 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 3 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_poincare.yaml --log_timestamp table1 --embed_dim 5 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 5 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_poincare.yaml --log_timestamp table1 --embed_dim 10 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 10 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_poincare.yaml --log_timestamp table1 --embed_dim 20 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 20 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_poincare.yaml --log_timestamp table1 --embed_dim 256 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_poincare.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 256 --dataset "proteins" --seed $SEED

python experiments/train_synthetic.py --config experiments/configs/proteins_lorentz.yaml --log_timestamp table1 --embed_dim 3 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 3 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_lorentz.yaml --log_timestamp table1 --embed_dim 5 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 5 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_lorentz.yaml --log_timestamp table1 --embed_dim 10 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 10 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_lorentz.yaml --log_timestamp table1 --embed_dim 20 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 20 --dataset "proteins" --seed $SEED
python experiments/train_synthetic.py --config experiments/configs/proteins_lorentz.yaml --log_timestamp table1 --embed_dim 256 --dataset "proteins" --seed $SEED
python experiments/evaluate_synthetic.py --config experiments/configs/proteins_lorentz.yaml --csv_file table1.csv --log_timestamp table1 --embed_dim 256 --dataset "proteins" --seed $SEED