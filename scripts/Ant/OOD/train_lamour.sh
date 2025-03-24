for seed in 1 2 3 4 5
do
	python run/retraining.py --env-name AntOOD-v3 --observation-type vector --cuda \
		--cuda-device 0 --num-steps 1000001 --policy lamour --use-aux-reward --aux-coef 0.2 --alpha 1 \
		--buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol-coef 1 --retrain-lamour
done
