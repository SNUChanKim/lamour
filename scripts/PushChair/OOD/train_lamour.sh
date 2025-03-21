for seed in 1 2 3 4 5
do
	python run/retraining.py --env-name PushChairOOD-v1 --observation-type vector --cuda \
		--cuda-device 0 --batch-size 1024 --num-steps 4000001 --gamma 0.95 --policy lamour --use-aux-reward --aux-coef 0.01 --alpha 1 \
		--buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol-coef 0.5 --retrain-lamour
done
