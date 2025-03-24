for seed in 1 2 3 4 5
do
	python run/retraining.py --env-name Walker2dOOD-v3 --observation-type vector --cuda \
		--cuda-device 0 --num-steps 1000001 --policy sero --use-aux-reward --aux-coef 1 \
		--buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol-coef 0.3
done
