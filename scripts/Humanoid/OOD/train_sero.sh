for seed in 1 2 3 4 5
do
	python run/retraining.py --env-name HumanoidOOD-v3 --observation-type vector --cuda \
		--cuda-device 0 --num-steps 10000001 --policy sero --use-aux-reward --aux-coef 0.5 --alpha 1 \
		--buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol-coef 0.5 --async-env --n-envs 5
done
