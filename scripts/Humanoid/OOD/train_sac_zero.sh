for seed in 1 2 3 4 5
do
python run/retraining.py --env-name HumanoidOOD-v3 --observation-type vector --cuda \
	--cuda-device 0 --num-steps 4000001 --policy sac --buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol-coef 0 --async-env --n-envs 5
done
