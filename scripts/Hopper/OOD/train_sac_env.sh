for seed in 1 2 3 4 5
do
python run/retraining.py --env-name HopperOOD-v3 --observation-type vector --cuda --use-env-reward \
	--cuda-device 0 --num-steps 1000001 --policy sac --buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol-coef 0
done
