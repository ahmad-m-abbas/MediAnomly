num_repeat=1
datasets="rsna vin brain lag isic c16 brats"
# done: ae-l2 ae-l1 ae-ssim ae-perceptual ae-spatial ae-grad ceae vae vae-combi vae-rec constrained-ae memae ganomaly
methods="aeu" #  
gpu=0




for data in $datasets;do
  for method in $methods;do
    	python train.py -d "$data" -m "$method" -g 0 -f 0 --beta 0 --grad-pen-weight 0;
	python test.py -d "$data" -m "$method" -g 0 -f 0 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 1 --beta 0.001  --grad-pen-weight 0.00001 --train-weight-decay 0.00001;
	python test.py -d "$data" -m "$method" -g 0 -f 1 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 2 --beta 0.001  --grad-pen-weight 0.00001;
	#python test.py -d "$data" -m "$method" -g 0 -f 2 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 3 --beta 0.001  --grad-pen-weight 0.00001 --train-weight-decay 0.000005;
	python test.py -d "$data" -m "$method" -g 0 -f 3 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 4 --beta 0.0001 --grad-pen-weight 0.00005;
	python test.py -d "$data" -m "$method" -g 0 -f 4 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 5 --beta 0.0005 --grad-pen-weight 0.00005;
	python test.py -d "$data" -m "$method" -g 0 -f 5 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 6 --beta 0.0001 --grad-pen-weight 0.00005;
	python test.py -d "$data" -m "$method" -g 0 -f 6 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 7 --beta 0.0001 --grad-pen-weight 0.0001;
	python test.py -d "$data" -m "$method" -g 0 -f 7 -save;

	python train.py -d "$data" -m "$method" -g 0 -f 8 --beta 0.001  --grad-pen-weight 0.0001;
	python test.py -d "$data" -m "$method" -g 0 -f 8 -save;
  done

  #for((i=0;i<num_repeat;i=i+1));do
   #   python train.py -d "$data" -m dae -g $gpu --input-size 128 -bs 16 -f "$i";
    #  python test.py -d "$data" -m dae -g $gpu --input-size 128 -f "$i" -save;
  #done
done
