# clear;
# python train.py duovae 2d --output-dir "ablation/output_1" --param-path "ablation/parameters/parameters_1.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation/output_2" --param-path "ablation/parameters/parameters_2.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation/output_3" --param-path "ablation/parameters/parameters_3.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation/output_4" --param-path "ablation/parameters/parameters_4.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation/output_5" --param-path "ablation/parameters/parameters_5.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation/output_6" --param-path "ablation/parameters/parameters_6.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation/output_7" --param-path "ablation/parameters/parameters_7.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation/output_8" --param-path "ablation/parameters/parameters_8.json" --subset-dataset True --load-dir "ablation/output_8/model2" --starting-epoch 1750
# python train.py duovae 2d --output-dir "ablation/output_9" --param-path "ablation/parameters/parameters_9.json" --subset-dataset True

clear;
# python train.py duovae 2d --output-dir "ablation2_best/output_1" --param-path "ablation2_best/parameters/parameters_1.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_2" --param-path "ablation2_best/parameters/parameters_2.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_3" --param-path "ablation2_best/parameters/parameters_3.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_4" --param-path "ablation2_best/parameters/parameters_4.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_5" --param-path "ablation2_best/parameters/parameters_5.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_6" --param-path "ablation2_best/parameters/parameters_6.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_7" --param-path "ablation2_best/parameters/parameters_7.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_8" --param-path "ablation2_best/parameters/parameters_8.json" --subset-dataset True
# python train.py duovae 2d --output-dir "ablation2_best/output_9" --param-path "ablation2_best/parameters/parameters_9.json" --subset-dataset True
python train.py duovae 2d --output-dir "ablation2_best/output_10" --param-path "ablation2_best/parameters/parameters_10.json" --subset-dataset True --load-dir "ablation2_best/output_10/model1" --starting-epoch 1500
python train.py duovae 2d --output-dir "ablation2_best/output_11" --param-path "ablation2_best/parameters/parameters_11.json" --subset-dataset True --load-dir "ablation2_best/output_11/model1" --starting-epoch 1500
python train.py duovae 2d --output-dir "ablation2_best/output_12" --param-path "ablation2_best/parameters/parameters_12.json" --subset-dataset True --load-dir "ablation2_best/output_12/model1" --starting-epoch 1500

