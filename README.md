
### Training model
```
PYTHONPATH=`pwd` python3 main.py --output_model_dir  korea-multilingual-cased  --language ko --do_train
```
### Eval model
```
PYTHONPATH=`pwd` python3 main.py --output_model_dir  korea-multilingual-cased  --language ko --do_eval
```

### Inference
```
PYTHONPATH=`pwd` python3 main.py --output_model_dir  korea-multilingual-cased --language ko --do_predict
```