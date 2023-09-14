## <div align="center">PyTorch Training Pipeline</div> 

<details open>

<summary>Install</summary><br>  
conda create --name torch_pipe pip python==3.10.6
pip install -r requirements.txt

</details>

<details open>
<summary>Work checklist</summary><br>  

- [X] Dataloader
    - [X] CSV
    - [ ] Bounding boxes

- [X] Albumentations

- [X] Optimizer
	- [X] Adam
	- [X] SGD
	- [X] RMSprop

- [X] lr scheduler
	- [X] ReduceLROnPlateau
	- [X] CosineAnnealingLR
	- [X] OneCycleLR

- [X] Balance weights

- [X] Early stopping

- [X] Criterion
	- [X] CrossEntropyLoss
	- [X] BCEWithLogitsLoss

- [X] Architectures
	- [X] resnet ::
	- [X] efficientnet ::

- [X] Torchmetrics
	- [X] Confusion Matrix
		- [X] Accuracy
		- [X] Precision
		- [X] Recall
		- [X] F1 score
	- [X] Average precision
	- [X] Precision@K

- [X] Other metrics
	- [X] Precision top k

- [X] WandB
    - [X] Error analysis table

- [X] Train
    - [X] Offline
	- [X] WandB integration
	- [X] Save model
	- [X] Choose metric for lr scheduler, 
		  model performance tracking and early stopping

- [X] Eval
    - [X] Offline
	- [X] WandB integration

- [X] Resume training
	- [X] Offline
	- [X] WandB integration

- [X] Job type
    - [X] Multiclass
    - [X] Multilabel

- [ ] Logging
	- [X] Offline
	- [X] WandB integration
		- [X] Metrics curves
		- [X] Plot confusion matrix
		- [ ] Plot metrics bar

- [X] Device agnostic

- [ ] Reproducibility

- [ ] Alerts

- [ ] Usage Documentation

</details>

<details open>
<summary>Usage Example</summary><br>  
</details>

```bash
python train.py --logging 20 --hyps hyps_none --arch architecture --data data --mode offline --device cuda --epochs 2 --batch_size 2 --patience 5 --save_weights False
```