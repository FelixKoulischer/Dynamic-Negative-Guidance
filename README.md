# Dynamic-Negative-Guidance

## Pretrained Model Weights
The pretrained model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1kTdcR3Yt_Yb8s1qcxZpRgjv_0YpsR6JS?usp=drive_link).

After downloading, place the `.pt` file in the `models/` directory.

## Description:

Images are generated from `utils/Generate_Batch.py`.
The posterior is computed iteratively using the `compute_posterior_DNG(...)` function present in `Generate_Batch.py` (Alg. 1). For numerical stability the posterior is clamped between p_min and p_max.


## Example run:
python main.py --N_tot 32 --N_batch 32 --to_remove_class 0 --guidance_type 'dynamic_negative_guidance' --guidance_scale --5. --prior 0.01 --Temp 0.2 --offset 2e-4 --p_max 0.8
