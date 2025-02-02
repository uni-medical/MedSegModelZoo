import os, sys, pdb
import argparse
import torch
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, isdir, save_json, load_pickle, subfiles
from time import time
from utils import get_weights_dir
from utils import download_file
from predict import predict_cases


def preprocess_input(input_path, output_path):
    input_folder = 'temp_inputs'
    output_folder = 'temp_outputs'
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    if len(input_path) == 1:
        input_name = input_path[0].split('/')[-1]
        input_name = input_name.split('.')[0]
        input_name = input_name + '_0000.nii.gz'
        nii_path = os.path.join(input_folder, input_name)
        shutil.copy(input_path[0], nii_path)
        list_of_lists = [[nii_path]]
    else:
        list_of_lists = []
        for i, i_path in enumerate(input_path):
            input_name = i_path.split('/')[-1]
            input_name = input_name.split('.')[0]
            input_name = input_name + '_' + '%04d' % i + '.nii.gz'
            nii_path = os.path.join(input_folder, input_name)
            shutil.copy(i_path, nii_path)
            list_of_lists.append(nii_path)
        list_of_lists = [list_of_lists]
    output_files = [os.path.join(output_folder, output_path.split('/')[-1])]
    return list_of_lists, output_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_path', required=True, nargs='+', type=str)
    parser.add_argument('-o', "--output_path", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', required=True, help='task name or task ID, required.')

    parser.add_argument('-tr', '--trainer_class_name', required=False, default="nnUNetTrainerV2")

    parser.add_argument('-p', '--plans_identifier', default="nnUNetPlansv2.1", required=False, help='do not touch this unless you know what you are doing')

    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax "
                             "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                             "merged between output_folders with nnUNet_ensemble_predictions")

    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")

    parser.add_argument("--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    save_npz = args.save_npz
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    step_size = args.step_size
    mode = args.mode
    all_in_gpu = args.all_in_gpu
    trainer = args.trainer_class_name
    task_name = args.task_name
    folds = 0

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    assert task_name.startswith("Task"), "task_name must start with Task"
    weights_dir = get_weights_dir()
    model_folder_name = join(weights_dir, task_name, trainer + "__" + args.plans_identifier)
    if not isdir(model_folder_name):
        download_file(f"https://github.com/uni-medical/MedSegModelZoo/releases/download/test_0.0.1/{task_name}.zip", os.path.join(str(weights_dir), f"{task_name}.zip"))
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    list_of_lists, output_files = preprocess_input(input_path, output_path)

    expected_num_modalities = load_pickle(join(model_folder_name, "plans.pkl"))['num_modalities']
    assert len(input_path) == expected_num_modalities, "Number of input modalities does not match expected number of modalities"

    st = time()
    predict_cases(model_folder_name, list_of_lists, output_files, folds, save_npz, num_threads_preprocessing, 
                  num_threads_nifti_save, None, not disable_tta,
                  mixed_precision=not args.disable_mixed_precision, all_in_gpu=all_in_gpu,
                  step_size=step_size, disable_postprocessing=False)
    for i in range(len(output_files)):
        shutil.move(output_files[i], output_path)
    shutil.rmtree('temp_inputs')
    shutil.rmtree('temp_outputs')
    end = time()
    print (f'Prediction time: {end - st}')


if __name__ == "__main__":
    main()
    # -i: 只接收一个输入，必须是nii.gz
    # 数据格式：nii.gz，# TODO: nii，dcm
    # 先挑几个task，验证后上线（挑哪几个task：MSD、AMOS、autoPET？）
    # 支持纯CPU
    # 上传模型:huggingface
