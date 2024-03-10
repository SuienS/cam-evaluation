import threading
import torch
from torch import nn
import matplotlib.pyplot as plt
import glob
import logging
import argparse
import csv
import time
import yaml



from model.resnet_imagenet import ResNet50V2ImageNet
from utils.process import Process

CONFIG_PATH = "./test_config.yaml"


# Model
model = None

# Model with softmax
model_with_softmax = None

def main():
    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
    logging.info("Initiating tests...")

    logging.info("Loading configs...")

    with open(CONFIG_PATH) as stream:
        try:
            test_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error("Error loading config file.")
            logging.error(exc)

    GPU_DEVICE = test_configs["gpu_device"]
    CLASS_COUNT = test_configs["class_count"]
    IMAGE_SIZE = test_configs["image_size"]
    SAMPLE_COUNT = test_configs["sample_count"]
    DATA_PATH = test_configs["data_path"]
    OUTPUT_PATH = test_configs["output_path"]

    logging.info("Configs loaded.")

    if torch.cuda.is_available():
        device = torch.device(GPU_DEVICE)
        logging.info("Using GPU")

    else:
        device = torch.device("cpu")
        logging.warning("No GPU detected! Using CPU.")
    
    model = ResNet50V2ImageNet().to(device)
    model_with_softmax = ResNet50V2ImageNet(activation=nn.Softmax(-1)).to(device)

    # Model config
    # Set model to eval mode
    model.eval()
    model_with_softmax.eval()

    logging.info("Models initiated.")

    # Model config
    last_conv_layer = model.model.layer4[2].conv3
    class_count = CLASS_COUNT
    class_list = model.weights.meta["categories"]
    logging.info("Categories Count: " + str(len(class_list)))
    
    img_h = IMAGE_SIZE
    img_w = IMAGE_SIZE
    img_pp_size = (img_w, img_h)
    target_classes = list(range(class_count))

    DATA_PATH = "./example_data/MSCOCO_Sample"
    valid_img_paths = glob.glob(DATA_PATH + "/*.jpg")

    logging.info("Data path: " + DATA_PATH)
    logging.info("Total Data Count: " + str(len(valid_img_paths)))

    sample_count = SAMPLE_COUNT
    if sample_count == -1:
        logging.info("Sample count set to -1. Using all data.")
    elif sample_count > len(valid_img_paths):
        logging.warning("Sample count exceeds data count. Using all data.")
    else:
        logging.info("Sample count set to "+ str(sample_count))
        valid_img_paths = valid_img_paths[::len(valid_img_paths)//sample_count]

    processor = Process(
        model, model_with_softmax, device, model.weights.transforms(), 
        last_conv_layer, image_width=img_w, image_height=img_h
    )

    logging.info("Processor initiated.")

    logging.info("Test 1/2 - Testing FMGCAM...")
    iauc_means_ms_fmgcam, ic_means_ms_fmgcam, dauc_means_ms_fmgcam, dc_means_ms_fmgcam = processor.get_scores_mstep(
        valid_img_paths, class_count, img_pp_size, target_classes, "fmgcam", 
        enhance=True, act_mode="relu"
    )

    fmgcam_results = [{
        "iauc_means_ms_fmgcam": iauc_means_ms_fmgcam[index],
        "ic_means_ms_fmgcam": ic_means_ms_fmgcam[index],
        "dauc_means_ms_fmgcam": dauc_means_ms_fmgcam[index],
        "dc_means_ms_fmgcam": dc_means_ms_fmgcam[index]
    } for index in range(len(iauc_means_ms_fmgcam))]

    time_epoch = int(time.time())
    fmgcam_results_csv_name = "fmgcam_results_" + str(time_epoch) + ".csv"

    with open(OUTPUT_PATH + fmgcam_results_csv_name, 'w') as f:
        w = csv.DictWriter(f, fmgcam_results[0].keys())
        w.writeheader()
        [w.writerow(fmgcam_result) for fmgcam_result in fmgcam_results]
    
    logging.info("FMGCAM results saved to: " + fmgcam_results_csv_name)


    logging.info("Test 2/2 - Testing GCAM...")
    iauc_means_ms_gcam, ic_means_ms_gcam, dauc_means_ms_gcam, dc_means_ms_gcam = processor.get_scores_mstep(
        valid_img_paths, class_count, img_pp_size, target_classes, "gcam", 
        enhance=False, act_mode="relu"
    )

    gcam_results = [{
        "iauc_means_ms_gcam": iauc_means_ms_gcam[index],
        "ic_means_ms_gcam": ic_means_ms_gcam[index],
        "dauc_means_ms_gcam": dauc_means_ms_gcam[index],
        "dc_means_ms_gcam": dc_means_ms_gcam[index]
    } for index in range(len(iauc_means_ms_gcam))]

    time_epoch = int(time.time())
    gcam_results_csv_name = "gcam_results_" + str(time_epoch) + ".csv"

    with open(OUTPUT_PATH + gcam_results_csv_name, 'w') as f:
        w = csv.DictWriter(f, gcam_results[0].keys())
        w.writeheader()
        [w.writerow(gcam_result) for gcam_result in gcam_results]
    
    logging.info("GCAM results saved to: " + gcam_results_csv_name)

    logging.info("Tests completed!")


if __name__ == "__main__":
    # Increase the stack size to prevent stack overflow
    threading.stack_size(9000000)
    testing_thread = threading.Thread(
        target=main,
    )
    testing_thread.daemon = True
    testing_thread.start()

    # Keep the main thread alive
    import time
    while True:
        time.sleep(1)
