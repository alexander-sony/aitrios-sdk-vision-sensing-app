#import json
import subprocess
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

HOME = 'work'
IMAGES_TRAINING_DIR = HOME + "/dataset/images/training"
IMAGES_VALIDATION_DIR = HOME + "/dataset/images/validation"
IMAGES_ORG_DIR = HOME + "/dataset/images_org"
MODELS_DIR = HOME + "/models"

TRAINING_DOCKER_IMAGE_NAME = "tf1_od_api_env:1.0.0"
TRAINING_DOCKER_VOLUME_DIR = "/root/samples/zone_detection"

INPUT_SIZE = 300
DNN_OUTPUT_DETECTIONS = 10

def run_shell_command(command: str):
    """Run shell command with output log and checking return code."""

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True,
        stderr=subprocess.STDOUT,
        bufsize=1,
        close_fds=True,
    ) as process:
        # for line in iter(process.stdout.readline, b""):
        while True:
            line = process.stdout.readline()
            if line:
                print(line.rstrip().decode("utf-8"))
            if process.poll() is not None:
                break
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)

def convert_to_tflite(output_file: str, graph_def_file: str):
    """Convert SavedModel to TFLite within training docker container."""

    output_arrays = (
        "TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,"
        "TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3"
    )

    cmd = f'docker run --rm -t -v $(pwd):{TRAINING_DOCKER_VOLUME_DIR} {TRAINING_DOCKER_IMAGE_NAME} \
            tflite_convert \
        --output_file={output_file} \
        --graph_def_file={graph_def_file} \
        --inference_type=QUANTIZED_UINT8 \
        --input_arrays="normalized_input_image_tensor" \
        --output_arrays={output_arrays} \
        --mean_values=128 \
        --std_dev_values=128 \
        --input_shapes=1,{INPUT_SIZE},{INPUT_SIZE},3 \
        --change_concat_input_ranges=false \
        --allow_nudging_weights_to_use_fast_gemm_kernel=true \
        --allow_custom_ops'
    '''
    docker run --rm -t -v $(pwd):/root/samples/zone_detection tf1_od_api_env:1.0.0 \
        tflite_convert \
        --output_file=/root/samples/zone_detection//home/l1000323954/github/aitrios-sdk-vision-sensing-app/samples/zone_detection/models/base_model_quantized_od.tflite \
        --graph_def_file=/root/samples/zone_detection//home/l1000323954/github/aitrios-sdk-vision-sensing-app/samples/zone_detection/models/out/ckpt/tflite_graph.pb \
        --inference_type=QUANTIZED_UINT8 \
        --input_arrays="normalized_input_image_tensor" \
        --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
        --mean_values=128 --std_dev_values=128 --input_shapes=1,300,300,3 \
        --change_concat_input_ranges=false \
        --allow_nudging_weights_to_use_fast_gemm_kernel=true \
        --allow_custom_ops
    '''
    print(cmd)
    run_shell_command(cmd)

class TFLiteInterpreter:
    """A class to evaluate TFLite."""

    def __init__(self, model_file: str):
        self.interpreter = TFLiteInterpreter._make_interpreter(model_file)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def _make_interpreter(model_file: str) -> Any:
        model_file, *device = model_file.split("@")
        interpreter = tf.lite.Interpreter(model_path=model_file)
        interpreter.allocate_tensors()
        return interpreter

    def _prepare_input(self, input_batch: np.ndarray) -> np.ndarray:
        if self.input_details[0]["dtype"] in [np.uint8, np.int8]:
            input_scale, input_zero_point = self.input_details[0]["quantization"]
            input_batch = input_batch / input_scale + input_zero_point
        return tf.cast(
            np.expand_dims(input_batch, axis=0), self.input_details[0]["dtype"]
        )

    @staticmethod
    def _get_output(interpreter, output_details, index):
        output = interpreter.get_tensor(output_details[index]["index"])
        if output_details[index]["dtype"] in [np.uint8, np.int8]:
            scale, zero_point = output_details[index]["quantization"]
            output = scale * (output - zero_point)
        return tf.cast(output, output_details[index]["dtype"])

    def _get_outputs(self) -> List[np.ndarray]:
        return [
            self._get_output(self.interpreter, self.output_details, index)
            for index in range(len(self.output_details))
        ]

    def run(self, input_batch: Any) -> None:
        self.interpreter.set_tensor(
            self.input_details[0]["index"], self._prepare_input(input_batch)
        )
        self.interpreter.invoke()
        return self._get_outputs()

def evaluate_and_display_tflite(model_path: str, image: cv2.Mat, score_threshold: float, dnn_output_bboxes: int) -> Tuple[list, list, cv2.Mat]:
    """Evaluate TFLite, convert Output Tensor format from TFLite to Edge AI Device and display
    the image overlaid with Output Tensor."""

    interp = TFLiteInterpreter(model_path)

    raw_output_tensors = interp.run(image.astype(np.float32) / 255.0)
    output_tensor = np.concatenate([np.array(x).flatten() for x in raw_output_tensors])

    plt.figure()

    num_of_detection = int(output_tensor[len(output_tensor) - 1])

    postprocessed = []
    for index in range(num_of_detection):
        inference = []
        inference.append(output_tensor[4 * dnn_output_bboxes + index])  # cls
        inference.append(output_tensor[5 * dnn_output_bboxes + index])  # score
        inference.append(output_tensor[0 + 4 * index])  # ymin
        inference.append(output_tensor[1 + 4 * index])  # xmin
        inference.append(output_tensor[2 + 4 * index])  # ymax
        inference.append(output_tensor[3 + 4 * index])  # xmax
        postprocessed.append(inference)

    # float array of Output Tensor on Edge AI Device
    output_tensor_device = [0.0 for _ in range(dnn_output_bboxes * 6 + 1)]
    output_tensor_device[dnn_output_bboxes * 6] = len(postprocessed)  # num of detection

    overlaid_text_image = np.zeros((INPUT_SIZE, INPUT_SIZE, 4), dtype=np.uint8)
    result_image = image.copy()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2RGBA)
    for idx, x in enumerate(postprocessed):
        cls, score, ymin, xmin, ymax, xmax = x
        print(f"score: {score}, class id: {int(cls)}")

        output_tensor_device[dnn_output_bboxes * 0 + idx] = float(ymin)
        output_tensor_device[dnn_output_bboxes * 1 + idx] = float(xmin)
        output_tensor_device[dnn_output_bboxes * 2 + idx] = float(ymax)
        output_tensor_device[dnn_output_bboxes * 3 + idx] = float(xmax)
        output_tensor_device[dnn_output_bboxes * 4 + idx] = float(cls)
        output_tensor_device[dnn_output_bboxes * 5 + idx] = float(score)

        ymin = int(ymin * INPUT_SIZE)
        xmin = int(xmin * INPUT_SIZE)
        ymax = int(ymax * INPUT_SIZE)
        xmax = int(xmax * INPUT_SIZE)
        if score > score_threshold:
            cv2.rectangle(
                result_image, (xmin, ymin), (xmax, ymax), (255, 255, 0, 255), 2
            )
            text_pos_x = xmin + 2
            text_pos_y = ymax - 2
            if text_pos_x > (INPUT_SIZE - 35):
                text_pos_x = INPUT_SIZE - 35
            elif text_pos_x < 2:
                text_pos_x = 2
            if text_pos_y < 14:
                text_pos_y = 14
            elif text_pos_y > (INPUT_SIZE - 2):
                text_pos_y = INPUT_SIZE - 2
            cv2.putText(
                overlaid_text_image,
                str("{0:.0f}%".format(score * 100.0)),
                (text_pos_x, text_pos_y),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.add(result_image, overlaid_text_image, result_image)
    plt.imshow(result_image)
    plt.axis("off")
    plt.show()
    plt.clf()
    plt.close()

    return output_tensor_device, postprocessed, result_image

def tflite_convert():
    OUTPUT_FILE = (
        f"{TRAINING_DOCKER_VOLUME_DIR}/{MODELS_DIR}/base_model_quantized_od.tflite"
    )
    GRAPH_DEF_FILE = (
        f"{TRAINING_DOCKER_VOLUME_DIR}/{MODELS_DIR}/out/ckpt/tflite_graph.pb"
    )
    convert_to_tflite(OUTPUT_FILE, GRAPH_DEF_FILE)
    print(
        f"Base AI model TFLite converted in ./{MODELS_DIR}/base_model_quantized_od.tflite"
    )
    convert_to_tflite(OUTPUT_FILE, GRAPH_DEF_FILE)

def evaluate_tflite():
    IMAGE_INPUT = f"{IMAGES_VALIDATION_DIR}/training_image_037.jpg"
    image = cv2.cvtColor(cv2.imread(IMAGE_INPUT), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(INPUT_SIZE, INPUT_SIZE))    
    MODEL = f"{MODELS_DIR}/base_model_quantized_od.tflite"
    SCORE_THRESHOLD = 0.3
    output_tensor_device, postprocessed, result_image = evaluate_and_display_tflite(
        MODEL, image, SCORE_THRESHOLD, DNN_OUTPUT_DETECTIONS
    )

if __name__ == '__main__':
    #tflite_convert()
    evaluate_tflite()