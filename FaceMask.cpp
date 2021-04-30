#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <cmath>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <unistd.h>
#define I2C_DEV_PATH "/dev/i2c-1"

/* Just in case */
#ifndef I2C_SMBUS_READ
#define I2C_SMBUS_READ 1
#endif
#ifndef I2C_SMBUS_WRITE
#define I2C_SMBUS_WRITE 0
#endif
typedef union i2c_smbus_data i2c_data;
using namespace cv;
using namespace std;

int model_width;
int model_height;
int model_channels;
//char mask='n';
//int n=0;

std::unique_ptr<tflite::Interpreter> interpreter;

double getTemperature(){
    int fdev = open(I2C_DEV_PATH, O_RDWR); // open i2c bus

    if (fdev < 0) {
        fprintf(stderr, "Failed to open I2C interface %s Error: %s\n", I2C_DEV_PATH, strerror(errno));
        return -1;
    }

    unsigned char i2c_addr = 0x5A;

    // set slave device address, default MLX is 0x5A
    if (ioctl(fdev, I2C_SLAVE, i2c_addr) < 0) {
        fprintf(stderr, "Failed to select I2C slave device! Error: %s\n", strerror(errno));
        return -1;
    }

    // enable checksums control
    if (ioctl(fdev, I2C_PEC, 1) < 0) {
        fprintf(stderr, "Failed to enable SMBus packet error checking, error: %s\n", strerror(errno));
        return -1;
    }


    // trying to read something from the device unsing SMBus READ request
    i2c_data data;
    char command = 0x06; // command 0x06 is reading thermopile sensor, see datasheet for all commands

    // build request structure
    struct i2c_smbus_ioctl_data sdat = {
        .read_write = I2C_SMBUS_READ,
        .command = command,
        .size = I2C_SMBUS_WORD_DATA,
        .data = &data
    };

    // do actual request
    if (ioctl(fdev, I2C_SMBUS, &sdat) < 0) {
        fprintf(stderr, "Failed to perform I2C_SMBUS transaction, error: %s\n", strerror(errno));
        return -1;
    }

    // calculate temperature in Celsius by formula from datasheet
    double temp = (double) data.word;
    temp = (temp * 0.02)-0.01;
    temp = temp - 273.15;
    if(close(fdev),0)printf("cannot close\n");
    return temp;
}

//-----------------------------------------------------------------------------------------------------------------------
void GetImageTFLite(float* out, Mat &src)
{
    int i,Len;
    float f;
    uint8_t *in;
    static Mat image;

    // copy image to input as input tensor
    cv::resize(src, image, Size(model_width,model_height),INTER_NEAREST);

    //model posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite runs from -1.0 ... +1.0
    //model multi_person_mobilenet_v1_075_float.tflite                 runs from  0.0 ... +1.0
    in=image.data;
    Len=image.rows*image.cols*image.channels();
    for(i=0;i<Len;i++){
        f     =in[i];
        out[i]=(f - 127.5f) / 127.5f;
    }
}
//-----------------------------------------------------------------------------------------------------------------------
void detect_from_video(Mat &src)
{
    Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    // copy image to input as input tensor
    GetImageTFLite(interpreter->typed_tensor<float>(interpreter->inputs()[0]), src);

    interpreter->AllocateTensors();
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      //quad core

    interpreter->Invoke();      // run your model

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    //there are ALWAYS 10 detections no matter how many objects are detectable
    //cout << "number of detections: " << num_detections << "\n";

    const float confidence_threshold = 0.5;
    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){
            int  det_index = (int)detection_classes[i];
            float y1=detection_locations[4*i  ]*cam_height;
            float x1=detection_locations[4*i+1]*cam_width;
            float y2=detection_locations[4*i+2]*cam_height;
            float x2=detection_locations[4*i+3]*cam_width;

            Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            if(det_index==0){
                double temp=getTemperature();
                string combo= "Mask : ";
                char x[__SIZEOF_FLOAT__];
                sprintf(x,"%g",temp);
                combo.append(x);
                combo.append(" F");
                rectangle(src,rec, Scalar(0, 255, 0), 2, 8, 0);
                putText(src,combo, Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.7, Scalar(0, 255, 0), 1, 8, 0);
           //     n++;
                //if(n==10){}

            }
            if(det_index==1){
                rectangle(src,rec, Scalar(0, 0, 255), 2, 8, 0);
                putText(src,"No mask. Please wear a mask", Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.7, Scalar(0, 0, 255), 1, 8, 0);
             //   n=0;
            }
            if(det_index==2){
                rectangle(src,rec, Scalar(0, 127, 255), 2, 8, 0);
                putText(src,"Wear incorrect. Wear it correctly", Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.7, Scalar(0, 127, 255), 1, 8, 0);
               // n=0;
            }
        }
    }
}
//-----------------------------------------------------------------------------------------------------------------------
int main(int argc,char ** argv)
{
    float f;
    float FPS[16];
    int i;
    int Fcnt=0;
    Mat frame;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("ssd_mobilenet_v2_fpnlite.tflite");
//    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("ssdlite_mobilenet_v2.tflite");
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();

    // Get input dimension from the input tensor metadata
    // Assuming one input only
    int In = interpreter->inputs()[0];
    model_height   = interpreter->tensor(In)->dims->data[1];
    model_width    = interpreter->tensor(In)->dims->data[2];
    model_channels = interpreter->tensor(In)->dims->data[3];
    cout << "height   : "<< model_height << endl;
    cout << "width    : "<< model_width << endl;
    cout << "channels : "<< model_channels << endl;

    cout << "tensors size: " << interpreter->tensors_size() << "\n";
    cout << "nodes size: " << interpreter->nodes_size() << "\n";
    cout << "inputs: " << interpreter->inputs().size() << "\n";
    cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
    cout << "outputs: " << interpreter->outputs().size() << "\n";

    VideoCapture cap(0);//Norton_M3.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;

    while(1){
//        frame=imread("Kapje_2.jpg");  //need to refresh frame before dnn class detection

        cap >> frame;
        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }

        detect_from_video(frame);

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();

        Tbegin = chrono::steady_clock::now();

        FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(frame, format("FPS %0.2f",f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));
        //show output
        imshow("RPi 4 - 2.0 GHz - 2 Mb RAM", frame);
        char esc=waitKey(5);
        if(esc == 27) break;

    }

    cout << "Closing the camera" << endl;

    // When everything done, release the video capture and write object
    cap.release();

    destroyAllWindows();
    cout << "Bye!" << endl;

    return 0;
}
//-----------------------------------------------------------------------------------------------------------------------

