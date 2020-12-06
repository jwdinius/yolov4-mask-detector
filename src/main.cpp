/** @file main.cpp
 *  @author Joe Dinius, Ph.D
 *  @date 28 Nov. 2020
 *
 *  @brief defines main executive for mask detector application using OpenCV and Yolov4 (Darknet)
 *
 *  @see https://docs.opencv.org/3.4/d4/db9/samples_2dnn_2object_detection_8cpp-example.html#_a20
 *  @see https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
 *  @see https://github.com/AlexeyAB/darknet/blob/master/README.md#how-to-train-to-detect-your-custom-objects
 */
//! c/c++ headers
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <set>
//! system headers
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <GL/gl.h>  //! @todo start here next time https://stackoverflow.com/questions/9097756/converting-data-from-glreadpixels-to-opencvmat
//! and https://answers.opencv.org/question/21062/write-content-of-namedwindow-to-a-mat-file/
//! project headers
#include "ThreadSafeFramesQueue.hpp"
#include "utilities.hpp"

/** @typedef TrackBarAction
 * callback for trackbar slider
 */
using TrackbarAction = std::function<void(int)>;

int main(int argc, char** argv) {
  //! command-line inputs for OpenCV's parser 
  std::string keys =
      "{ help  h     | | Print help message. }"
      "{ device      | 0 | camera device number. }"
      "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
      "{ output o    | "" | Path to output video file. }"
      "{ config      | | yolo model configuration file. }"
      "{ weights     | | yolo model weights. }"
      "{ classes     | | path to a text file with names of classes to label detected objects. }"
      "{ backend     | 5 | Choose one of the following available backends: "
                           "0: DNN_BACKEND_DEFAULT, "
                           "1: DNN_BACKEND_HALIDE, "
                           "2: DNN_BACKEND_INFERENCE_ENGINE, "
                           "3: DNN_BACKEND_OPENCV, "
                           "4: DNN_BACKEND_VKCOM, "
                           "5: DNN_BACKEND_CUDA }"
      "{ target      | 6 | Choose one of the following target computation devices: "
                           "0: DNN_TARGET_CPU, "
                           "1: DNN_TARGET_OPENCL, "
                           "2: DNN_TARGET_OPENCL_FP16, "
                           "3: DNN_TARGET_MYRIAD, "
                           "4: DNN_TARGET_VULKAN, "
                           "5: DNN_TARGET_FPGA, "
                           "6: DNN_TARGET_CUDA, "
                           "7: DNN_TARGET_CUDA_FP16 }";
  
  //! allowable input extensions 
  std::set<std::string> inputExtensions = { "mp4", "png", "jpg" };

  //! kill-signal to stop running threads
  bool stopRunning = false;
  
  //! command-line arg parser
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("This application performs mask/no-mask detection from sample input using OpenCV and DarkNet-based object detectors.");
  
  if (parser.has("help")) {
      parser.printMessage();
      return EXIT_SUCCESS;
  }

  /** 
   * lambda-helper to validate input file paths
   */
  auto checkForField = [&parser](std::string field) {
    assert(parser.has(field));
    std::string filename = parser.get<std::string>(field);
    std::string out =  cv::samples::findFile(filename, false, true);
    assert(out.size() > 0);
    std::cout << "Found valid file for: " << field << std::endl;
    return out;
  };

  //! validate model weights path
  std::string const modelWeightsFile = checkForField("weights");
  //! validate model config path
  std::string const modelConfigFile = checkForField("config");
  //! validate classes definition file path
  std::string const classesFile = checkForField("classes");

  //! read class names from file
  std::vector<std::string> classes;
  {
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
      classes.emplace_back(line);
    }
  }

  //! read input height and width from config file
  //! - this is a nice feature of the Yolo config definition
  int inpHeight{0}, inpWidth{0};
  {
    std::ifstream ifs(modelConfigFile.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
      auto pos = line.find("=");
      if (pos != std::string::npos) {
        auto field = line.substr(0, pos);
        if (field.find("height") != std::string::npos) {
          inpHeight = std::stoi(line.substr(pos+1, line.size()-1));
        } else if (field.find("width") != std::string::npos) {
          inpWidth = std::stoi(line.substr(pos+1, line.size()-1));
        } else if (inpHeight > 0 && inpWidth > 0) {
          break;
        }
      }
    }
  }
  //! make sure that we actually read the height and width from the config file
  assert(inpHeight > 0 && inpWidth > 0);

  /** the user can define the computational backend and target processor for
   * computations, and the application will validate that the input is supported
   * for the detected installation */
  size_t const backend = parser.get<size_t>("backend");
  size_t const target = parser.get<size_t>("target");
  bool validSetup = false;
  auto vec = cv::dnn::getAvailableBackends();
  for (auto &v : vec) {
    if ((static_cast<size_t>(v.first) == backend)
        && (static_cast<size_t>(v.second) == target)) {
      validSetup = true;
      break;
    }
  }

  if (!validSetup) {
    std::cout << "User asked for invalid (Backend, Target) combination.  Exiting." << std::endl;
    std::cout << "The OpenCV install has the following (Backend, Target) pairs available:" << std::endl;
    for (auto &v : vec) {
      std::cout << "  (" << v.first << ", " << v.second << ")" << std::endl;
    }
    return EXIT_FAILURE;
  }

  //! setup the neural-network for inference
  auto nn = cv::dnn::readNetFromDarknet(modelConfigFile, modelWeightsFile);
  nn.setPreferableBackend(backend);
  nn.setPreferableTarget(target);
  std::vector<std::string> layerNames = nn.getUnconnectedOutLayersNames();

  //! setup output visualization
  std::string windowName("Mask Detection Example");
  cv::namedWindow(windowName, cv::WINDOW_NORMAL);
  cv::resizeWindow(windowName, 1080, 720);

  //! initialize confidence and non-maximal suppression thresholds
  double confThreshold = 0.5;
  double nmsThreshold = 0.4;
  
  /** setup lambdas with capture-by-ref to modify confThreshold and nmsThreshold via
   * OpenCV's trackbars
   * @see https://gist.github.com/acarabott/39ec44eddd8df48fd8a34aaa1481ee27 
   */
  cv::TrackbarCallback trackbarCallback = [](int pos, void* data) {
    return (*(TrackbarAction*)data)(pos);
  };
  int initConf = 100 * confThreshold;
  TrackbarAction confAction = [&](int pos) { confThreshold = pos * 0.01f; };
  cv::createTrackbar("Confidence threshold, %", windowName, &initConf, 99, trackbarCallback, (void*)&confAction);
  int initNms = 100 * nmsThreshold;
  TrackbarAction nmsAction = [&](int pos) { nmsThreshold = pos * 0.01f; };
  cv::createTrackbar("Non-max Suppression threshold, %", windowName, &initNms, 99, trackbarCallback, (void*)&nmsAction);

  //! SETUP INPUT CAPTURE
  cv::VideoCapture cap;  //! used if input video or webcam is specified
  cv::Mat inputImg;  //! used if input image is specified
  int waitTime;  //! to accommodate video playback option at real-time speed, introduce a wait time to delay frames
  if (parser.has("input")) {
    //! user has asked to run inference on input video or image file
    waitTime = 33;
    std::string inpFile = parser.get<std::string>("input");
    
    //! find file extension
    auto pos = inpFile.find_last_of(".");
    auto ext = inpFile.substr(pos+1, inpFile.size()-1);
    
    //! make sure that the extension is in the list of allowable extensions, tell the user if it is not and return
    if (inputExtensions.find(ext) == inputExtensions.end()) {
      std::cerr << "Input file cannot be processed" << std::endl;
      std::cerr << "Available extensions are: ";
      std::copy(inputExtensions.begin(), inputExtensions.end(), std::ostream_iterator<std::string>(std::cerr, ", "));
      return EXIT_FAILURE;
    }

    //! check if user passed a video
    if (ext.compare("mp4") == 0) {
      //! open input stream
      cap.open(inpFile);
      if (!cap.isOpened()) {
        std::cerr << "Unable to stream from source: " << inpFile << std::endl;
        return EXIT_FAILURE;
      }
    } else {  //! user passed an image file
      //! open input image
      inputImg = cv::imread(inpFile);
      if (inputImg.empty()) {
        std::cerr << "Unable to load image from: " << inpFile << std::endl;
        return EXIT_FAILURE;
      }
    }
  } else {  //! user indicated stream from webcam
    waitTime = 1;
    int webcamId = parser.get<int>("device");
    //! open webcam for streaming 
    cap.open(webcamId, cv::CAP_V4L2);
    if (!cap.isOpened()) {
      std::cerr << "Unable to stream from webcam; device id: " << webcamId << std::endl;
      return EXIT_FAILURE;
    }
  }

  //! SETUP OUTPUT
  cv::VideoWriter writer;
  if (parser.has("output")) {
    std::string outFile = parser.get<std::string>("output") ;
    writer.open(outFile, cv::VideoWriter::fourcc('H', '2', '6', '4'),
        30, cv::Size(1080, 720), true);
    //!@todo force .avi extension
    if (!writer.isOpened()) {
      std::cerr << "Unable to write to output file: " << outFile << std::endl;
      return EXIT_FAILURE;
    }
  }

  /** THREAD 1: Frames-capturing thread
   * UPDATES: raw frames queue
   */
  ThreadSafeFramesQueue<cv::Mat> rawFrames;
  std::thread captureThread([&](){
    while (1) {
      cv::Mat inp;
      if (!inputImg.empty()) {
        inp = inputImg.clone();
      } else {
        cap >> inp;
      }
      char key = cv::waitKey(waitTime);
      if (key == 27 || inp.empty()) {
        //! If ESC is pressed OR stream is exhausted, Send the kill-signal to other threads
        stopRunning = true;
        break;
      }
      /** @note need to use clone here to pass a copy, otherwise input could be corrupted */
      rawFrames.push(inp.clone());  
      }
  });

  /** THREAD 2: Inference thread
   * UPDATES: inference frames queue - pair of raw image and inferred bounding-boxes
   */
  ThreadSafeFramesQueue<std::pair<cv::Mat, std::vector<cv::Mat>>> inferenceQueue;
  std::thread inferenceThread([&](){
    while (!stopRunning || !rawFrames.empty()) {
      cv::Mat inp;
      if (rawFrames.Pop(inp)) {
        //! inp is valid, run inference on it
        PreProcess(inp, nn, cv::Size(inpWidth, inpHeight));
        std::vector<cv::Mat> boundingBoxes;
        nn.forward(boundingBoxes, GetOutputsNames(nn));
        inferenceQueue.Push(std::make_pair(inp.clone(), boundingBoxes));
      }
    }
  });
  
  /** THREAD 3: Post-processing thread
   * UPDATES: processed frames queue - raw frame with appended annotated bounding boxes with confidence and
   * nms-suppression applied
   */
  ThreadSafeFramesQueue<cv::Mat> processedFrames;
  std::thread postProcThread([&](){
    while (!stopRunning || !inferenceQueue.empty()) {
      std::pair<cv::Mat, std::vector<cv::Mat>> inp;
      if (inferenceQueue.Pop(inp)) {
        //! inp is valid, post-process it
        PostProcess(inp.first, inp.second, classes, confThreshold, nmsThreshold);
        processedFrames.Push(inp.first.clone());
      }
    }
  });

  /** MAIN THREAD: handle plotting and video writing
   * UPDATES: plot window and output video
   */
  while(!stopRunning || !processedFrames.empty()) {
    cv::Mat imOut;
    if (processedFrames.Pop(imOut)) {
      int printPos = 15;
      int deltaHeight = 15;
      {
        //! Add raw frame streaming telemetry to drawing frame
        std::string label = cv::format("Raw Frames Streaming@ %.2f FPS", rawFrames.GetFPS());
        cv::putText(imOut, label, cv::Point(0, printPos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
      }
      printPos += deltaHeight;
      {
        //! Add inference timing telemetry to drawing frame
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = nn.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("YOLOv4 Inference Time: %.2f ms", t);
        cv::putText(imOut, label, cv::Point(0, printPos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
      }
      printPos += deltaHeight;
      {
        //! Add post-processing timing telemetry to drawing frame
        std::string label = cv::format("Post-processed Frames Streaming@ %.2f FPS", processedFrames.GetFPS());
        cv::putText(imOut, label, cv::Point(0, printPos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
      }
      if (writer.isOpened()) {
        cv::Mat imOutResize;
        cv::resize(imOut, imOutResize, cv::Size(1080, 720));
        writer << imOutResize;
      }
      cv::imshow(windowName, imOut);
    }
  }

  //! cleanup after all thread callbacks have exited
  captureThread.join();
  inferenceThread.join();
  postProcThread.join();
  cap.release();
  writer.release();
  cv::destroyAllWindows();

  //! exit cleanly
  return EXIT_SUCCESS;
}
