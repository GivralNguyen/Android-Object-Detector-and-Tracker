/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.detector;

import android.app.Application;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.os.SystemClock;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE.NeuralNetworkBuilder;
import com.qualcomm.qti.snpe.SnpeError;
import com.qualcomm.qti.snpe.TF8UserBufferTensor;
import com.qualcomm.qti.snpe.Tensor;
import com.qualcomm.qti.snpe.UserBufferTensor;


import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class RetinaDetector {
    static final String LOGTAG = RetinaDetector.class.getSimpleName();

    public boolean isUsingQuantized = false;
    static final double CONFIDENCE_THRESHOLD = 0.8;//0.8;//0.6;
    static final double IOU_THRESHOLD = 0.4;
    public static final float TRUE_FACE_THRESHOLD = 0.75f;
    public static final int IMG_WIDTH = 850;//568;// //512; //850;//640;
    public static final int IMG_HEIGHT = 480;//320;// //288; // 480;//360;
    public static final List<Anchor> anchors = new ArrayList<Anchor>();
    NeuralNetwork network = null;

    // Prepare input buffer
    String mInputLayer = "";
    Set<String> mOutputLayer;

    private FloatTensor inputTensor = null;
    private Map<String, FloatTensor> inputs = new HashMap<>();
    private Map<String, TF8UserBufferTensor> inputTensors = new HashMap<>();
    private Map<String, TF8UserBufferTensor> outputTensors = new HashMap<>();
    private Map<String, ByteBuffer> inputBuffers = new HashMap<>();
    private Map<String, ByteBuffer> outputBuffers = new HashMap<>();

    private float[] inputValues = new float[IMG_WIDTH * IMG_HEIGHT * 3];

    //private  HexagonDelegate hexagonDelegate;

    //True Face filter
    //private TrueFaceDetector mTrueFaceDetector;
    public Mat mFrameCv = new Mat();

    public RetinaDetector(
            Context context,
            Application application,
            int modelRes
//        int trueFaceModelRes
    ) {
        //For hexagon delegate
        // Create the Delegate instance.
        try {
            //Log.d(LOGTAG, "Hexagon: Native libraryDir is: " + context.getApplicationInfo().nativeLibraryDir);
            //hexagonDelegate = new HexagonDelegate(context);
            //options.addDelegate(hexagonDelegate);
            //Log.d(LOGTAG,"Accelerate by Hexagon");
        } catch (UnsupportedOperationException e) {
            // Hexagon delegate is not supported on this device.
            Log.d(LOGTAG,"Hexagon delegate is not supported on this device.");
        }

        final Resources res = context.getResources();
        final InputStream modelInputStream = res.openRawResource(modelRes);
        try {
            final NeuralNetworkBuilder builder = new NeuralNetworkBuilder(application)
                    .setDebugEnabled(false)
                    .setRuntimeOrder(
                            NeuralNetwork.Runtime.GPU_FLOAT16,
                            //NeuralNetwork.Runtime.DSP,
                            NeuralNetwork.Runtime.GPU,
                            NeuralNetwork.Runtime.CPU
                    )
                    .setModel(modelInputStream, modelInputStream.available())
                    .setOutputLayers("concatenation_3",
                            "concatenation_4",
                            "concatenation_5")
                    .setCpuFallbackEnabled(true)
                    .setUseUserSuppliedBuffers(isUsingQuantized)
                    .setPerformanceProfile(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE);
            network = builder.build();

            // Prepare inputs buffer
            mInputLayer = network.getInputTensorsNames().iterator().next();
            mOutputLayer = network.getOutputTensorsNames();
            inputTensor = network.createFloatTensor(network.getInputTensorsShapes().get(mInputLayer));

            createAnchor();

            //Init True Face detector
            /*
            mTrueFaceDetector = new TrueFaceDetector(
                    context,
                    application,
                    trueFaceModelRes
            );
             */

            Log.d(LOGTAG, "RetinaDetector inited " + network.getInputTensorsShapes().entrySet().iterator().next().getValue().length + " anchor " + anchors.size());
        } catch (IOException e) {
            // Do something here
        }
    }

    private static native void nativeAlign(long frameCvPtr, long frameRetPtr, float[][] landmarkObjArr);

    public Mat prepareAlignNative(Mat frameCv, Bbox box)
    {
        float scaleX = RetinaDetector.IMG_WIDTH / (float) frameCv.cols();//frame.getWidth();
        float scaleY = RetinaDetector.IMG_HEIGHT / (float) frameCv.rows();//frame.getHeight();

        //float landmarks[] = {
        float landmarks[][] = {
                {box.landmarks[0].x / scaleX, box.landmarks[0].y / scaleY},
                {box.landmarks[1].x / scaleX, box.landmarks[1].y / scaleY},
                {box.landmarks[2].x / scaleX, box.landmarks[2].y / scaleY},
                {box.landmarks[3].x / scaleX, box.landmarks[3].y / scaleY},
                {box.landmarks[4].x / scaleX, box.landmarks[4].y / scaleY}
        };

        Mat faceAligned = new Mat();//112, 112, CV_32FC1);
        nativeAlign(frameCv.getNativeObjAddr(), faceAligned.getNativeObjAddr(), landmarks);

        return faceAligned;
    }

    private void createAnchor() {
        final float[][] featureMap = new float[3][3];
        final float[][] minSizes = {{10, 20}, {32, 64}, {128, 256}};
        final float[] steps = {8, 16, 32};
        for (int i = 0; i < 3; ++i) {
            featureMap[i][0] = (float) Math.ceil(IMG_HEIGHT / steps[i]);
            featureMap[i][1] = (float) Math.ceil(IMG_WIDTH / steps[i]);
        }
        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < featureMap[k][0]; ++i) {
                for (int j = 0; j < featureMap[k][1]; ++j) {
                    for (int l = 0; l < 2; ++l) {//2//minSizes.size
                        final float s_ky = minSizes[k][l] / IMG_HEIGHT;
                        final float s_kx = minSizes[k][l] / IMG_WIDTH;
                        final float cx = (float) (j + 0.5) * steps[k] / IMG_WIDTH;
                        final float cy = (float) (i + 0.5) * steps[k] / IMG_HEIGHT;
                        final Anchor anchor = new Anchor(cx, cy, s_kx, s_ky);
                        anchors.add(anchor);
                    }
                }
            }
        }
    }

    private void prepareInputs(Bitmap frame) {
        loadRgbBitmapAsFloat(frame);
        inputTensor.write(inputValues, 0, inputValues.length);
        inputs.put(mInputLayer, inputTensor);
    }

    private void loadRgbBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;
                final int batchIdx = idx * 3;

                final float[] rgb = extractColorChannels(pixels[idx]);
                inputValues[batchIdx]     = rgb[0];
                inputValues[batchIdx + 1] = rgb[1];
                inputValues[batchIdx + 2] = rgb[2];
            }
        }
    }

    private float[] extractColorChannels(int pixel) {
        float b = ((pixel)       & 0xFF);
        float g = ((pixel >>  8) & 0xFF);
        float r = ((pixel >> 16) & 0xFF);

        return new float[]{
                r - 123,
                g - 117,
                b - 104
        };
    }

    private List<Bbox> convertOutputs(Map<String, FloatTensor> outputs) {
        float[] locs = {};
        float[] landmarks = {};
        float[] confidences = {};
        for (Map.Entry<String, FloatTensor> output : outputs.entrySet()) {
            FloatTensor outputTensor = output.getValue();
            switch (output.getKey()) {
                case "loc0":
                    locs = new float[outputTensor.getSize()];
                    outputTensor.read(locs, 0, locs.length);
                    break;
                case "landmark0":
                    landmarks = new float[outputTensor.getSize()];
                    outputTensor.read(landmarks, 0, landmarks.length);
                    break;
                case "conf0":
                    confidences = new float[outputTensor.getSize()];
                    outputTensor.read(confidences, 0, confidences.length);
                    break;
            }
        }
        List<Bbox> bboxes = buildBbox(locs, confidences, landmarks);
        bboxes = nms(bboxes);
        return bboxes;
    }

    private List<Bbox> convertTf8Outputs() {
        float[] locs = {};
        float[] landmarks = {};
        float[] confidences = {};
        Iterator<String> outputLayers = mOutputLayer.iterator();
        while (outputLayers.hasNext()) {
            String outputLayer = outputLayers.next();
            switch (outputLayer) {
                case "loc0": {
                    locs = TensorUtils.dequantize(outputTensors.get(outputLayer), outputBuffers.get(outputLayer));
                    break;
                }
                case "landmark0": {
                    landmarks = TensorUtils.dequantize(outputTensors.get(outputLayer), outputBuffers.get(outputLayer));
                    break;
                }
                case "conf0": {
                    confidences = TensorUtils.dequantize(outputTensors.get(outputLayer), outputBuffers.get(outputLayer));
                    break;
                }
            }
        }
        List<Bbox> bboxes = buildBbox(locs, confidences, landmarks);
        bboxes = nms(bboxes);
        return bboxes;
    }

    public List<Bbox> detectFrame(Bitmap frame) {
        //long startTime = System.currentTimeMillis();
        //prepareInputs(frame);
        long inputProcessStart = System.currentTimeMillis();
        /*Old process*/
        Mat frameCv = new Mat();//frame.getWidth(), frame.getHeight(), CvType.CV_8UC3);
        Bitmap frame32 = frame.copy(Bitmap.Config.ARGB_8888, true);//ismutable
        Utils.bitmapToMat(frame32, frameCv);//frame32, frameCv);
        Imgproc.cvtColor(frameCv , frameCv , 1);//COLOR_RGBA2RGB
        //Mat imgARgb = new Mat(frameCv.rows(), frameCv.cols(), CV_32FC1);
        frameCv.convertTo(frameCv, CvType.CV_32F);//, 1.0, 0);
        Core.subtract(frameCv, new Scalar(123.0f, 117.0f, 104.0f), frameCv);
        frameCv.get(0, 0, inputValues);

        //TODO: if not using quantized
        if (!isUsingQuantized){

            inputTensor.write(inputValues, 0, inputValues.length);
            inputs.put(mInputLayer, inputTensor);
        } else {
            /**/
            /*Tf8 Buffer*/
            //TensorUtils.prepareTf8Inputs(network, mInputLayer, inputTensors, inputBuffers, frameCv, inputValues);

            //TensorUtils.prepareTf8Outputs(network, mOutputLayer, outputTensors, outputBuffers);

        }
        long inputProcessTime = System.currentTimeMillis() - inputProcessStart;

        //long preProcessTime = System.currentTimeMillis();

        //Log.d(LOGTAG, "prepareInputs size " + inputs.entrySet().iterator().next().getValue().getSize());
        long modelExecutionStart = System.currentTimeMillis();
        if (!isUsingQuantized){
            final Map<String, FloatTensor> outputs = network.execute(inputs);
            long modelExecutionTime = System.currentTimeMillis() - modelExecutionStart;
            //Log.d(LOGTAG + "_checkTimeFd", "preprocess time = " + inputProcessTime + " | execute time = " + modelExecutionTime);

            long postProcessStart = System.currentTimeMillis();
            final List<Bbox> bboxes = convertOutputs(outputs);
            long postProcessTime = System.currentTimeMillis() - postProcessStart;

            Log.d(LOGTAG + "_checkTimeFd", "preprocess time = " + inputProcessTime + " | execute time = " + modelExecutionTime + " | post process time = " + postProcessTime);
            return bboxes;
        } else {

            network.execute(inputTensors, outputTensors);
            long modelExecutionTime = System.currentTimeMillis() - modelExecutionStart;
            Log.d(LOGTAG + "_checkTimeFd", "preprocess time = " + inputProcessTime + " | execute time = " + modelExecutionTime);
            final List<Bbox> bboxes = convertTf8Outputs();
            return bboxes;
        }


//        long preProcessRunTime = preProcessTime - startTime;
//        long dlcRunTime = System.currentTimeMillis() - preProcessTime;
        //Log.d(LOGTAG, "DLC: Frame processed preprocess in: " + preProcessRunTime +  " detect in: " + dlcRunTime);
        //Log.d(LOGTAG, "network executed");
        //long outputProcessStart = System.currentTimeMillis();
        //clear & releasr for next run
//        inputs.clear();
//        releaseTensors(outputs);
        //long outputProcessTime = System.currentTimeMillis() - outputProcessStart;
        //Log.d(LOGTAG + "_tf8_runtime", "prepareInput " + inputProcessTime + "ms |" + " execute " + modelExecutionTime + "ms |" +" post process: " + outputProcessTime + "ms");


    }

    private List<Bbox> buildBbox(float[] locs, float[] confidences, float[] landmarks)
    {
        final ArrayList<Bbox> bboxes = new ArrayList<Bbox>();
        ReadWriteLock lock = new ReentrantReadWriteLock();

        //int locIndex = 0;
        //int confIndex = 0;
        //int landmarkIndex = 0;

        //synchronized (bboxes) {
        for (int i = 0; i < anchors.size(); ++i)
        //IntStream.range(0, anchors.size() - 1).parallel().forEach(i ->
        {
            float cx = confidences[i * 2];
            float cy = confidences[i * 2 + 1];
            float conf = (float) (Math.exp(cy) / (Math.exp(cx) + Math.exp(cy)));
//            if (conf > CONFIDENCE_THRESHOLD) {
            if (conf > 0.8) {
                Anchor tmp = anchors.get(i);
                Anchor tmp1 = new Anchor();
                Bbox result = new Bbox();

                tmp1.cx = (float) (tmp.cx + locs[i * 4] * 0.1 * tmp.sx);
                tmp1.cy = (float) (tmp.cy + locs[i * 4 + 1] * 0.1 * tmp.sy);
                tmp1.sx = (float) (tmp.sx * Math.exp(locs[i * 4 + 2] * 0.2));
                tmp1.sy = (float) (tmp.sy * Math.exp(locs[i * 4 + 3] * 0.2));

                // Extract bbox and confidences
                result.x1 = (tmp1.cx - tmp1.sx / 2) * IMG_WIDTH;
                if (result.x1 < 0) {
                    result.x1 = 0;
                }
                result.y1 = (tmp1.cy - tmp1.sy / 2) * IMG_HEIGHT;
                if (result.y1 < 0) {
                    result.y1 = 0;
                }
                result.x2 = (tmp1.cx + tmp1.sx / 2) * IMG_WIDTH;
                if (result.x2 > IMG_WIDTH) {
                    result.x2 = IMG_WIDTH;
                }
                result.y2 = (tmp1.cy + tmp1.sy / 2) * IMG_HEIGHT;
                if (result.y2 > IMG_HEIGHT) {
                    result.y2 = IMG_HEIGHT;
                }
                result.conf = conf;

                // Skip extracting landmark
                for (int j = 0; j < 5; ++j) {
                    float lx = (tmp.cx + (landmarks[i * 10 + j * 2]) * 0.1f * tmp.sx) * IMG_WIDTH;
                    float ly = (tmp.cy + (landmarks[i * 10 + j * 2 + 1]) * 0.1f * tmp.sy) * IMG_HEIGHT;
                    result.landmarks[j] = new PointF(lx, ly);
                }

                //Check face probability/*
                /*
                Mat inputAlign = prepareAlignNative(mFrameCv, result);
                result.alignMat = inputAlign;
                Mat inputMat = new Mat(inputAlign.rows(), inputAlign.cols(), CV_32FC1);
                inputAlign.convertTo(inputMat, CvType.CV_32F, 1.0 / 255, 0);

                Core.subtract(inputMat, new Scalar(0.485f, 0.456f, 0.406f), inputMat);
                Core.divide(inputMat, new Scalar(0.229f, 0.224f, 0.225f), inputMat);
                long filterStart = System.currentTimeMillis();
                final float[] filterOutputs = mTrueFaceDetector.filterTrueFace(inputMat);
                long filterRunTime = System.currentTimeMillis() - filterStart;
                Log.d(LOGTAG, "DLC: Filtered true face in: " + filterRunTime);
                double faceProb = Math.exp(filterOutputs[0]) / (Math.exp(filterOutputs[0]) + Math.exp(filterOutputs[1]));
                float finalProb = (float)(0.6 * faceProb + 0.4 * conf);
                Log.d("FaceProb", "" + finalProb);
                if (finalProb >= TRUE_FACE_THRESHOLD) {
                    result.faceProb = finalProb;
                    //lock.writeLock().lock();
                    bboxes.add(result);
                    //lock.writeLock().unlock();
                }
                */
                bboxes.add(result);
            }
        }
        //});
        //}
        Collections.sort(bboxes);
        return bboxes;
    }

    private List<Bbox> nms(List<Bbox> bboxes) {
        List<Bbox> selected = new ArrayList<Bbox>();

        for (Bbox boxA : bboxes) {
            boolean shouldSelect = true;

            // Does the current box overlap one of the selected boxes more than the
            // given threshold amount? Then it's too similar, so don't keep it.
            for (Bbox boxB : selected) {
                if (IOU(boxA, boxB) > IOU_THRESHOLD) {
                    shouldSelect = false;
                    break;
                }
            }

            // This bounding box did not overlap too much with any previously selected
            // bounding box, so we'll keep it.
            if (shouldSelect) {
                selected.add(boxA);
            }
        }

        return selected;
    }

    private float IOU(Bbox a, Bbox b ) {
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        if (areaA <= 0) {
            return 0;
        }

        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        if (areaB <= 0) {
            return 0;
        }

        float intersectionMinX = Math.max(a.x1, b.x1);
        float intersectionMinY = Math.max(a.y1, b.y1);
        float intersectionMaxX = Math.min(a.x2, b.x2);
        float intersectionMaxY = Math.min(a.y2, b.y2);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    @SafeVarargs
    private final void releaseTensors(Map<String, ? extends Tensor>... tensorMaps) {
        for (Map<String, ? extends Tensor> tensorMap: tensorMaps) {
            for (Tensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }

    public void close() {
        network.release();
//        releaseTensors(inputs);
        releaseTf8Tensors(inputTensors, outputTensors);
    }

    private final void releaseTf8Tensors(Map<String, ? extends UserBufferTensor>... tensorMaps) {
        for (Map<String, ? extends UserBufferTensor> tensorMap: tensorMaps) {
            for (UserBufferTensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }
}
