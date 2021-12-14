package com.qualcomm.qti.snpe.imageclassifiers.detector;


import android.app.Application;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.TF8UserBufferTensor;
import com.qualcomm.qti.snpe.Tensor;
import com.qualcomm.qti.snpe.UserBufferTensor;
import com.qualcomm.qti.snpe.imageclassifiers.thread.PreprocessResult;

import org.apache.commons.lang3.ArrayUtils;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;


public class TFMobilenetQuantizeDetector {
    static final String LOGTAG = TFMobilenetQuantizeDetector.class.getSimpleName();
    /**Model size**/
    public static final int  MODEL_WIDTH = 300;
    public static final int MODEL_HEIGHT = 300;

    public boolean isUsingQuantized = false;
    NeuralNetwork network = null;

    // Prepare input buffer
    String mInputLayer = "";
    Set<String> mOutputLayer;

    private FloatTensor inputTensor = null;
    private Map<String, TF8UserBufferTensor> inputTensors = new HashMap<>();
    private Map<String, TF8UserBufferTensor> outputTensors = new HashMap<>();
    private float mRatioWidth;
    private float mRatioHeight;
    private float[] inputValues = new float[MODEL_WIDTH * MODEL_HEIGHT * 3];
    private Map<String, FloatTensor> inputs = new HashMap<>();

    public TFMobilenetQuantizeDetector(
            Context context,
            Application application,
            int modelRes
    ) {

        final Resources res = context.getResources();
        final InputStream modelInputStream = res.openRawResource(modelRes);
        try {
            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(application)
                    .setDebugEnabled(false)
                    .setRuntimeOrder(
//                            NeuralNetwork.Runtime.AIP,
                            NeuralNetwork.Runtime.DSP,
                            NeuralNetwork.Runtime.GPU_FLOAT16,
                            NeuralNetwork.Runtime.GPU,
                            NeuralNetwork.Runtime.CPU
                    )
                    .setModel(modelInputStream, modelInputStream.available())
                    .setOutputLayers("Postprocessor/BatchMultiClassNonMaxSuppression")
                    .setCpuFallbackEnabled(true)
                    .setUseUserSuppliedBuffers(isUsingQuantized)
                    .setPerformanceProfile(NeuralNetwork.PerformanceProfile.BURST);
            network = builder.build();

            // Prepare inputs buffer
            mInputLayer = network.getInputTensorsNames().iterator().next();
            mOutputLayer = network.getOutputTensorsNames();
            inputTensor = network.createFloatTensor(network.getInputTensorsShapes().get(mInputLayer));
            Log.d(LOGTAG, "TF Mobilenet Quantized Detector initiated " + network.getInputTensorsShapes().entrySet().iterator().next().getValue().length);
        } catch (IOException e) {
            // Do something here
        }
    }


    public List<float[]> detectFrame(PreprocessResult preprocessResult) {

        Bitmap frame = preprocessResult.getFrame();
        inputValues = preprocessResult.getInputs();


        /**Convert to FloatTensor**/
        long convertTensorStart = System.currentTimeMillis();
        if (!isUsingQuantized){
            inputTensor.write(inputValues, 0, inputValues.length);
            inputs.put(mInputLayer, inputTensor);
        } else {
        }

        /**Execute model**/

        long modelExecutionStart = System.currentTimeMillis();
        final Map<String, FloatTensor> outputs = network.execute(inputs);
        long modelExecutionTime = System.currentTimeMillis() - modelExecutionStart;
        Log.d(LOGTAG,"model_Execute: "+ modelExecutionTime);
        /**Execute model**/
        List<float[]> detectList = convertOutputs(outputs,frame);

        return detectList;
    }

    private List<float[]> convertOutputs(Map<String, FloatTensor> outputs,Bitmap frame) {
        float MIN_CONF = 0.3F;
        float[] boxes = {};
        float[] scores = {};
        float[] classes = {};
        float[] boxes_selected = {};
        float[] scores_selected = {};
        float[] classes_selected = {};
        List<float[]> detectList = new ArrayList<>();
        for (Map.Entry<String, FloatTensor> output : outputs.entrySet()) {
            FloatTensor outputTensor = output.getValue();
            switch (output.getKey()) {
                case "Postprocessor/BatchMultiClassNonMaxSuppression_boxes":
                    boxes = new float[outputTensor.getSize()];

                    outputTensor.read(boxes, 0, boxes.length);
//                    Log.d(LOGTAG,"locations" + Arrays.toString(locations));
                    break;
                case "Postprocessor/BatchMultiClassNonMaxSuppression_scores":
                    scores = new float[outputTensor.getSize()];

                    outputTensor.read(scores, 0, scores.length);
//                    Log.d(LOGTAG,"confidences" + Arrays.toString(confidences));
                    break;
                case "Postprocessor/BatchMultiClassNonMaxSuppression_classes":
                    classes = new float[outputTensor.getSize()];

                    outputTensor.read(classes, 0, classes.length);
//                    Log.d(LOGTAG,"locations" + Arrays.toString(locations));
                    break;
            }
        }
        for (int i = 0 ; i < scores.length; i++)
        {
            if (scores[i] < MIN_CONF){
                break;
            }
            else {
                scores_selected = ArrayUtils.add(scores_selected,scores[i]);
                boxes_selected = ArrayUtils.add(boxes_selected,Math.max(1,(boxes[i*4]*frame.getHeight())));
                boxes_selected = ArrayUtils.add(boxes_selected,Math.max(1,(boxes[i*4+1]*frame.getWidth())));
                boxes_selected = ArrayUtils.add(boxes_selected,Math.min(frame.getHeight(),(boxes[i*4+2]*frame.getHeight())));
                boxes_selected = ArrayUtils.add(boxes_selected,Math.min(frame.getWidth(),(boxes[i*4+3]*frame.getWidth())));
                classes_selected = ArrayUtils.add(classes_selected,classes[i]);
            }
        }
        detectList.add(scores_selected);
        detectList.add(classes_selected);
        detectList.add(boxes_selected);
        return detectList;
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



