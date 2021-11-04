package com.qualcomm.qti.snpe.imageclassifiers.detector;


import android.app.Application;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.TF8UserBufferTensor;
import com.qualcomm.qti.snpe.Tensor;
import com.qualcomm.qti.snpe.UserBufferTensor;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MobilenetDetector {
    static final String LOGTAG = MobilenetDetector.class.getSimpleName();
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
    public static final List<Anchor> anchors = new ArrayList<Anchor>();
    private float IOU_THRESHOLD = (float) 0.35;

    public MobilenetDetector(
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
                            NeuralNetwork.Runtime.AIP,
                            NeuralNetwork.Runtime.DSP,
                            NeuralNetwork.Runtime.GPU_FLOAT16,
                            NeuralNetwork.Runtime.GPU,
                            NeuralNetwork.Runtime.CPU
                    )
                    .setModel(modelInputStream, modelInputStream.available())
                    .setOutputLayers("concatenation_0",
                            "concatenation_1")
                    .setCpuFallbackEnabled(true)
                    .setUseUserSuppliedBuffers(isUsingQuantized)
                    .setPerformanceProfile(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE);
            network = builder.build();

            // Prepare inputs buffer
            mInputLayer = network.getInputTensorsNames().iterator().next();
            mOutputLayer = network.getOutputTensorsNames();
            inputTensor = network.createFloatTensor(network.getInputTensorsShapes().get(mInputLayer));
            createAnchor();
            Log.d(LOGTAG, "Mobilenet Detector initiated " + network.getInputTensorsShapes().entrySet().iterator().next().getValue().length);
        } catch (IOException e) {
            // Do something here
        }
    }


    public List<Bbox> detectFrame(Bitmap frame) {

        mRatioWidth = (float) (MODEL_WIDTH / (frame.getWidth() * 1.0));
        mRatioHeight =  (float)(MODEL_HEIGHT / (frame.getHeight()* 1.0));

        /**Preprocessing**/
        long preProcessStart = System.currentTimeMillis();

        /**Resizing Bitmap**/
        final float scaleX = MODEL_WIDTH / (float) (frame.getWidth());
        final float scaleY = MODEL_HEIGHT / (float) (frame.getHeight());
        final Matrix scalingMatrix = new Matrix();
        scalingMatrix.postScale(scaleX, scaleY);
        Bitmap resizedBitmap = Bitmap.createBitmap(frame,
                0, 0,
                frame.getWidth(), frame.getHeight(),scalingMatrix,false);
        Mat frameCv = new Mat();//frame.getWidth(), frame.getHeight(), CvType.CV_8UC3);
        Bitmap frame32 = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true);//ismutable
        Utils.bitmapToMat(frame32, frameCv);//frame32, frameCv);
        Imgproc.cvtColor(frameCv , frameCv , Imgproc.COLOR_RGBA2RGB);//COLOR_RGBA2RGB
        frameCv.convertTo(frameCv, CvType.CV_32F);//, 1.0, 0); //convert to 32F
        Core.subtract(frameCv, new Scalar(127.0f, 127.0f, 127.0f), frameCv);
        Core.divide(frameCv, new Scalar(128.0f, 128.0f, 128.0f), frameCv);
        frameCv.get(0, 0, inputValues); //image.astype(np.float32)
        /**Resizing Bitmap**/

        /**Resizing opencv**/
//        Mat frameCv = new Mat();
//        Bitmap frame32 = frame.copy(Bitmap.Config.ARGB_8888, true);
//        Utils.bitmapToMat(frame32, frameCv);
//        Mat resizeimage = new Mat();
//        Size sz = new Size(MODEL_WIDTH,MODEL_HEIGHT);
//        Imgproc.resize( frameCv, resizeimage, sz );
//        Imgproc.cvtColor(resizeimage , resizeimage , Imgproc.COLOR_RGBA2RGB);//COLOR_RGBA2RGB
//        resizeimage.convertTo(resizeimage, CvType.CV_32F);//, 1.0, 0); //convert to 32F
//        Core.subtract(resizeimage, new Scalar(127.0f, 127.0f, 127.0f), resizeimage);
//        Core.divide(resizeimage, new Scalar(128.0f, 128.0f, 128.0f), resizeimage);
//        resizeimage.get(0, 0, inputValues); //image.astype(np.float32)
        /**Resizing opencv**/


        long preProcessTime = System.currentTimeMillis()- preProcessStart;
        Log.d(LOGTAG,"Preprocess_time: "+ preProcessTime);
        /**Preprocessing**/

        /**Convert to FloatTensor**/
        long convertTensorStart = System.currentTimeMillis();
        if (!isUsingQuantized){
            inputTensor.write(inputValues, 0, inputValues.length);
            inputs.put(mInputLayer, inputTensor);
        } else {
        }
        long convertTensorTime = System.currentTimeMillis()- convertTensorStart;
        Log.d(LOGTAG,"convertTensor_time: "+ convertTensorTime);
        /**Convert to FloatTensor**/

        /**Execute model**/
        long modelExecutionStart = System.currentTimeMillis();
        final Map<String, FloatTensor> outputs = network.execute(inputs);
        long modelExecutionTime = System.currentTimeMillis() - modelExecutionStart;
        Log.d(LOGTAG,"model_Execute: "+ modelExecutionTime);
        /**Execute model**/

        /**Anchor**/
        long anchorStart = System.currentTimeMillis();
        List<float[]> detectList = convertOutputs(outputs);
        float[] locations = detectList.get(0);
        float[] confidences = detectList.get(1);
        List<Bbox> bboxes = buildBbox(confidences, locations);
        long anchorTime = System.currentTimeMillis() - anchorStart;
        Log.d(LOGTAG,"anchorExecute: "+ anchorTime);
        /**Anchor**/

        /**NMS**/
        long NMSstart = System.currentTimeMillis();
        bboxes = nms(bboxes);
        long NMStime = System.currentTimeMillis() - NMSstart;
        Log.d(LOGTAG,"NMS : "+ NMStime);
        /**NMS**/
        return bboxes;
    }

    private List<float[]> convertOutputs(Map<String, FloatTensor> outputs) {
        float[] locations = {};
        float[] confidences = {};
        List<float[]> detectList = new ArrayList<>();
        for (Map.Entry<String, FloatTensor> output : outputs.entrySet()) {
            FloatTensor outputTensor = output.getValue();
            switch (output.getKey()) {
                case "locations":
                    locations = new float[outputTensor.getSize()];

                    outputTensor.read(locations, 0, locations.length);
//                    Log.d(LOGTAG,"locations" + Arrays.toString(locations));
                    break;
                case "confidences":
                    confidences = new float[outputTensor.getSize()];

                    outputTensor.read(confidences, 0, confidences.length);
//                    Log.d(LOGTAG,"confidences" + Arrays.toString(confidences));
                    break;
            }
        }
        detectList.add(locations);
        detectList.add(confidences);

        return detectList;
    }



    private void createAnchor() {
        float[] feature_map_sizes = {19, 10, 5, 3, 2, 1};
        float[] shrinkage = {16, 32, 64, 100, 150, 300};
        float[][] box_sizes = {{60, 105}, {105, 150}, {150, 195}, {195, 240}, {240, 285}, {285, 330}};
        float[] ratios = {2.0f, 3.0f};
        float image_size = 300;
        float[] priors = {};

        for (int index=0; index< feature_map_sizes.length ; index++ )
        {
            float scale = image_size / shrinkage[index];
            for (int j = 0; j < feature_map_sizes[index]; j++) {
                for (int i = 0; i < feature_map_sizes[index]; i++) {
                    float x_center = (float) (i + 0.5) / scale;
                    float y_center = (float) (j + 0.5) / scale;

                    float size1 = box_sizes[index][0];
                    float h1 = size1 / image_size;
                    float w1 = size1 / image_size;
                    final Anchor anchor1 = new Anchor(x_center, y_center, w1, h1);
                    anchors.add(anchor1);
                    float size2 = (float) Math.sqrt(box_sizes[index][0] * box_sizes[index][1]);
                    float h2 = size2 / image_size;
                    float w2 = size2 / image_size;
                    final Anchor anchor2 = new Anchor(x_center, y_center, w2, h2 );
                    anchors.add(anchor2);


                    for (float ratio: ratios) {
                        float ratio_sqrt = (float) Math.sqrt(ratio);
                        final Anchor anchor3 = new Anchor(x_center, y_center, w1*ratio_sqrt, h1 / ratio_sqrt);
                        anchors.add(anchor3);
                        final Anchor anchor4 = new Anchor(x_center, y_center, w1/ratio_sqrt, h1 * ratio_sqrt);
                        anchors.add(anchor4);
                    }
                }
            }
        }

    }

    public RectF translate(final RectF location) {
        //Log.d(LOGTAG,"During translate: " + mRatioWidth + " " + mRatioHeight);
        return new RectF((location.left / mRatioWidth),
                (location.top / mRatioHeight),
                (location.right / mRatioWidth),
                (location.bottom / mRatioHeight));
    }

    private List<Bbox> buildBbox(float[] scores, float[] boxes) {
//        final ArrayList<Recognition> bboxes = new ArrayList<Recognition>();
        final ArrayList<Bbox> bboxes_ = new ArrayList<Bbox>();
        {
            for (int i = 0; i < anchors.size(); ++i) {
                float cx = scores[i * 21];
                float c_car = scores[i * 21 + 7];
                float c_bicycle = scores[i * 21 + 2];
                float c_bus = scores[i * 21 + 6];
                float c_motorbike = scores[i * 21 + 14];
                float c_person = scores[i * 21 + 15];
                float sum_of_exp = 0 ;
                for (int j = 0;j < 21;j++){
                    sum_of_exp += (float) Math.exp(scores[i * 21 + j]);
                }
                List confidences = new ArrayList<Float>();
                confidences.add((float) (Math.exp(c_car))/sum_of_exp);
                confidences.add((float) (Math.exp(c_bicycle))/sum_of_exp);
                confidences.add((float) (Math.exp(c_bus))/sum_of_exp);
                confidences.add((float) (Math.exp(c_motorbike))/sum_of_exp);
                confidences.add((float) (Math.exp(c_person))/sum_of_exp);
                float confidenceMax = (float) Collections.max(confidences);
                int labelId = confidences.indexOf(confidenceMax);
                if (confidenceMax > 0.3) {
                    Anchor tmp = anchors.get(i);
                    Anchor tmp1 = new Anchor();
//                    Recognition result = new Recognition();
                    Bbox result_ = new Bbox();
                    tmp1.cx = (float) (tmp.cx + boxes[i * 4] * 0.1 * tmp.sx);
                    tmp1.cy = (float) (tmp.cy + boxes[i * 4 + 1] * 0.1 * tmp.sy);
                    tmp1.sx = (float) (tmp.sx * Math.exp(boxes[i * 4 + 2] * 0.2));
                    tmp1.sy = (float) (tmp.sy * Math.exp(boxes[i * 4 + 3] * 0.2));

                    // Extract bbox and confidences
                    float x1 = (tmp1.cx - tmp1.sx / 2) * MODEL_WIDTH;//result
                    if (x1 < 0) x1 = 0;

                    float y1 = (tmp1.cy - tmp1.sy / 2) * MODEL_HEIGHT;
                    if (y1 < 0) y1 = 0;

                    float x2 = (tmp1.cx + tmp1.sx / 2) * MODEL_WIDTH;
                    if (x2 > MODEL_WIDTH) x2 = MODEL_WIDTH;

                    float y2 = (tmp1.cy + tmp1.sy / 2) * MODEL_HEIGHT;
                    if (y2 > MODEL_HEIGHT) y2 = MODEL_HEIGHT;

                    RectF loc = new RectF(x1, y1, x2, y2);
                    loc = translate(loc);
//                    result.mLocation = loc;
//                    //translate before add
//                    result.mConfidenceX = cx;//conf
//                    result.mConfidenceY = cy;

                    result_.x1 = loc.left;
                    result_.y1 = loc.top;
                    result_.x2 = loc.right;
                    result_.y2 = loc.bottom;
                    result_.conf = confidenceMax;
                    result_.label = labelId;

                    bboxes_.add(result_);
//                    bboxes.add(result);
                }
            }
        }

//        Comparator<Recognition> boxComparator = new Comparator<Recognition>() {
//            @Override
//            public int compare(Recognition box1, Recognition box2) {
//                return (box1.getConfidence() > box2.getConfidence() ? 1 : 0);
//            }
//        };
        Comparator<Bbox> boxComparator_ = new Comparator<Bbox>() {
            @Override
            public int compare(Bbox box1, Bbox box2) {
                return (box1.getConfidence() > box2.getConfidence() ? 1 : 0);
            }
        };

//        Collections.sort(bboxes, boxComparator);
        Collections.sort(bboxes_, boxComparator_);

        return bboxes_;
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



