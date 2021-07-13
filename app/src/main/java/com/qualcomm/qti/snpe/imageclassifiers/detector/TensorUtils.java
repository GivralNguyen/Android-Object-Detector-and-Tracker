package com.qualcomm.qti.snpe.imageclassifiers.detector;

import android.graphics.Bitmap;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.TF8UserBufferTensor;
import com.qualcomm.qti.snpe.TensorAttributes;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TensorUtils {
    private static class Tf8Params {
        int size;
        int[] strides;
        int stepExactly0;
        float stepSize;

        Tf8Params(int size, int[] strides) {
            this.size = size;
            this.strides = strides;
        }
    }

    private static class Tf8Encoding {
        float min;
        float max;
        float delta;
        float offset;
    }

    private static final int TF8_SIZE = 1;
    private static final int TF8_BITWIDTH = 8;
    private static final int mStepExactly0 = 0;
    private static final float mStepSize = 1.0f;

    public static boolean prepareFloatInputs(NeuralNetwork neuralNetwork, String inputLayer,
                                             FloatTensor inputTensor,
                                             final Map<String, FloatTensor> inputs,
                                             final Map<String, ByteBuffer> inputBuffers,
                                             float[] inputValues,
                                             Bitmap image) {
        final int[] dimensions = inputTensor.getShape();
        final boolean isGrayScale = (dimensions[dimensions.length - 1] == 1);

        Mat frameCv = new Mat();//frame.getWidth(), frame.getHeight(), CvType.CV_8UC3);
        Bitmap frame32 = image.copy(Bitmap.Config.ARGB_8888, true);//ismutable
        Utils.bitmapToMat(frame32, frameCv);//frame32, frameCv);
        Imgproc.cvtColor(frameCv , frameCv , 1);//COLOR_RGBA2RGB
        //Mat imgARgb = new Mat(frameCv.rows(), frameCv.cols(), CV_32FC1);
        frameCv.convertTo(frameCv, CvType.CV_32F);//, 1.0, 0);
        Core.subtract(frameCv, new Scalar(123.0f, 117.0f, 104.0f), frameCv);
        frameCv.get(0, 0, inputValues);

        inputTensor.write(inputValues, 0, inputValues.length);
        inputs.put(inputLayer, inputTensor);
        return true;
    }

    public static boolean prepareTf8InputsWithMat(NeuralNetwork neuralNetwork, String inputLayer,
                                                  final Map<String, TF8UserBufferTensor> inputTensors,
                                                  final Map<String, ByteBuffer> inputBuffers,
                                                  float[] inputValues, Bitmap image) {
        TensorAttributes inputAttributes = neuralNetwork.getTensorAttributes(inputLayer);
        Tf8Params inputParams = resolveTf8Params(inputAttributes);

        inputBuffers.put(inputLayer, ByteBuffer.allocateDirect(inputParams.size).order(ByteOrder.nativeOrder()));

        //Get float[]
        Mat frameCv = new Mat();//frame.getWidth(), frame.getHeight(), CvType.CV_8UC3);
        Bitmap frame32 = image.copy(Bitmap.Config.ARGB_8888, true);//ismutable
        Utils.bitmapToMat(frame32, frameCv);//frame32, frameCv);
        Imgproc.cvtColor(frameCv , frameCv , 1);//COLOR_RGBA2RGB
        frameCv.convertTo(frameCv, CvType.CV_32F);//, 1.0, 0);
        Core.subtract(frameCv, new Scalar(123.0f, 117.0f, 104.0f), frameCv);
        frameCv.get(0, 0, inputValues);

        quantizeWithMat(frameCv, inputValues, inputBuffers.get(inputLayer), inputParams);

        inputTensors.put(inputLayer, neuralNetwork.createTF8UserBufferTensor(
                inputParams.size, inputParams.strides,
                inputParams.stepExactly0, inputParams.stepSize,
                inputBuffers.get(inputLayer)
        ));
        return true;
    }

    public static boolean prepareTf8Inputs(NeuralNetwork neuralNetwork, String inputLayer,
                                           final Map<String, TF8UserBufferTensor> inputTensors,
                                           final Map<String, ByteBuffer> inputBuffers,
                                           float[] inputValues, Bitmap image) {
        TensorAttributes inputAttributes = neuralNetwork.getTensorAttributes(inputLayer);
        Tf8Params inputParams = resolveTf8Params(inputAttributes);

        inputBuffers.put(inputLayer, ByteBuffer.allocateDirect(inputParams.size).order(ByteOrder.nativeOrder()));

        //Get float[]
//        final int[] dimensions = inputAttributes.getDims();
//        final boolean isGrayScale = (dimensions[dimensions.length - 1] == 1);
//        if (!isGrayScale) {
//            inputValues = loadRgbBitmapAsFloat(image);
//        } else {
//            inputValues = loadGrayScaleBitmapAsFloat(image);
//        }

        //Get float[] with opencvMat
        Mat frameCv = new Mat();//frame.getWidth(), frame.getHeight(), CvType.CV_8UC3);
        Bitmap frame32 = image.copy(Bitmap.Config.ARGB_8888, true);//ismutable
        Utils.bitmapToMat(frame32, frameCv);//frame32, frameCv);
        Imgproc.cvtColor(frameCv , frameCv , 1);//COLOR_RGBA2RGB
        frameCv.convertTo(frameCv, CvType.CV_32F);//, 1.0, 0);
        Core.subtract(frameCv, new Scalar(123.0f, 117.0f, 104.0f), frameCv);
        frameCv.get(0, 0, inputValues);

        quantize(inputValues, inputBuffers.get(inputLayer), inputParams);

        inputTensors.put(inputLayer, neuralNetwork.createTF8UserBufferTensor(
                inputParams.size, inputParams.strides,
                inputParams.stepExactly0, inputParams.stepSize,
                inputBuffers.get(inputLayer)
        ));
        return true;
    }

    public static void prepareTf8Outputs(NeuralNetwork neuralNetwork, Set<String> outputLayers,
                                         final Map<String, TF8UserBufferTensor> outputTensors,
                                         final Map<String, ByteBuffer> outputBuffers) {
        Iterator<String> outputIterator = outputLayers.iterator();
        while (outputIterator.hasNext()) {
            String outputLayer = outputIterator.next();
            TensorAttributes outputAttributes = neuralNetwork.getTensorAttributes(outputLayer);
            Tf8Params outputParams = resolveTf8Params(outputAttributes);
            outputParams.stepExactly0 = mStepExactly0;
            outputParams.stepSize = mStepSize;

            outputBuffers.put(outputLayer, ByteBuffer.allocateDirect(outputParams.size).order(ByteOrder.nativeOrder()));
            try {
                outputTensors.put(outputLayer, neuralNetwork.createTF8UserBufferTensor(
                        outputParams.size, outputParams.strides,
                        outputParams.stepExactly0, outputParams.stepSize,
                        outputBuffers.get(outputLayer)));
            } catch (Exception e) {
                Log.d("HeadDetectorException", "" + e.getMessage());
            }
        }
    }

    public static float[] dequantize(TF8UserBufferTensor tensor, ByteBuffer buffer) {
        final int outputSize = buffer.capacity();
        final byte[] quantizedArray = new byte[outputSize];
        buffer.get(quantizedArray);

        final float[] dequantizedArray = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            int quantizedValue = (int)quantizedArray[i] & 0xFF;
            dequantizedArray[i] = tensor.getMin() + quantizedValue *  tensor.getQuantizedStepSize();
        }

        return dequantizedArray;
    }

    private static Tf8Params resolveTf8Params(TensorAttributes attribute) {
        int rank = attribute.getDims().length;
        int[] strides = new int[rank];
        strides[rank - 1] = TF8_SIZE;
        for (int i = rank - 1; i > 0; i--) {
            strides[i-1] = strides[i] * attribute.getDims()[i];
        }

        int bufferSize = TF8_SIZE;
        for (int dim: attribute.getDims()) {
            bufferSize *= dim;
        }

        return new Tf8Params(bufferSize, strides);
    }

    //Load rgb image as float
    static float[] loadRgbBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length * 3];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;
                final int batchIdx = idx * 3;

                final float[] rgb = extractColorChannels(pixels[idx]);
                pixelsBatched[batchIdx]     = rgb[0];
                pixelsBatched[batchIdx + 1] = rgb[1];
                pixelsBatched[batchIdx + 2] = rgb[2];
            }
        }
        return pixelsBatched;
    }

    //Load gray scale bitmap as float
    static float[] loadGrayScaleBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;

                final int rgb = pixels[idx];
                final float b = ((rgb)       & 0xFF);
                final float g = ((rgb >>  8) & 0xFF);
                final float r = ((rgb >> 16) & 0xFF);
                float grayscale = (float) (r * 0.3 + g * 0.59 + b * 0.11);

                pixelsBatched[idx] = preProcess(grayscale);
            }
        }
        return pixelsBatched;
    }

    private static float[] extractColorChannels(int pixel) {
        float b = ((pixel)       & 0xFF);
        float g = ((pixel >>  8) & 0xFF);
        float r = ((pixel >> 16) & 0xFF);
        return new float[] {preProcess(r), preProcess(g), preProcess(b)};
    }

    private static float preProcess(float original) {
        return (original - 128) / 128;
    }

    public static void quantizeWithMat(Mat cvMat, float[] src, ByteBuffer dst, Tf8Params tf8Params) {
        Tf8Encoding encoding = getTf8EncodingWithMat(cvMat);

        byte[] quantized = new byte[src.length];
        for (int i = 0; i < src.length; i++) {
            float data = Math.max(Math.min(src[i], encoding.max), encoding.min);
            data = data / encoding.delta - encoding.offset;
            quantized[i] = (byte) Math.round(data);
        }

        dst.put(quantized);
        tf8Params.stepSize = encoding.delta;
        tf8Params.stepExactly0 = Math.round(-encoding.min / encoding.delta);
    }

    public static void quantize(float[] src, ByteBuffer dst, Tf8Params tf8Params) {
        Tf8Encoding encoding = getTf8Encoding(src);

        byte[] quantized = new byte[src.length];
        for (int i = 0; i < src.length; i++) {
            float data = Math.max(Math.min(src[i], encoding.max), encoding.min);
            data = data / encoding.delta - encoding.offset;
            quantized[i] = (byte) Math.round(data);
        }

        dst.put(quantized);
        tf8Params.stepSize = encoding.delta;
        tf8Params.stepExactly0 = Math.round(-encoding.min / encoding.delta);
    }

    private static Tf8Encoding getTf8EncodingWithMat(Mat frameCv) {
        Tf8Encoding encoding = new Tf8Encoding();

        int num_steps = (int) Math.pow(2, TF8_BITWIDTH) - 1;
        List<Mat> rgb = new ArrayList<>();
        Float[] maxVals = new Float[3];
        Float[] minVals = new Float[3];
        Core.split(frameCv, rgb);

        int i = 0;
        for (Mat mat : rgb) {
            Core.MinMaxLocResult result = Core.minMaxLoc(mat);
            maxVals[i] = (float) result.maxVal;
            minVals[i] = (float) result.minVal;
            i++;
        }

        float newMax = Math.max(Collections.max(Arrays.asList(maxVals)), 0);
        float newMin = Math.min(Collections.min(Arrays.asList(minVals)), 0);

        float minRange = 0.1f;
        newMax = Math.max(newMax, newMin + minRange);
        encoding.delta = (newMax - newMin) / num_steps;

        if (newMin < 0 && newMax > 0) {
            float quantizedZero = Math.round(-newMin / encoding.delta);
            quantizedZero = (float) Math.min(num_steps, Math.max(0.0, quantizedZero));
            encoding.offset = -quantizedZero;
        } else {
            encoding.offset = Math.round(newMin / encoding.delta);
        }

        encoding.min = encoding.delta * encoding.offset;
        encoding.max = encoding.delta * num_steps + encoding.min;

        return encoding;
    }

    private static Tf8Encoding getTf8Encoding(float[] array) {
        Tf8Encoding encoding = new Tf8Encoding();

        int num_steps = (int) Math.pow(2, TF8_BITWIDTH) - 1;
        float new_min = Math.min(getMin(array), 0);
        float new_max = Math.max(getMax(array), 0);

        float min_range = 0.1f;
        new_max = Math.max(new_max, new_min + min_range);
        encoding.delta = (new_max - new_min) / num_steps;

        if (new_min < 0 && new_max > 0) {
            float quantized_zero = Math.round(-new_min / encoding.delta);
            quantized_zero = (float) Math.min(num_steps, Math.max(0.0, quantized_zero));
            encoding.offset = -quantized_zero;
        } else {
            encoding.offset = Math.round(new_min / encoding.delta);
        }

        encoding.min = encoding.delta * encoding.offset;
        encoding.max = encoding.delta * num_steps + encoding.min;

        return encoding;
    }

    static float getMin(float[] array) {
        float min = Float.MAX_VALUE;
        for (float value : array) {
            if (value < min) {
                min = value;
            }
        }
        return min;
    }

    static float getMax(float[] array) {
        float max = Float.MIN_VALUE;
        for (float value : array) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }

}
